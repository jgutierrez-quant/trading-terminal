"""
Market screener — S&P 500 universe, top movers, unusual volume, anomaly scan.

Public API:
    get_stock_universe()                          -> list[str]
    get_top_movers(n=20)                          -> dict
    get_unusual_volume(n=20)                      -> list[dict]
    run_market_scan(movers, unusual, max=50)      -> list[dict]

Note: run_market_scan is for standalone / test use.  The dashboard's
cached_market_scan() in app.py drives the scan using cached_technicals
to avoid redundant yfinance calls.
"""

import logging
import re

import requests
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup

from data.technicals import get_technicals
from data.anomaly_detector import compute_anomaly

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://finviz.com/",
}
_BASE_URL = "https://finviz.com/screener.ashx"
_TIMEOUT = 12

# 100-ticker S&P 500 large-cap fallback used when Wikipedia / Finviz is unavailable
_SP500_FALLBACK = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "BRK-B",
    "JPM", "UNH", "XOM", "LLY", "JNJ", "V", "PG", "MA", "HD", "CVX",
    "MRK", "ABBV", "AVGO", "COST", "PEP", "KO", "WMT", "MCD", "CRM",
    "NFLX", "ACN", "TMO", "LIN", "ABT", "ORCL", "DHR", "BAC", "AMD",
    "TXN", "QCOM", "NEE", "PM", "RTX", "HON", "LOW", "UPS", "IBM",
    "CAT", "GE", "DE", "BA", "PYPL", "AMGN", "GILD", "REGN", "VRTX",
    "ISRG", "MDT", "BSX", "ELV", "CI", "HUM", "CVS", "AXP", "GS",
    "MS", "BLK", "SCHW", "C", "WFC", "USB", "PNC", "TFC", "BK",
    "MMC", "AON", "CB", "ALL", "MET", "PRU", "AFL", "TRV",
    "SPG", "PLD", "AMT", "CCI", "DLR", "EQIX", "PSA",
    "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL",
    "WMB", "KMI", "OKE", "EPD", "ENPH", "FSLR",
]
# Deduplicate while preserving order
_seen: set = set()
_SP500_FALLBACK_DEDUP = []
for _t in _SP500_FALLBACK:
    if _t not in _seen:
        _seen.add(_t)
        _SP500_FALLBACK_DEDUP.append(_t)
_SP500_FALLBACK = _SP500_FALLBACK_DEDUP[:100]
del _seen, _t, _SP500_FALLBACK_DEDUP


# ── Public API ────────────────────────────────────────────────────────────────

def get_stock_universe() -> list:
    """
    Fetch the S&P 500 ticker list from Wikipedia.
    Falls back to a 100-ticker hardcoded list if the page is unreachable.
    Dots replaced with dashes (BRK.B → BRK-B) for yfinance compatibility.
    """
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            attrs={"id": "constituents"},
        )
        df = tables[0]
        raw = df["Symbol"].tolist()
        return [str(t).replace(".", "-").strip() for t in raw if t]
    except Exception as exc:
        logger.warning("get_stock_universe Wikipedia failed (%s) — using fallback", exc)
        return _SP500_FALLBACK[:]


def get_top_movers(n: int = 20) -> dict:
    """
    Scrape Finviz for today's top gainers and top losers.
    Falls back to yfinance batch download if Finviz returns non-200.

    Returns:
        {
            "gainers": list[dict],
            "losers":  list[dict],
            "error":   str | None,   # None when Finviz succeeded
        }
    Each ticker dict: {ticker, company, price, change_pct, volume}
    """
    gainers = _fetch_screener("ta_topgainers", n)
    losers  = _fetch_screener("ta_toplosers",  n)
    if gainers or losers:
        return {"gainers": gainers, "losers": losers, "error": None}

    logger.warning("Finviz unavailable for top movers — using yfinance fallback")
    return _yf_movers_fallback(n)


def get_unusual_volume(n: int = 20) -> list:
    """
    Scrape Finviz for stocks trading at unusual volume (3x+ average).
    Falls back to an empty list if Finviz is unavailable.

    Returns list of {ticker, company, price, change_pct, volume}.
    """
    result = _fetch_screener("ta_unusualvolume", n)
    if result:
        return result
    logger.warning("Finviz unavailable for unusual volume — returning empty")
    return []


def run_market_scan(
    movers: dict = None,
    unusual: list = None,
    max_tickers: int = 50,
) -> list:
    """
    Run anomaly detection across a combined ticker universe.
    Uses mock neutral sentiment to isolate technical signals (same as test_technicals.py).
    Only returns tickers flagged as Watch (score >= 3 signals).

    Args:
        movers:      pre-fetched result of get_top_movers(), or None to auto-fetch.
        unusual:     pre-fetched result of get_unusual_volume(), or None to auto-fetch.
        max_tickers: cap on universe size scanned.

    Returns:
        list of anomaly dicts sorted by score descending, is_watch=True only.
        Each dict: {ticker, score, is_watch, reason, signals, price, change_pct}
    """
    if movers is None:
        movers = get_top_movers()
    if unusual is None:
        unusual = get_unusual_volume()

    # Price lookup for movers/unusual (may be missing for base universe tickers)
    price_map = {
        r["ticker"]: {"price": r.get("price"), "change_pct": r.get("change_pct")}
        for r in (movers.get("gainers", []) + movers.get("losers", []) + unusual)
    }

    # Build deduplicated universe: movers first, then unusual, then S&P 500 base
    mover_list   = [r["ticker"] for r in (movers.get("gainers", []) + movers.get("losers", []))]
    unusual_list = [r["ticker"] for r in unusual]
    seen: set = set()
    universe: list = []
    for t in mover_list + unusual_list + get_stock_universe():
        if t not in seen:
            seen.add(t)
            universe.append(t)
        if len(universe) >= max_tickers:
            break

    mock_sent = {"sentiment_label": "Neutral", "overall_sentiment": 0.0}
    results = []
    for ticker in universe:
        try:
            tech = get_technicals(ticker)
            if tech.get("error"):
                continue
            anomaly = compute_anomaly(ticker, tech, mock_sent)
            if not anomaly.get("is_watch"):
                continue
            pi = price_map.get(ticker, {})
            row = dict(anomaly)
            row["price"]      = pi.get("price") or tech.get("current_price")
            row["change_pct"] = pi.get("change_pct")
            results.append(row)
        except Exception as exc:
            logger.warning("Scan failed for %s: %s", ticker, exc)

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fetch_screener(signal: str, n: int) -> list:
    """GET one Finviz screener signal page and parse it."""
    url = f"{_BASE_URL}?v=111&s={signal}"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        if resp.status_code != 200:
            logger.warning("Finviz screener %s → HTTP %d", signal, resp.status_code)
            return []
        return _parse_screener_html(resp.text, n)
    except Exception as exc:
        logger.error("Finviz screener fetch failed (%s): %s", signal, exc)
        return []


def _parse_screener_html(html: str, n: int) -> list:
    """
    Parse Finviz screener HTML into a list of ticker dicts.

    Strategy: every cell in a screener result row links to the same
    quote page (quote.ashx?t=TICKER). We extract the ticker symbol from
    the href, de-duplicate rows by ticker, then read cell positions:
      v=111 (Overview): No[0], Ticker[1], Company[2], Sector[3], Industry[4],
                        Country[5], Mkt Cap[6], P/E[7], Price[8], Change[9], Volume[10]
    Confirmed live against https://finviz.com/screener.ashx?v=111 (2026-03-01).
    """
    soup    = BeautifulSoup(html, "lxml")
    results = []
    seen: set = set()

    for a in soup.find_all("a", href=lambda h: h and "quote.ashx?t=" in h):
        href = a.get("href", "")
        m = re.search(r"quote\.ashx\?t=([A-Z\-]+)", href)
        if not m:
            continue
        ticker = m.group(1)
        if ticker in seen:
            continue

        row = a.find_parent("tr")
        if not row:
            continue
        cells = row.find_all("td")
        if len(cells) < 11:
            continue

        seen.add(ticker)
        results.append({
            "ticker":     ticker,
            "company":    cells[2].get_text(strip=True),
            "price":      _safe_float(cells[8].get_text()),
            "change_pct": _safe_pct(cells[9].get_text()),
            "volume":     _safe_int(cells[10].get_text()),
        })
        if len(results) >= n:
            break

    if not results:
        logger.warning("Finviz: parsed 0 rows from screener HTML (structure may have changed)")
    return results


def _yf_movers_fallback(n: int) -> dict:
    """
    Compute today's top movers via yfinance batch download of the fallback universe.
    Called only when Finviz is blocked.
    """
    try:
        universe = _SP500_FALLBACK[:60]
        symbols  = " ".join(universe)
        df = yf.download(symbols, period="2d", interval="1d",
                         auto_adjust=True, progress=False)
        if df.empty or len(df) < 2:
            return {"gainers": [], "losers": [], "error": "yfinance also unavailable"}

        close = df["Close"]
        vol   = df["Volume"]
        rows  = []
        for t in universe:
            try:
                closes = close[t].dropna() if isinstance(close, pd.DataFrame) else close.dropna()
                vols   = vol[t].dropna()   if isinstance(vol,   pd.DataFrame) else vol.dropna()
                if len(closes) < 2:
                    continue
                p   = float(closes.iloc[-1])
                pc  = float(closes.iloc[-2])
                chg = round((p - pc) / pc * 100, 2) if pc else None
                v   = int(vols.iloc[-1]) if len(vols) > 0 else None
                rows.append({
                    "ticker":     t,
                    "company":    "",
                    "price":      round(p, 2),
                    "change_pct": chg,
                    "volume":     v,
                })
            except Exception:
                continue

        rows = [r for r in rows if r["change_pct"] is not None]
        rows.sort(key=lambda r: r["change_pct"], reverse=True)
        return {
            "gainers": rows[:n],
            "losers":  rows[-n:][::-1],
            "error":   "Finviz unavailable — yfinance fallback used",
        }
    except Exception as exc:
        logger.error("yfinance movers fallback failed: %s", exc)
        return {"gainers": [], "losers": [], "error": str(exc)}


# ── Parsing utilities ─────────────────────────────────────────────────────────

def _safe_float(s: str):
    try:
        return float(str(s).replace("$", "").replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def _safe_pct(s: str):
    """'+8.52%' → 8.52,  '-1.23%' → -1.23."""
    try:
        return float(str(s).replace("%", "").replace("+", "").strip())
    except (ValueError, AttributeError):
        return None


def _safe_int(s: str):
    """'1,234,567' or '1.23M' or '987K' → int."""
    s = str(s).strip()
    try:
        if s.upper().endswith("M"):
            return int(float(s[:-1]) * 1_000_000)
        if s.upper().endswith("K"):
            return int(float(s[:-1]) * 1_000)
        if s.upper().endswith("B"):
            return int(float(s[:-1]) * 1_000_000_000)
        return int(s.replace(",", ""))
    except (ValueError, AttributeError):
        return None
