"""
Catalyst detector — stock equivalent of late_breaking.py from props_analyzer.

Detects overnight / pre-market catalysts before market open:
    - Earnings surprises
    - Analyst upgrades/downgrades
    - Insider activity
    - Pre-market volume spikes
    - Breaking news (Yahoo + Finviz)
    - Gap up/down at open

Public API:
    detect_catalysts(ticker) -> dict
    scan_catalysts(tickers)  -> list[dict]

Each catalyst produces a boost (-100 to +100), direction, and plain English why.
Results are cached for 15 minutes per ticker.
"""

import logging
import re
import time
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)

# ── 15-minute in-memory cache ────────────────────────────────────────────────
_cache: dict[str, tuple[float, dict]] = {}
_CACHE_TTL = 900  # 15 minutes

# ── Noise filter — financial clickbait / listicle patterns ───────────────────
NOISE_PATTERNS = [
    r"\d+\s+stocks?\s+to\s+(buy|watch|sell|avoid)",
    r"top\s+\d+\s+(picks?|stocks?|etfs?)",
    r"best\s+\d+\s+",
    r"worst\s+\d+\s+",
    r"motley\s+fool\s+recommends",
    r"should\s+you\s+buy",
    r"is\s+it\s+time\s+to\s+buy",
    r"dividend\s+(king|aristocrat|champion)",
    r"passive\s+income",
    r"millionaire[\s-]maker",
    r"cathie\s+wood\s+(buys?|sells?)",
    r"wall\s+street\s+bets",
    r"meme\s+stock",
    r"reddit\s+favorite",
    r"retire\s+(early|rich)",
    r"weekly\s+roundup",
    r"market\s+wrap",
    r"morning\s+brief",
    r"here'?s?\s+why",
]
_noise_re = re.compile("|".join(NOISE_PATTERNS), re.IGNORECASE)

# ── Catalyst keywords — headline must contain at least one ───────────────────
CATALYST_KEYWORDS = [
    "earnings", "revenue", "beat", "miss", "guidance", "outlook", "forecast",
    "fda", "approval", "trial", "clinical", "drug",
    "merger", "acquisition", "acquire", "buyout", "takeover", "deal",
    "sec", "investigation", "subpoena", "fraud", "settlement",
    "ceo", "cfo", "cto", "resign", "fired", "appointed", "management",
    "upgrade", "downgrade", "price target", "initiate", "overweight", "underweight",
    "insider", "bought", "sold", "purchase", "filing",
    "recall", "lawsuit", "patent", "contract", "partnership",
    "dividend", "buyback", "split", "offering", "dilution",
    "gap up", "gap down", "premarket", "after hours",
    "warning", "downside", "upside", "surprise", "shock",
    "restructuring", "layoff", "cut", "expansion",
]

# ── Source weights (matches props sentiment approach) ────────────────────────
SOURCE_WEIGHTS = {
    "reuters":      2.0,
    "bloomberg":    2.0,
    "wsj":          1.8,
    "cnbc":         1.5,
    "barrons":      1.5,
    "seekingalpha": 1.0,
    "benzinga":     1.2,
    "marketwatch":  1.2,
    "yahoo":        1.0,
    "investopedia": 0.8,
    "motley fool":  0.5,
    "fool.com":     0.5,
    "insider monkey": 0.5,
    "24/7 wall":    0.5,
}


def _is_noise(headline: str) -> bool:
    """Filter out clickbait / listicle articles."""
    return bool(_noise_re.search(headline))


def _is_catalyst(headline: str) -> bool:
    """Check if headline contains catalyst-relevant keywords."""
    hl = headline.lower()
    return any(kw in hl for kw in CATALYST_KEYWORDS)


def _source_weight(source: str) -> float:
    """Weight headlines by source reliability."""
    src = source.lower()
    for name, weight in SOURCE_WEIGHTS.items():
        if name in src:
            return weight
    return 1.0


# ── Individual detectors ─────────────────────────────────────────────────────

def _check_earnings_surprise(ticker: str) -> list[dict]:
    """Check for recent earnings surprise via yfinance earnings_history."""
    catalysts = []
    try:
        t = yf.Ticker(ticker)
        eh = getattr(t, "earnings_history", None)
        if eh is None or (hasattr(eh, "empty") and eh.empty):
            return catalysts
        if isinstance(eh, pd.DataFrame) and not eh.empty:
            recent = eh.head(1)
            for _, row in recent.iterrows():
                surprise_pct = row.get("epsActual", 0) - row.get("epsEstimate", 0)
                eps_est = row.get("epsEstimate", 0)
                if eps_est and abs(eps_est) > 0.01:
                    surprise_pct = (row.get("epsActual", 0) - eps_est) / abs(eps_est) * 100
                else:
                    surprise_pct = 0

                # Only flag if within last 5 days
                report_date = row.get("quarter") or row.name
                if hasattr(report_date, "date"):
                    days_ago = (datetime.now().date() - report_date.date()).days
                    if days_ago > 5:
                        continue

                if surprise_pct > 5:
                    boost = min(15 + surprise_pct * 0.5, 25)
                    catalysts.append({
                        "type": "earnings_surprise",
                        "boost": round(boost),
                        "direction": "Bullish",
                        "why": f"Earnings beat by {surprise_pct:.1f}%",
                    })
                elif surprise_pct < -5:
                    boost = max(-15 + surprise_pct * 0.5, -25)
                    catalysts.append({
                        "type": "earnings_surprise",
                        "boost": round(boost),
                        "direction": "Bearish",
                        "why": f"Earnings missed by {abs(surprise_pct):.1f}%",
                    })
    except Exception as exc:
        logger.debug("Earnings surprise check failed for %s: %s", ticker, exc)
    return catalysts


def _check_analyst_actions(ticker: str) -> list[dict]:
    """Check for recent analyst upgrades/downgrades via yfinance."""
    catalysts = []
    try:
        t = yf.Ticker(ticker)
        ud = getattr(t, "upgrades_downgrades", None)
        if ud is None or (hasattr(ud, "empty") and ud.empty):
            return catalysts
        if isinstance(ud, pd.DataFrame) and not ud.empty:
            cutoff = datetime.now() - timedelta(days=3)
            for idx, row in ud.head(5).iterrows():
                # Check if recent
                if hasattr(idx, "date"):
                    if idx.date() < cutoff.date():
                        continue
                elif hasattr(idx, "to_pydatetime"):
                    if idx.to_pydatetime().replace(tzinfo=None) < cutoff:
                        continue

                action = str(row.get("Action", "")).lower()
                firm = str(row.get("Firm", "unknown"))
                to_grade = str(row.get("ToGrade", ""))

                if "upgrade" in action or "initiate" in action:
                    catalysts.append({
                        "type": "analyst_action",
                        "boost": 10,
                        "direction": "Bullish",
                        "why": f"{firm}: Upgraded to {to_grade}",
                    })
                elif "downgrade" in action:
                    catalysts.append({
                        "type": "analyst_action",
                        "boost": -10,
                        "direction": "Bearish",
                        "why": f"{firm}: Downgraded to {to_grade}",
                    })
    except Exception as exc:
        logger.debug("Analyst action check failed for %s: %s", ticker, exc)
    return catalysts


def _check_insider_activity(ticker: str) -> list[dict]:
    """Check for notable insider transactions via yfinance."""
    catalysts = []
    try:
        t = yf.Ticker(ticker)
        it = getattr(t, "insider_transactions", None)
        if it is None or (hasattr(it, "empty") and it.empty):
            return catalysts
        if isinstance(it, pd.DataFrame) and not it.empty:
            cutoff = datetime.now() - timedelta(days=7)
            total_buys = 0
            total_sells = 0
            for _, row in it.head(10).iterrows():
                start_date = row.get("startDate") or row.get("Start Date")
                if hasattr(start_date, "date"):
                    if start_date.date() < cutoff.date():
                        continue

                text = str(row.get("Text", "") or row.get("text", "")).lower()
                shares = abs(row.get("Shares", 0) or row.get("shares", 0) or 0)

                if "purchase" in text or "buy" in text:
                    total_buys += shares
                elif "sale" in text or "sell" in text:
                    total_sells += shares

            if total_buys > 0 and total_buys > total_sells * 2:
                catalysts.append({
                    "type": "insider_activity",
                    "boost": 12,
                    "direction": "Bullish",
                    "why": f"Insider buying detected ({total_buys:,} shares purchased)",
                })
            elif total_sells > 0 and total_sells > total_buys * 3:
                catalysts.append({
                    "type": "insider_activity",
                    "boost": -5,
                    "direction": "Bearish",
                    "why": f"Insider selling detected ({total_sells:,} shares sold)",
                })
    except Exception as exc:
        logger.debug("Insider activity check failed for %s: %s", ticker, exc)
    return catalysts


def _check_premarket_volume(ticker: str) -> list[dict]:
    """Check pre-market volume spike via Alpaca snapshot."""
    catalysts = []
    try:
        from data.alpaca_client import get_snapshot
        snap = get_snapshot(ticker)
        if snap.get("error"):
            return catalysts

        pm_vol = snap.get("premarket_volume", 0) or 0
        avg_vol = snap.get("avg_volume", 0) or 0

        if avg_vol > 0 and pm_vol > 0:
            ratio = pm_vol / avg_vol
            if ratio > 2.0:
                catalysts.append({
                    "type": "premarket_volume",
                    "boost": 5,
                    "direction": "Bullish",
                    "why": f"Pre-market volume {ratio:.1f}x average",
                })
            elif ratio > 1.5:
                catalysts.append({
                    "type": "premarket_volume",
                    "boost": 3,
                    "direction": "Bullish",
                    "why": f"Elevated pre-market volume ({ratio:.1f}x avg)",
                })
    except Exception as exc:
        logger.debug("Pre-market volume check failed for %s: %s", ticker, exc)
    return catalysts


def _check_news_catalysts(ticker: str) -> list[dict]:
    """Check Yahoo + Finviz headlines for catalyst-grade news."""
    catalysts = []
    try:
        from sentiment.yahoo_news_client import get_news_sentiment as yahoo_news
        from sentiment.finviz_client import get_news_sentiment as finviz_news
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        vader = SentimentIntensityAnalyzer()

        all_headlines = []

        # Yahoo headlines
        yahoo = yahoo_news(ticker, max_headlines=10)
        for h in yahoo.get("headlines", []):
            h["source_name"] = h.get("publisher", "Yahoo")
            all_headlines.append(h)

        # Finviz headlines
        finviz = finviz_news(ticker, max_headlines=10)
        for h in finviz.get("headlines", []):
            h["source_name"] = "Finviz"
            all_headlines.append(h)

        # Filter and score
        catalyst_headlines = []
        for h in all_headlines:
            title = h.get("title", "")
            if not title:
                continue
            if _is_noise(title):
                continue
            if not _is_catalyst(title):
                continue

            compound = vader.polarity_scores(title)["compound"]
            weight = _source_weight(h.get("source_name", ""))
            weighted_score = compound * weight

            catalyst_headlines.append({
                "title": title,
                "score": weighted_score,
                "source": h.get("source_name", ""),
            })

        if catalyst_headlines:
            avg_score = sum(h["score"] for h in catalyst_headlines) / len(catalyst_headlines)
            top = sorted(catalyst_headlines, key=lambda x: abs(x["score"]), reverse=True)[:3]

            if avg_score > 0.1:
                boost = min(round(avg_score * 16), 8)
                catalysts.append({
                    "type": "news_catalyst",
                    "boost": boost,
                    "direction": "Bullish",
                    "why": f"Bullish news: {top[0]['title'][:80]}",
                })
            elif avg_score < -0.1:
                boost = max(round(avg_score * 16), -8)
                catalysts.append({
                    "type": "news_catalyst",
                    "boost": boost,
                    "direction": "Bearish",
                    "why": f"Bearish news: {top[0]['title'][:80]}",
                })

    except Exception as exc:
        logger.debug("News catalyst check failed for %s: %s", ticker, exc)
    return catalysts


def _check_gap(ticker: str) -> list[dict]:
    """Check for gap up/down at open using yfinance + Alpaca."""
    catalysts = []
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="5d", interval="1d")
        if hist is None or len(hist) < 2:
            return catalysts

        prev_close = float(hist["Close"].iloc[-2])
        today_open = float(hist["Open"].iloc[-1])

        if prev_close <= 0:
            return catalysts

        gap_pct = (today_open - prev_close) / prev_close * 100

        if gap_pct > 2.0:
            boost = min(round(gap_pct * 2), 10)
            catalysts.append({
                "type": "gap",
                "boost": boost,
                "direction": "Bullish",
                "why": f"Gap up {gap_pct:.1f}% at open (${prev_close:.2f} -> ${today_open:.2f})",
            })
        elif gap_pct < -2.0:
            boost = max(round(gap_pct * 2), -10)
            catalysts.append({
                "type": "gap",
                "boost": boost,
                "direction": "Bearish",
                "why": f"Gap down {gap_pct:.1f}% at open (${prev_close:.2f} -> ${today_open:.2f})",
            })
    except Exception as exc:
        logger.debug("Gap check failed for %s: %s", ticker, exc)
    return catalysts


# ── Public API ───────────────────────────────────────────────────────────────

def detect_catalysts(ticker: str) -> dict:
    """
    Detect all active catalysts for a ticker.

    Returns:
        {
            "ticker":     str,
            "catalysts":  list[dict],   # each: {type, boost, direction, why}
            "boost":      int,          # net boost clamped to [-100, +100]
            "direction":  str,          # "Bullish" / "Bearish" / "Neutral"
            "summary":    str,          # one-line summary
            "why":        list[str],    # plain English reasons
        }
    """
    ticker = ticker.upper().strip()

    # Check cache
    now = time.time()
    if ticker in _cache:
        ts, cached = _cache[ticker]
        if now - ts < _CACHE_TTL:
            return cached

    # Run all detectors
    catalysts = []
    catalysts.extend(_check_earnings_surprise(ticker))
    catalysts.extend(_check_analyst_actions(ticker))
    catalysts.extend(_check_insider_activity(ticker))
    catalysts.extend(_check_premarket_volume(ticker))
    catalysts.extend(_check_news_catalysts(ticker))
    catalysts.extend(_check_gap(ticker))

    # Aggregate
    total_boost = sum(c["boost"] for c in catalysts)
    total_boost = max(-100, min(100, total_boost))  # clamp

    if total_boost > 20:
        direction = "Bullish"
    elif total_boost < -20:
        direction = "Bearish"
    else:
        direction = "Neutral"

    why = [c["why"] for c in catalysts]
    summary = f"{ticker}: {direction} ({total_boost:+d})" if catalysts else f"{ticker}: No catalysts"

    result = {
        "ticker": ticker,
        "catalysts": catalysts,
        "boost": total_boost,
        "direction": direction,
        "summary": summary,
        "why": why,
    }

    _cache[ticker] = (now, result)
    return result


def scan_catalysts(tickers: list[str]) -> list[dict]:
    """
    Scan multiple tickers for catalysts.
    Returns sorted list (highest absolute boost first), excluding quiet tickers.
    """
    results = []
    for ticker in tickers:
        try:
            result = detect_catalysts(ticker)
            if result["catalysts"]:  # only include tickers with signals
                results.append(result)
        except Exception as exc:
            logger.debug("Catalyst scan failed for %s: %s", ticker, exc)
            continue

    results.sort(key=lambda r: abs(r["boost"]), reverse=True)
    return results
