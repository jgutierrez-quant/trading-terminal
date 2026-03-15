"""
Sector performance monitor — tracks the 11 S&P 500 SPDR sector ETFs.

Public API:
    get_sector_data() -> list[dict]
    SECTOR_ETFS       -> list of (ticker, name) tuples
"""

import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

SECTOR_ETFS = [
    ("XLK",  "Technology"),
    ("XLF",  "Financials"),
    ("XLE",  "Energy"),
    ("XLV",  "Health Care"),
    ("XLY",  "Cons. Discret."),
    ("XLI",  "Industrials"),
    ("XLB",  "Materials"),
    ("XLRE", "Real Estate"),
    ("XLU",  "Utilities"),
    ("XLP",  "Cons. Staples"),
    ("XLC",  "Comm. Services"),
]


def get_sector_data() -> list:
    """
    Fetch today's price change % for all 11 sector ETFs in a single
    yfinance batch download.

    Returns a list of dicts sorted by change_pct descending (best first).
    Each dict: {ticker, name, price, change_pct, volume}
    """
    tickers  = [t for t, _ in SECTOR_ETFS]
    name_map = {t: n for t, n in SECTOR_ETFS}
    try:
        symbols = " ".join(tickers)
        df = yf.download(symbols, period="5d", interval="1d",
                         auto_adjust=True, progress=False)
        if df.empty:
            return _empty(name_map)

        results = []
        for ticker in tickers:
            price = chg = volume = None
            try:
                # Multi-ticker download returns MultiIndex columns
                if isinstance(df.columns, pd.MultiIndex):
                    closes = df["Close"][ticker].dropna()
                    vols   = df["Volume"][ticker].dropna()
                else:
                    # Single-ticker fallback (shouldn't occur with 11 ETFs)
                    closes = df["Close"].dropna()
                    vols   = df["Volume"].dropna()

                if len(closes) >= 2:
                    price = round(float(closes.iloc[-1]), 2)
                    prev  = float(closes.iloc[-2])
                    chg   = round((price - prev) / prev * 100, 2) if prev else None
                elif len(closes) == 1:
                    price = round(float(closes.iloc[-1]), 2)

                if len(vols) > 0:
                    volume = int(vols.iloc[-1])

            except Exception as exc:
                logger.warning("Sector data failed for %s: %s", ticker, exc)

            results.append({
                "ticker":     ticker,
                "name":       name_map[ticker],
                "price":      price,
                "change_pct": chg,
                "volume":     volume,
            })

        # Sort best to worst; tickers with None change_pct go last
        results.sort(
            key=lambda r: (r["change_pct"] is None, -(r["change_pct"] or 0))
        )
        return results

    except Exception as exc:
        logger.error("get_sector_data failed: %s", exc)
        return _empty(name_map)


def _empty(name_map: dict) -> list:
    return [
        {"ticker": t, "name": name_map[t], "price": None,
         "change_pct": None, "volume": None}
        for t in name_map
    ]
