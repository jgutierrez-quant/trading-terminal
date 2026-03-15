"""
Google Trends sentiment proxy client.
Uses pytrends to fetch 7-day interest over time for one or more tickers
and derives a trend direction (rising / stable / falling).

Note: Google Trends measures *search interest*, not sentiment.
A rising trend means more people are searching — we treat it as a mild
bullish signal (retail attention). Falling = mild bearish proxy.
"""

import logging

# urllib3 v2 compat patch applied in sentiment/__init__.py
from pytrends.request import TrendReq

logger = logging.getLogger(__name__)


def get_trend(ticker: str) -> dict:
    """
    Fetch 7-day hourly interest over time for `ticker` from Google Trends.

    Trend direction is determined by comparing the slope of the last 3
    data points:
        rising  — both recent steps moved up
        falling — both recent steps moved down
        stable  — mixed or flat

    Returns:
        {
            "ticker":     str,
            "direction":  "rising" | "falling" | "stable" | None,
            "last_value": int | None,   # 0–100 Google interest score
            "peak_value": int | None,
            "series":     list[dict],   # [{datetime, value}, ...]
            "source":     "google_trends",
        }
    """
    ticker = ticker.upper()
    try:
        pt = TrendReq(hl="en-US", tz=300, timeout=(10, 25), retries=2, backoff_factor=0.5)
        pt.build_payload([ticker], timeframe="now 7-d", geo="US")
        df = pt.interest_over_time()

        if df is None or df.empty or ticker not in df.columns:
            logger.warning("Google Trends returned no data for %s", ticker)
            return _empty(ticker)

        # Drop the 'isPartial' flag column if present
        series_col = df[ticker]
        series = [
            {"datetime": str(dt), "value": int(val)}
            for dt, val in series_col.items()
        ]

        values = [s["value"] for s in series]
        direction = _trend_direction(values)

        return {
            "ticker":     ticker,
            "direction":  direction,
            "last_value": values[-1] if values else None,
            "peak_value": max(values) if values else None,
            "series":     series,
            "source":     "google_trends",
        }

    except Exception as e:
        logger.error("Google Trends failed for %s: %s", ticker, e)
        return {**_empty(ticker), "error": str(e)}


def compare_tickers(tickers: list[str]) -> dict:
    """
    Compare relative Google Trends interest for up to 5 tickers simultaneously.
    Returns a dict mapping each ticker to its average interest (0–100).
    """
    tickers = [t.upper() for t in tickers[:5]]
    try:
        pt = TrendReq(hl="en-US", tz=300, timeout=(10, 25), retries=2, backoff_factor=0.5)
        pt.build_payload(tickers, timeframe="now 7-d", geo="US")
        df = pt.interest_over_time()

        if df is None or df.empty:
            return {t: None for t in tickers}

        result = {}
        for t in tickers:
            if t in df.columns:
                result[t] = round(float(df[t].mean()), 1)
            else:
                result[t] = None
        return result

    except Exception as e:
        logger.error("Google Trends compare failed: %s", e)
        return {t: None for t in tickers}


# --- helpers ---

def _trend_direction(values: list[int]) -> str:
    """
    Classify trend from the last 3 values.
    Uses slope sign: positive both steps = rising, negative both = falling.
    """
    if len(values) < 3:
        return "stable"

    v1, v2, v3 = values[-3], values[-2], values[-1]
    d1 = v2 - v1  # older step
    d2 = v3 - v2  # newer step

    if d1 > 0 and d2 > 0:
        return "rising"
    if d1 < 0 and d2 < 0:
        return "falling"
    # mixed — look at net change over the 3-point window
    net = v3 - v1
    if net > 5:
        return "rising"
    if net < -5:
        return "falling"
    return "stable"


def _empty(ticker: str) -> dict:
    return {
        "ticker":     ticker,
        "direction":  None,
        "last_value": None,
        "peak_value": None,
        "series":     [],
        "source":     "google_trends",
    }
