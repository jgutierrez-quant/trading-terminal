"""
Unified market data wrapper.
Calls Polygon and yfinance independently — one source failing won't crash the other.
Returns a single clean dict for any ticker.
"""

import logging

from data.polygon_client  import get_quote, get_intraday_bars, get_options_chain
from data.yfinance_client import get_price_and_change, get_fundamentals, get_earnings_dates

logger = logging.getLogger(__name__)


def get_ticker_data(ticker: str) -> dict:
    """
    Pull all available data for a ticker from both sources.

    Price priority: Polygon live quote → yfinance fallback.
    Each sub-call is wrapped independently so partial failures degrade gracefully.
    """
    ticker = ticker.upper().strip()
    logger.info("Fetching full market data for %s", ticker)

    # --- Fetch from both sources independently ---
    polygon_quote  = _safe(get_quote,           ticker)
    intraday_bars  = _safe(get_intraday_bars,   ticker)
    options_chain  = _safe(get_options_chain,   ticker)
    yf_price       = _safe(get_price_and_change, ticker)
    fundamentals   = _safe(get_fundamentals,    ticker)
    earnings       = _safe(get_earnings_dates,  ticker, default=[])

    # --- Resolve best live price ---
    live_price = (
        polygon_quote.get("last_price")
        or polygon_quote.get("day_close")
        or yf_price.get("current_price")
    )

    # --- Resolve best change % ---
    change_pct = (
        polygon_quote.get("change_pct")
        or yf_price.get("change_pct")
    )

    return {
        "ticker":       ticker,
        "live_price":   live_price,
        "change_pct":   change_pct,
        # Granular sub-payloads for dashboard use
        "quote":        polygon_quote,
        "price_yf":     yf_price,
        "fundamentals": fundamentals,
        "earnings":     earnings,
        "intraday_bars": intraday_bars,
        "options_chain": options_chain,
        # Quick health check — useful for the test script and dashboard
        "sources": {
            "polygon_quote_ok":    "error" not in polygon_quote,
            "yfinance_ok":         "error" not in yf_price,
            "intraday_bars_count": len(intraday_bars) if isinstance(intraday_bars, list) else 0,
            "options_count":       len(options_chain) if isinstance(options_chain, list) else 0,
        },
    }


def _safe(fn, ticker: str, default: dict | list | None = None):
    """Call fn(ticker) and return `default` dict on unexpected exceptions."""
    if default is None:
        default = {"ticker": ticker, "error": "unexpected failure"}
    try:
        return fn(ticker)
    except Exception as e:
        logger.error("Unexpected error in %s for %s: %s", fn.__name__, ticker, e)
        return default
