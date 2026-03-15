"""
Unified market data wrapper.
Calls Alpaca, Polygon, and yfinance independently — one source failing won't crash the other.
Returns a single clean dict for any ticker.

Price priority:
    1. Alpaca IEX real-time mid (bid+ask)/2  — free, real-time via IEX feed
    2. Polygon last_price / day_close        — free tier: previous-day data only
    3. yfinance current_price               — 15-min delayed fallback
"""

import logging

from data.alpaca_client   import get_real_time_quote
from data.polygon_client  import get_quote, get_intraday_bars, get_options_chain
from data.yfinance_client import get_price_and_change, get_fundamentals, get_earnings_dates

logger = logging.getLogger(__name__)


def get_ticker_data(ticker: str) -> dict:
    """
    Pull all available data for a ticker from all three sources.

    Price priority: Alpaca IEX real-time → Polygon → yfinance (15-min delayed).
    Each sub-call is wrapped independently so partial failures degrade gracefully.

    Returns `price_source` key: "alpaca" | "polygon" | "yfinance" | None
    """
    ticker = ticker.upper().strip()
    logger.info("Fetching full market data for %s", ticker)

    # --- Fetch from all sources independently ---
    alpaca_quote   = _safe(get_real_time_quote,  ticker)
    polygon_quote  = _safe(get_quote,            ticker)
    intraday_bars  = _safe(get_intraday_bars,    ticker)
    options_chain  = _safe(get_options_chain,    ticker)
    yf_price       = _safe(get_price_and_change, ticker)
    fundamentals   = _safe(get_fundamentals,     ticker)
    earnings       = _safe(get_earnings_dates,   ticker, default=[])

    # --- Resolve best live price (Alpaca IEX → Polygon → yfinance) ---
    alpaca_mid = alpaca_quote.get("mid") if not alpaca_quote.get("error") else None
    poly_price = (polygon_quote.get("last_price") or polygon_quote.get("day_close")
                  if not polygon_quote.get("error") else None)
    yf_current = yf_price.get("current_price") if not yf_price.get("error") else None

    if alpaca_mid:
        live_price   = alpaca_mid
        price_source = "alpaca"
    elif poly_price:
        live_price   = poly_price
        price_source = "polygon"
    elif yf_current:
        live_price   = yf_current
        price_source = "yfinance"
    else:
        live_price   = None
        price_source = None

    # --- Resolve best change % (Polygon → yfinance) ---
    change_pct = (
        polygon_quote.get("change_pct")
        or yf_price.get("change_pct")
    )

    return {
        "ticker":        ticker,
        "live_price":    live_price,
        "change_pct":    change_pct,
        "price_source":  price_source,   # "alpaca" | "polygon" | "yfinance" | None
        # Granular sub-payloads for dashboard use
        "alpaca_quote":  alpaca_quote,
        "quote":         polygon_quote,
        "price_yf":      yf_price,
        "fundamentals":  fundamentals,
        "earnings":      earnings,
        "intraday_bars": intraday_bars,
        "options_chain": options_chain,
        # Quick health check — useful for the test script and dashboard
        "sources": {
            "alpaca_quote_ok":     not alpaca_quote.get("error"),
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
