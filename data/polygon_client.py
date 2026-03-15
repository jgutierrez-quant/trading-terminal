"""
Polygon.io REST client.
Provides real-time quotes, 1-minute intraday bars, and options chain snapshots.
"""

import os
import logging
from datetime import datetime, timedelta

from dotenv import load_dotenv
from polygon import RESTClient

load_dotenv()
logger = logging.getLogger(__name__)


def _client() -> RESTClient:
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise ValueError("POLYGON_API_KEY not set in .env")
    return RESTClient(api_key=api_key)


def get_quote(ticker: str) -> dict:
    """
    Real-time snapshot quote for a ticker.
    Returns last price, bid/ask, day OHLCV, change %, and prev close.
    """
    ticker = ticker.upper()
    try:
        snap = _client().get_snapshot_ticker("stocks", ticker)

        result = {
            "ticker": ticker,
            "last_price": None,
            "bid": None,
            "ask": None,
            "day_open": None,
            "day_high": None,
            "day_low": None,
            "day_close": None,
            "day_volume": None,
            "change_pct": None,
            "prev_close": None,
        }

        if snap is None:
            return result

        lt = getattr(snap, "last_trade", None)
        if lt:
            result["last_price"] = getattr(lt, "price", None)

        lq = getattr(snap, "last_quote", None)
        if lq:
            result["bid"] = getattr(lq, "bid_price", None)
            result["ask"] = getattr(lq, "ask_price", None)

        day = getattr(snap, "day", None)
        if day:
            result["day_open"] = getattr(day, "open", None)
            result["day_high"] = getattr(day, "high", None)
            result["day_low"] = getattr(day, "low", None)
            result["day_close"] = getattr(day, "close", None)
            result["day_volume"] = getattr(day, "volume", None)

        result["change_pct"] = getattr(snap, "today_change_percent", None)

        prev = getattr(snap, "prev_day", None)
        if prev:
            result["prev_close"] = getattr(prev, "close", None)

        return result

    except Exception as e:
        logger.error("Polygon get_quote failed for %s: %s", ticker, e)
        return {"ticker": ticker, "error": str(e)}


def get_intraday_bars(ticker: str, days_back: int = 1) -> list[dict]:
    """
    1-minute OHLCV bars going back `days_back` trading days.
    Returns a list of bar dicts sorted ascending by timestamp.
    """
    ticker = ticker.upper()
    try:
        to_dt = datetime.now()
        from_dt = to_dt - timedelta(days=days_back + 3)  # buffer for weekends
        from_str = from_dt.strftime("%Y-%m-%d")
        to_str = to_dt.strftime("%Y-%m-%d")

        bars = []
        for agg in _client().list_aggs(
            ticker,
            multiplier=1,
            timespan="minute",
            from_=from_str,
            to=to_str,
            adjusted=True,
            sort="asc",
            limit=500,
        ):
            bars.append({
                "timestamp": getattr(agg, "timestamp", None),
                "open":      getattr(agg, "open",      None),
                "high":      getattr(agg, "high",      None),
                "low":       getattr(agg, "low",       None),
                "close":     getattr(agg, "close",     None),
                "volume":    getattr(agg, "volume",    None),
                "vwap":      getattr(agg, "vwap",      None),
            })

        return bars

    except Exception as e:
        logger.error("Polygon get_intraday_bars failed for %s: %s", ticker, e)
        return []


def get_options_chain(ticker: str, limit: int = 25) -> list[dict]:
    """
    Options chain snapshots for a ticker (requires Polygon Starter plan or above).
    Returns a list of contracts with greeks, IV, OI, and day OHLCV.
    Returns an empty list gracefully on free-tier 403 errors.
    """
    ticker = ticker.upper()
    try:
        contracts = []
        for opt in _client().list_snapshot_options_chain(ticker):
            details = getattr(opt, "details", None)
            greeks  = getattr(opt, "greeks",  None)
            day     = getattr(opt, "day",     None)

            if len(contracts) >= limit:
                break
            contracts.append({
                "contract_ticker":  getattr(details, "ticker",          None) if details else None,
                "contract_type":    getattr(details, "contract_type",   None) if details else None,
                "expiration_date":  getattr(details, "expiration_date", None) if details else None,
                "strike_price":     getattr(details, "strike_price",    None) if details else None,
                "delta":            getattr(greeks,  "delta",           None) if greeks  else None,
                "gamma":            getattr(greeks,  "gamma",           None) if greeks  else None,
                "theta":            getattr(greeks,  "theta",           None) if greeks  else None,
                "vega":             getattr(greeks,  "vega",            None) if greeks  else None,
                "implied_vol":      getattr(opt,     "implied_volatility", None),
                "open_interest":    getattr(opt,     "open_interest",   None),
                "day_close":        getattr(day,     "close",           None) if day     else None,
                "day_volume":       getattr(day,     "volume",          None) if day     else None,
            })

        return contracts

    except Exception as e:
        # 403 on free tier — degrade gracefully
        logger.warning("Polygon get_options_chain failed for %s: %s", ticker, e)
        return []
