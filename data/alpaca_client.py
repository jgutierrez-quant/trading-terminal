"""
Alpaca Paper Trading REST API client.

Public API:
    get_account()                                              -> dict
    get_positions()                                            -> list[dict]
    get_orders(status='all', limit=20)                         -> list[dict]
    place_order(ticker, qty, side, order_type, limit_price, stop_price) -> dict
    close_position(ticker)                                     -> dict
    cancel_order(order_id)                                     -> dict
    get_real_time_quote(ticker)                                -> dict

All functions return error dicts on failure (never raise).
Auth read from environment: ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL.
"""

import json
import logging
import os

import requests

logger = logging.getLogger(__name__)

_TRADING_URL = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
_DATA_URL    = "https://data.alpaca.markets"
_TIMEOUT     = 10


def _headers() -> dict:
    return {
        "APCA-API-KEY-ID":     os.environ.get("ALPACA_API_KEY", ""),
        "APCA-API-SECRET-KEY": os.environ.get("ALPACA_SECRET_KEY", ""),
        "Content-Type":        "application/json",
    }


def _get(url: str, params: dict = None):
    resp = requests.get(url, headers=_headers(), params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _post(url: str, body: dict) -> dict:
    resp = requests.post(url, headers=_headers(), data=json.dumps(body), timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _delete(url: str):
    resp = requests.delete(url, headers=_headers(), timeout=_TIMEOUT)
    resp.raise_for_status()
    if resp.status_code == 204 or not resp.text:
        return {"status": "ok"}
    return resp.json()


# ── Public functions ───────────────────────────────────────────────────────────

def get_account() -> dict:
    """
    Fetch Alpaca paper account summary.

    Returns:
        {portfolio_value, equity, cash, buying_power, daily_pnl, daily_pnl_pct,
         unrealized_pl, unrealized_plpc, error}
    """
    try:
        raw        = _get(f"{_TRADING_URL}/v2/account")
        equity     = float(raw.get("equity") or 0)
        last_eq    = float(raw.get("last_equity") or equity)
        daily_pnl  = round(equity - last_eq, 2)
        daily_pnl_pct = round((daily_pnl / last_eq * 100) if last_eq else 0, 2)
        return {
            "portfolio_value": round(float(raw.get("portfolio_value") or 0), 2),
            "equity":          round(equity, 2),
            "cash":            round(float(raw.get("cash") or 0), 2),
            "buying_power":    round(float(raw.get("buying_power") or 0), 2),
            "daily_pnl":       daily_pnl,
            "daily_pnl_pct":   daily_pnl_pct,
            "unrealized_pl":   round(float(raw.get("unrealized_pl") or 0), 2),
            "unrealized_plpc": round(float(raw.get("unrealized_plpc") or 0) * 100, 2),
            "error":           None,
        }
    except Exception as exc:
        logger.error("get_account failed: %s", exc)
        return {"error": str(exc)}


def get_positions() -> list:
    """
    Fetch all open positions.

    Returns list of:
        {ticker, qty, side, avg_entry_price, current_price,
         unrealized_pl, unrealized_plpc, market_value, cost_basis}
    """
    try:
        raw    = _get(f"{_TRADING_URL}/v2/positions")
        result = []
        for p in raw:
            result.append({
                "ticker":          p.get("symbol", ""),
                "qty":             float(p.get("qty") or 0),
                "side":            p.get("side", ""),
                "avg_entry_price": round(float(p.get("avg_entry_price") or 0), 2),
                "current_price":   round(float(p.get("current_price") or 0), 2),
                "unrealized_pl":   round(float(p.get("unrealized_pl") or 0), 2),
                "unrealized_plpc": round(float(p.get("unrealized_plpc") or 0) * 100, 2),
                "market_value":    round(float(p.get("market_value") or 0), 2),
                "cost_basis":      round(float(p.get("cost_basis") or 0), 2),
            })
        return result
    except Exception as exc:
        logger.error("get_positions failed: %s", exc)
        return []


def get_orders(status: str = "all", limit: int = 20) -> list:
    """
    Fetch recent orders.

    Args:
        status: 'open', 'closed', or 'all'.
        limit:  max number of orders to return.

    Returns list of {id, ticker, qty, side, order_type, status,
                     filled_price, limit_price, submitted_at, filled_at}.
    """
    try:
        raw    = _get(f"{_TRADING_URL}/v2/orders",
                      params={"status": status, "limit": limit, "direction": "desc"})
        result = []
        for o in raw:
            fp = float(o.get("filled_avg_price") or 0)
            lp = float(o.get("limit_price") or 0)
            result.append({
                "id":           o.get("id", ""),
                "ticker":       o.get("symbol", ""),
                "qty":          float(o.get("qty") or o.get("filled_qty") or 0),
                "side":         o.get("side", ""),
                "order_type":   o.get("type", ""),
                "status":       o.get("status", ""),
                "filled_price": round(fp, 2) if fp else None,
                "limit_price":  round(lp, 2) if lp else None,
                "submitted_at": (o.get("submitted_at") or "")[:19],
                "filled_at":    (o.get("filled_at") or "")[:19],
            })
        return result
    except Exception as exc:
        logger.error("get_orders failed: %s", exc)
        return []


def place_order(
    ticker: str,
    qty: int,
    side: str,
    order_type: str = "market",
    limit_price: float = None,
    stop_price: float = None,
) -> dict:
    """
    Submit a paper order to Alpaca.

    Args:
        ticker:      Stock symbol (e.g. 'AAPL').
        qty:         Whole shares (positive int).
        side:        'buy' or 'sell'.
        order_type:  'market', 'limit', 'stop', or 'stop_limit'.
        limit_price: Required for 'limit' and 'stop_limit' orders.
        stop_price:  Required for 'stop' and 'stop_limit' orders.

    Returns:
        {id, ticker, qty, side, order_type, status, submitted_at, error}
        or {'error': reason} on failure.
    """
    try:
        body: dict = {
            "symbol":        ticker.upper(),
            "qty":           str(int(qty)),
            "side":          side.lower(),
            "type":          order_type.lower(),
            "time_in_force": "day",
        }
        if limit_price is not None:
            body["limit_price"] = str(round(limit_price, 2))
        if stop_price is not None:
            body["stop_price"] = str(round(stop_price, 2))
        raw = _post(f"{_TRADING_URL}/v2/orders", body)
        return {
            "id":           raw.get("id", ""),
            "ticker":       raw.get("symbol", ""),
            "qty":          float(raw.get("qty") or 0),
            "side":         raw.get("side", ""),
            "order_type":   raw.get("type", ""),
            "status":       raw.get("status", ""),
            "submitted_at": (raw.get("submitted_at") or "")[:19],
            "error":        None,
        }
    except Exception as exc:
        logger.error("place_order %s %s %s failed: %s", side, qty, ticker, exc)
        return {"error": str(exc)}


def close_position(ticker: str) -> dict:
    """
    Liquidate an entire open position at market price.

    Returns:
        {'ticker': ticker, 'status': 'ok', 'error': None} on success,
        {'error': reason} on failure.
    """
    try:
        _delete(f"{_TRADING_URL}/v2/positions/{ticker.upper()}")
        return {"ticker": ticker.upper(), "status": "ok", "error": None}
    except Exception as exc:
        logger.error("close_position %s failed: %s", ticker, exc)
        return {"error": str(exc)}


def cancel_order(order_id: str) -> dict:
    """
    Cancel an open order by Alpaca order ID.

    Returns:
        {'order_id': order_id, 'status': 'cancelled', 'error': None} on success,
        {'error': reason} on failure.
    """
    try:
        _delete(f"{_TRADING_URL}/v2/orders/{order_id}")
        return {"order_id": order_id, "status": "cancelled", "error": None}
    except Exception as exc:
        logger.error("cancel_order %s failed: %s", order_id, exc)
        return {"error": str(exc)}


def get_real_time_quote(ticker: str) -> dict:
    """
    Fetch latest bid/ask quote from Alpaca IEX data feed.

    Returns:
        {ticker, bid, ask, timestamp, error}
    """
    try:
        url = f"{_DATA_URL}/v2/stocks/{ticker.upper()}/quotes/latest"
        raw = _get(url, params={"feed": "iex"})
        q   = raw.get("quote", {})
        bid = float(q.get("bp") or 0)
        ask = float(q.get("ap") or 0)
        return {
            "ticker":    ticker.upper(),
            "bid":       round(bid, 2) if bid else None,
            "ask":       round(ask, 2) if ask else None,
            "mid":       round((bid + ask) / 2, 2) if bid and ask else None,
            "timestamp": (q.get("t") or "")[:19],
            "error":     None,
        }
    except Exception as exc:
        logger.error("get_real_time_quote %s failed: %s", ticker, exc)
        return {"ticker": ticker.upper(), "error": str(exc)}
