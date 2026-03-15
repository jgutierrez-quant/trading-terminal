"""
Risk management utilities for position sizing and trade validation.

Public API:
    calculate_position_size(account_value, risk_percent, entry_price, stop_price) -> int
    calculate_stop_loss(ticker, entry_price, side, atr_multiplier=2.0)            -> float | None
    calculate_take_profit(entry_price, stop_price, side, risk_reward=2.0)         -> float | None
    validate_trade(ticker, side, account_value, buying_power, positions=None)     -> dict
    get_market_hours()                                                             -> dict
"""

import logging
import math
from datetime import datetime

import yfinance as yf

logger = logging.getLogger(__name__)

_MARKET_OPEN_H  = 9
_MARKET_OPEN_M  = 30
_MARKET_CLOSE_H = 16
_MAX_POSITION_PCT = 0.20   # No single position > 20% of account


def calculate_position_size(
    account_value: float,
    risk_percent: float,
    entry_price: float,
    stop_price: float,
) -> int:
    """
    Dollar-risk-based position sizing.

    Formula: shares = floor((account * risk%) / |entry - stop|)
    Also enforces a 20% max-position cap per single trade.

    Args:
        account_value: Total portfolio value in dollars.
        risk_percent:  Percent of account to risk (e.g. 1.0 for 1%).
        entry_price:   Intended entry price per share.
        stop_price:    Stop-loss price per share.

    Returns:
        Number of whole shares to trade, or 0 if inputs are invalid.
    """
    try:
        stop_distance = abs(entry_price - stop_price)
        if stop_distance <= 0 or account_value <= 0 or risk_percent <= 0 or entry_price <= 0:
            return 0
        dollar_risk = account_value * (risk_percent / 100)
        shares      = int(math.floor(dollar_risk / stop_distance))
        max_by_cap  = int(math.floor(account_value * _MAX_POSITION_PCT / entry_price))
        return min(shares, max_by_cap)
    except Exception as exc:
        logger.error("calculate_position_size failed: %s", exc)
        return 0


def calculate_stop_loss(
    ticker: str,
    entry_price: float,
    side: str,
    atr_multiplier: float = 2.0,
) -> float | None:
    """
    ATR-based stop loss calculation.

    Fetches 20 trading days of daily OHLC via yfinance, computes 14-period ATR,
    then returns:
        Long:  stop = entry - (ATR * multiplier)
        Short: stop = entry + (ATR * multiplier)

    Falls back to a simple 3% stop if ATR data is unavailable.

    Returns:
        Stop price (float) or None if entry_price is invalid.
    """
    if not entry_price or entry_price <= 0:
        return None
    try:
        hist  = yf.Ticker(ticker).history(period="20d", interval="1d")
        if hist.empty or len(hist) < 2:
            raise ValueError("insufficient data")

        high  = hist["High"]
        low   = hist["Low"]
        close = hist["Close"]

        tr_hl = high - low
        tr_hc = (high - close.shift(1)).abs()
        tr_lc = (low  - close.shift(1)).abs()
        tr    = tr_hl.combine(tr_hc, max).combine(tr_lc, max)
        atr   = tr.rolling(14).mean().iloc[-1]

        if not atr or atr != atr:   # NaN or zero
            raise ValueError("ATR unavailable")

        offset = atr * atr_multiplier
        if side.lower() == "buy":
            return round(entry_price - offset, 2)
        else:
            return round(entry_price + offset, 2)

    except Exception as exc:
        logger.warning("calculate_stop_loss ATR failed for %s (%s) — using 3%% fallback", ticker, exc)
        if side.lower() == "buy":
            return round(entry_price * 0.97, 2)
        else:
            return round(entry_price * 1.03, 2)


def calculate_take_profit(
    entry_price: float,
    stop_price: float,
    side: str,
    risk_reward: float = 2.0,
) -> float | None:
    """
    Risk-reward take-profit target.

    Formula:
        Long:  target = entry + (|entry - stop| * risk_reward)
        Short: target = entry - (|entry - stop| * risk_reward)

    Returns:
        Target price (float), or None if inputs are invalid.
    """
    try:
        stop_distance = abs(entry_price - stop_price)
        if stop_distance <= 0:
            return None
        profit_distance = stop_distance * risk_reward
        if side.lower() == "buy":
            return round(entry_price + profit_distance, 2)
        else:
            return round(entry_price - profit_distance, 2)
    except Exception as exc:
        logger.error("calculate_take_profit failed: %s", exc)
        return None


def validate_trade(
    ticker: str,
    side: str,
    account_value: float,
    buying_power: float,
    positions: list = None,
) -> dict:
    """
    Pre-trade validation checks.

    Checks performed:
        1. Market hours (warning if closed, not a hard block).
        2. Buying power sufficient for buys (error if < 1% of account).
        3. Existing position in same ticker (warning if adding to position).

    Args:
        ticker:        Symbol to trade.
        side:          'buy' or 'sell'.
        account_value: Total portfolio value.
        buying_power:  Available buying power from Alpaca account.
        positions:     Optional list of current position dicts (with 'ticker' key).
                       Pass None to skip existing-position check.

    Returns:
        {'valid': bool, 'warnings': list[str], 'errors': list[str]}
    """
    warnings: list = []
    errors:   list = []

    # Market hours check (warning only — paper orders queue for next session)
    mh = get_market_hours()
    if not mh.get("is_open"):
        warnings.append(
            f"Market is currently closed ({mh.get('market_session', 'Unknown')}). "
            f"Order will queue for next open."
        )

    # Buying power check (buy orders only)
    if side.lower() == "buy":
        if buying_power < account_value * 0.01:
            errors.append(
                f"Insufficient buying power: ${buying_power:,.2f} "
                f"(account equity: ${account_value:,.2f})."
            )

    # Existing position check
    if positions is not None:
        existing = [p for p in positions if p.get("ticker", "").upper() == ticker.upper()]
        if existing:
            p = existing[0]
            warnings.append(
                f"Already hold {p.get('qty', '?')} shares of {ticker} "
                f"(unrealized P&L: ${p.get('unrealized_pl', 0):+,.2f}). "
                f"This will add to the existing position."
            )

    return {
        "valid":    len(errors) == 0,
        "warnings": warnings,
        "errors":   errors,
    }


def get_market_hours() -> dict:
    """
    Check whether the US stock market is currently open (Eastern Time).

    Returns:
        {is_open, current_time_et, market_session, next_open, next_close}
    """
    try:
        try:
            from zoneinfo import ZoneInfo
            tz_et = ZoneInfo("America/New_York")
        except ImportError:
            import pytz
            tz_et = pytz.timezone("America/New_York")

        now  = datetime.now(tz_et)
        wday = now.weekday()   # 0=Mon … 6=Sun
        h, m = now.hour, now.minute

        after_open  = h > _MARKET_OPEN_H  or (h == _MARKET_OPEN_H  and m >= _MARKET_OPEN_M)
        before_close = h < _MARKET_CLOSE_H

        is_open = wday < 5 and after_open and before_close

        if wday >= 5:
            session = "Weekend — closed"
        elif not after_open:
            session = "Pre-market"
        elif not before_close:
            session = "After-hours"
        else:
            session = "Regular hours"

        return {
            "is_open":         is_open,
            "current_time_et": now.strftime("%H:%M:%S ET"),
            "market_session":  session,
            "next_open":       "09:30 ET" if not is_open else "Now open",
            "next_close":      "16:00 ET" if is_open else "N/A",
        }
    except Exception as exc:
        logger.error("get_market_hours failed: %s", exc)
        return {"is_open": False, "market_session": "Unknown", "error": str(exc)}
