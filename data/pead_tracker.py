"""
PEAD Tracker — Post-Earnings Announcement Drift candidates.

Post-earnings announcement drift (PEAD) is a well-documented anomaly
(Ball & Brown 1968, Bernard & Thomas 1989): stocks with positive earnings
surprises continue to drift higher for 20-60 trading days post-announcement.

Public API:
    scan_pead_candidates(tickers, max_results=10)  -> list[dict]
    get_pead_status(ticker)                         -> dict

Each candidate dict:
    ticker, earnings_date, days_since, surprise_pct, drift_pct,
    est_remaining_drift, is_active, signal_strength
"""

import logging
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ── PEAD decay model parameters ───────────────────────────────────────────────
# Based on Bernard & Thomas (1989): drift accelerates then decays.
# Approx 60% of total PEAD realized by day 20, 85% by day 40, 100% by day 60.
_PEAD_WINDOW_TRADING_DAYS = 20          # active window for new candidates
_PEAD_TOTAL_WINDOW        = 60          # total drift window (trading days)

# Fraction of total expected drift realized by trading day t
# Calibrated to Bernard & Thomas cumulative abnormal return curve
_DECAY_CHECKPOINTS = {
     5: 0.20,
    10: 0.38,
    15: 0.52,
    20: 0.62,
    30: 0.75,
    40: 0.85,
    50: 0.93,
    60: 1.00,
}

# Expected total drift multiplier per unit of SUE (standardized unexpected earnings)
# A SUE of +1.0 (1 std above consensus) → approx +3% total drift (S&P 500 estimate)
_DRIFT_PER_SUE_UNIT = 0.030


# ── Public API ─────────────────────────────────────────────────────────────────

def scan_pead_candidates(tickers: list, max_results: int = 10) -> list:
    """
    Scan a list of tickers for active PEAD long opportunities.

    Criteria for inclusion:
        - Positive earnings surprise (actual > estimate)
        - Within _PEAD_WINDOW_TRADING_DAYS of earnings announcement
        - Price still above earnings-date closing price (trend intact)
        - SUE >= 0.5 (at least half a standard deviation beat)

    Returns list of candidate dicts sorted by signal_strength descending.
    """
    candidates = []
    for ticker in tickers:
        try:
            status = get_pead_status(ticker)
            if status.get("is_active"):
                candidates.append(status)
        except Exception as exc:
            logger.warning("PEAD scan failed for %s: %s", ticker, exc)

    candidates.sort(key=lambda r: r.get("signal_strength", 0), reverse=True)
    return candidates[:max_results]


def get_pead_status(ticker: str) -> dict:
    """
    Compute PEAD metrics for a single ticker.

    Returns dict with:
        ticker              str
        earnings_date       str | None    — most recent earnings date (YYYY-MM-DD)
        days_since          int | None    — calendar days since earnings
        trading_days_since  int | None    — trading days since earnings
        surprise_pct        float | None  — EPS % beat (positive = beat)
        sue                 float | None  — standardized unexpected earnings
        drift_pct           float | None  — % price change since earnings date
        est_total_drift     float | None  — estimated total expected drift (%)
        est_remaining_drift float | None  — estimated remaining drift (%)
        pct_drift_realized  float | None  — fraction of total drift already captured
        is_active           bool          — True if valid PEAD long candidate
        signal_strength     float         — 0-100 composite signal strength
        error               str | None
    """
    ticker = ticker.upper().strip()
    base = {
        "ticker":              ticker,
        "earnings_date":       None,
        "days_since":          None,
        "trading_days_since":  None,
        "surprise_pct":        None,
        "sue":                 None,
        "drift_pct":           None,
        "est_total_drift":     None,
        "est_remaining_drift": None,
        "pct_drift_realized":  None,
        "is_active":           False,
        "signal_strength":     0.0,
        "error":               None,
    }

    try:
        t = yf.Ticker(ticker)

        # ── Earnings history ──────────────────────────────────────────────────
        earnings_hist = _get_earnings_history(t, ticker)
        if not earnings_hist:
            base["error"] = "no earnings history"
            return base

        # Most recent quarter
        last = earnings_hist[0]
        earnings_dt   = last["date"]        # datetime.date
        surprise_pct  = last["surprise_pct"]  # % beat/miss vs estimate
        sue           = last["sue"]            # standardized

        base["earnings_date"] = earnings_dt.isoformat()
        base["surprise_pct"]  = round(surprise_pct, 2) if surprise_pct is not None else None
        base["sue"]           = round(sue, 2) if sue is not None else None

        today     = date.today()
        cal_days  = (today - earnings_dt).days
        base["days_since"] = cal_days

        # Count trading days since earnings
        trading_days = _count_trading_days(earnings_dt, today, ticker)
        base["trading_days_since"] = trading_days

        # ── Drift since earnings ──────────────────────────────────────────────
        drift_pct = _compute_drift(t, ticker, earnings_dt, today)
        base["drift_pct"] = round(drift_pct, 2) if drift_pct is not None else None

        # ── Estimated drift ───────────────────────────────────────────────────
        if sue is not None and sue > 0:
            est_total = sue * _DRIFT_PER_SUE_UNIT * 100  # % expected total drift
            pct_realized = _drift_realized_fraction(trading_days)
            est_remaining = est_total * (1.0 - pct_realized)

            base["est_total_drift"]     = round(est_total, 2)
            base["est_remaining_drift"] = round(max(est_remaining, 0.0), 2)
            base["pct_drift_realized"]  = round(pct_realized * 100, 1)
        else:
            est_total = None
            est_remaining = None

        # ── Active PEAD check ─────────────────────────────────────────────────
        is_active = (
            surprise_pct is not None and surprise_pct > 0
            and sue is not None and sue >= 0.5
            and trading_days is not None and trading_days <= _PEAD_WINDOW_TRADING_DAYS
            and drift_pct is not None and drift_pct >= -1.0  # tolerate minor pullback
        )
        base["is_active"] = is_active

        # ── Signal strength (0-100) ───────────────────────────────────────────
        if is_active:
            sue_pts     = min(sue / 3.0, 1.0) * 40          # max 40 pts at SUE = 3
            recency_pts = max(1.0 - trading_days / _PEAD_WINDOW_TRADING_DAYS, 0) * 30  # fresher = more pts
            drift_pts   = min(max(drift_pct / 5.0, 0), 1.0) * 20  # up to 20 pts for drift momentum
            remain_pts  = min((est_remaining or 0) / 3.0, 1.0) * 10
            base["signal_strength"] = round(sue_pts + recency_pts + drift_pts + remain_pts, 1)

        return base

    except Exception as exc:
        logger.error("get_pead_status failed for %s: %s", ticker, exc)
        base["error"] = str(exc)
        return base


# ── Internal helpers ───────────────────────────────────────────────────────────

def _get_earnings_history(ticker_obj: yf.Ticker, ticker: str) -> list:
    """
    Extract recent earnings surprises from yfinance.

    Returns list of dicts sorted newest-first:
        [{date, actual_eps, estimate_eps, surprise_pct, sue}]
    """
    try:
        hist = ticker_obj.earnings_history
        if hist is None or (hasattr(hist, "empty") and hist.empty):
            return []

        df = hist.copy()
        df = df.sort_index(ascending=False)

        results = []
        eps_surprises = []

        for idx, row in df.iterrows():
            actual   = _sf(row.get("epsActual"))
            estimate = _sf(row.get("epsEstimate"))
            if actual is None or estimate is None:
                continue

            surprise_raw = actual - estimate
            if estimate != 0:
                surprise_pct = (surprise_raw / abs(estimate)) * 100
            else:
                surprise_pct = 0.0

            eps_surprises.append(surprise_raw)

            # Parse date
            if isinstance(idx, (datetime, pd.Timestamp)):
                dt = idx.date() if hasattr(idx, "date") else idx
            elif isinstance(idx, date):
                dt = idx
            else:
                try:
                    dt = pd.to_datetime(idx).date()
                except Exception:
                    continue

            results.append({
                "date":         dt,
                "actual_eps":   actual,
                "estimate_eps": estimate,
                "surprise_pct": surprise_pct,
                "surprise_raw": surprise_raw,
                "sue":          None,  # filled in after computing std
            })

        # Compute SUE (standardized unexpected earnings)
        # SUE = (actual - estimate) / std(last N surprises)
        if len(eps_surprises) >= 2:
            surprise_std = float(np.std(eps_surprises, ddof=1))
        else:
            surprise_std = None

        for i, r in enumerate(results):
            if surprise_std and surprise_std > 0:
                r["sue"] = r["surprise_raw"] / surprise_std
            else:
                # Fallback: use %-based heuristic (SUE proxy)
                r["sue"] = r["surprise_pct"] / 10.0  # 10% beat ≈ SUE 1.0

        return results

    except Exception as exc:
        logger.warning("_get_earnings_history failed for %s: %s", ticker, exc)
        return []


def _compute_drift(ticker_obj: yf.Ticker, ticker: str,
                   earnings_dt: date, today: date) -> float | None:
    """
    Compute % price change from earnings-date close to today's close.
    Uses a window starting 1 day before earnings to catch any AH move.
    """
    try:
        start = earnings_dt - timedelta(days=3)
        end   = today + timedelta(days=1)

        hist = ticker_obj.history(start=start.isoformat(), end=end.isoformat(),
                                  interval="1d", auto_adjust=True)
        if hist is None or hist.empty:
            return None

        hist = hist.sort_index()

        # Normalize index to tz-naive for comparisons
        close_col = "Close"
        if hist.index.tzinfo is not None or (hasattr(hist.index, "tz") and hist.index.tz is not None):
            hist.index = hist.index.tz_localize(None)
        hist.index = hist.index.normalize()

        earnings_ts = pd.Timestamp(earnings_dt)
        available_dates = hist.index.tolist()

        # Find first available date on or after earnings
        base_date = None
        for d in available_dates:
            if d >= earnings_ts:
                base_date = d
                break

        if base_date is None:
            return None

        base_price = float(hist.loc[base_date, close_col])
        last_price = float(hist[close_col].iloc[-1])

        if base_price <= 0:
            return None

        return (last_price - base_price) / base_price * 100

    except Exception as exc:
        logger.warning("_compute_drift failed for %s: %s", ticker, exc)
        return None


def _count_trading_days(start: date, end: date, ticker: str) -> int | None:
    """Count approximate trading days between two dates (Mon-Fri, no holidays)."""
    try:
        if end <= start:
            return 0
        # Business days (Mon-Fri) — simple approximation (ignores market holidays)
        days = pd.bdate_range(start=start, end=end - timedelta(days=1))
        return len(days)
    except Exception:
        return None


def _drift_realized_fraction(trading_days: int | None) -> float:
    """
    Return the fraction of total expected PEAD drift realized by trading_days.
    Uses the calibrated decay checkpoint table.
    """
    if trading_days is None:
        return 0.5

    checkpoints = sorted(_DECAY_CHECKPOINTS.items())

    # Below first checkpoint
    if trading_days <= checkpoints[0][0]:
        return checkpoints[0][1] * (trading_days / checkpoints[0][0])

    # Above last checkpoint
    if trading_days >= checkpoints[-1][0]:
        return 1.0

    # Interpolate between surrounding checkpoints
    for i in range(len(checkpoints) - 1):
        t0, f0 = checkpoints[i]
        t1, f1 = checkpoints[i + 1]
        if t0 <= trading_days <= t1:
            alpha = (trading_days - t0) / (t1 - t0)
            return f0 + alpha * (f1 - f0)

    return 1.0


def _sf(val):
    """Return float or None for NaN/None/inf."""
    try:
        v = float(val)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


# Lazy import to avoid circular at module level
import math
