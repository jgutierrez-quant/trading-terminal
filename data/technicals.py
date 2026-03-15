"""
Technical analysis indicators — computed from yfinance daily + intraday data.

All indicators are implemented with pure pandas/numpy for full pandas 2.x
compatibility. pandas-ta is listed in requirements.txt for reference but
the native implementations here are used for reliability.

get_technicals(ticker) is the single public entry point.
"""

import math
import logging
from datetime import datetime

import pandas as pd
import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)


# ── Public entry point ────────────────────────────────────────────────────────

def get_technicals(ticker: str) -> dict:
    """
    Fetch 90 days of daily OHLCV + today's 1-min intraday data and compute:
      RSI (14), MACD (12/26/9), Bollinger Bands (20, 2σ),
      VWAP (intraday), Volume anomaly (20d avg), SMA 20/50,
      Support & Resistance (top 3 levels), daily chart bars.

    Returns a flat dict — never raises. Errors surfaced in 'error' key.
    """
    ticker = ticker.upper()
    try:
        # ── 90-day daily OHLCV ──────────────────────────────────────────────
        df = yf.Ticker(ticker).history(period="90d", interval="1d")
        if df.empty or len(df) < 30:
            return _empty(ticker, "Insufficient daily data (need ≥ 30 bars)")

        close = df["Close"]
        high  = df["High"]
        low   = df["Low"]
        vol   = df["Volume"]
        current_price = float(close.iloc[-1])

        # ── RSI ─────────────────────────────────────────────────────────────
        rsi_s   = _rsi(close, 14)
        cur_rsi = _safe(rsi_s.iloc[-1])
        if cur_rsi is None:
            rsi_signal = "Neutral"
        elif cur_rsi > 70:
            rsi_signal = "Overbought"
        elif cur_rsi < 30:
            rsi_signal = "Oversold"
        else:
            rsi_signal = "Neutral"

        # ── MACD ────────────────────────────────────────────────────────────
        macd_line, sig_line, hist_s = _macd(close, 12, 26, 9)
        cur_macd = _safe(macd_line.iloc[-1])
        cur_sig  = _safe(sig_line.iloc[-1])
        cur_hist = _safe(hist_s.iloc[-1])
        prv_hist = _safe(hist_s.iloc[-2]) if len(hist_s) >= 2 else None

        if prv_hist is not None and cur_hist is not None:
            if prv_hist < 0 and cur_hist >= 0:
                macd_signal = "Bullish Cross"
            elif prv_hist > 0 and cur_hist <= 0:
                macd_signal = "Bearish Cross"
            elif cur_hist > 0:
                macd_signal = "Bullish"
            else:
                macd_signal = "Bearish"
        else:
            macd_signal = "Neutral"

        # ── Bollinger Bands ──────────────────────────────────────────────────
        bb_up, bb_mid, bb_low = _bbands(close, 20, 2.0)
        cur_bb_up  = _safe(bb_up.iloc[-1])
        cur_bb_mid = _safe(bb_mid.iloc[-1])
        cur_bb_low = _safe(bb_low.iloc[-1])

        if cur_bb_up is not None and current_price > cur_bb_up:
            bb_signal = "Above Upper"
        elif cur_bb_low is not None and current_price < cur_bb_low:
            bb_signal = "Below Lower"
        else:
            bb_signal = "Inside Bands"

        # ── Moving Averages ──────────────────────────────────────────────────
        sma20_s = _sma(close, 20)
        sma50_s = _sma(close, 50)
        cur_sma20 = _safe(sma20_s.iloc[-1])
        cur_sma50 = _safe(sma50_s.iloc[-1])
        price_vs_sma20 = ("Above" if cur_sma20 and current_price > cur_sma20 else "Below")
        price_vs_sma50 = ("Above" if cur_sma50 and current_price > cur_sma50 else "Below")

        # ── Volume anomaly ──────────────────────────────────────────────────
        today_vol  = int(vol.iloc[-1]) if not pd.isna(vol.iloc[-1]) else 0
        avg_vol_20 = float(vol.iloc[-21:-1].mean()) if len(vol) >= 21 else float(vol.mean())
        vol_ratio  = round(today_vol / avg_vol_20, 2) if avg_vol_20 > 0 else None

        if vol_ratio is None:
            vol_signal = "Normal"
        elif vol_ratio >= 2.0:
            vol_signal = "High Volume"
        elif vol_ratio >= 1.5:
            vol_signal = "Elevated"
        else:
            vol_signal = "Normal"

        # ── VWAP (intraday 1-min) ────────────────────────────────────────────
        vwap, vwap_signal = _fetch_vwap(ticker, current_price)

        # ── Support / Resistance ─────────────────────────────────────────────
        sr_levels = _find_sr(high, low, n_levels=3)

        # ── Daily bars with all indicators for chart rendering ───────────────
        daily_bars = _build_daily_bars(
            df, sma20_s, sma50_s,
            bb_up, bb_mid, bb_low,
            macd_line, sig_line, hist_s,
        )

        return {
            "ticker":          ticker,
            "current_price":   current_price,
            "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M"),
            # RSI
            "rsi":             cur_rsi,
            "rsi_signal":      rsi_signal,
            # MACD
            "macd_line":       cur_macd,
            "macd_signal_line": cur_sig,
            "macd_hist":       cur_hist,
            "macd_signal":     macd_signal,
            # Bollinger Bands
            "bb_upper":        cur_bb_up,
            "bb_middle":       cur_bb_mid,
            "bb_lower":        cur_bb_low,
            "bb_signal":       bb_signal,
            # VWAP
            "vwap":            vwap,
            "vwap_signal":     vwap_signal,
            # Volume
            "today_volume":    today_vol,
            "avg_volume_20d":  round(avg_vol_20, 0),
            "volume_ratio":    vol_ratio,
            "volume_signal":   vol_signal,
            # Moving Averages
            "sma20":           cur_sma20,
            "sma50":           cur_sma50,
            "price_vs_sma20":  price_vs_sma20,
            "price_vs_sma50":  price_vs_sma50,
            # Support / Resistance
            "support_resistance": sr_levels,
            # Full daily bars for chart (list of dicts — st.cache_data safe)
            "daily_bars":      daily_bars,
            "error":           None,
        }

    except Exception as exc:
        logger.error("get_technicals failed for %s: %s", ticker, exc)
        return _empty(ticker, str(exc))


# ── Indicator implementations (pure pandas) ───────────────────────────────────

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast   = series.ewm(span=fast,   adjust=False).mean()
    ema_slow   = series.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bbands(series: pd.Series, period: int = 20, std: float = 2.0):
    middle = series.rolling(period).mean()
    sigma  = series.rolling(period).std(ddof=0)
    return middle + std * sigma, middle, middle - std * sigma


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def _fetch_vwap(ticker: str, current_price: float):
    """
    Compute today's VWAP from 1-min intraday bars.
    Returns (vwap_value, signal_string) or (None, None) on failure.
    """
    try:
        df1m = yf.Ticker(ticker).history(period="1d", interval="1m")
        if df1m.empty or len(df1m) < 2:
            return None, None
        typical  = (df1m["High"] + df1m["Low"] + df1m["Close"]) / 3
        cum_vol  = df1m["Volume"].cumsum()
        cum_tpv  = (typical * df1m["Volume"]).cumsum()
        vwap_val = float(cum_tpv.iloc[-1] / cum_vol.iloc[-1])
        signal   = "Above VWAP" if current_price > vwap_val else "Below VWAP"
        return round(vwap_val, 2), signal
    except Exception as exc:
        logger.warning("VWAP fetch failed for %s: %s", ticker, exc)
        return None, None


def _find_sr(high: pd.Series, low: pd.Series,
             tolerance: float = 0.015, n_levels: int = 3) -> list:
    """
    Find support/resistance levels by clustering local highs and lows.
    tolerance: two prices within this fraction of each other are merged.
    Returns up to n_levels centroid prices, sorted by number of touches.
    """
    highs = high.values
    lows  = low.values
    n     = len(highs)
    levels = []

    for i in range(1, n - 1):
        if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            levels.append(float(highs[i]))
        if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            levels.append(float(lows[i]))

    if not levels:
        return []

    levels.sort()
    clusters, current = [], [levels[0]]
    for lv in levels[1:]:
        if (lv - current[0]) / max(current[0], 1e-10) < tolerance:
            current.append(lv)
        else:
            clusters.append(current)
            current = [lv]
    clusters.append(current)

    clusters.sort(key=len, reverse=True)
    return [round(sum(c) / len(c), 2) for c in clusters[:n_levels]]


def _build_daily_bars(df, sma20, sma50, bb_up, bb_mid, bb_low,
                      macd_line, macd_sig, macd_hist) -> list:
    """
    Convert daily OHLCV + indicator series into a list of dicts
    suitable for Plotly charting (st.cache_data-safe, no DataFrames).
    NaN values become None.
    """
    bars = []
    for i in range(len(df)):
        row     = df.iloc[i]
        dt      = df.index[i]
        date_str = str(dt.date()) if hasattr(dt, "date") else str(dt)[:10]
        bars.append({
            "date":        date_str,
            "open":        _safe(row["Open"]),
            "high":        _safe(row["High"]),
            "low":         _safe(row["Low"]),
            "close":       _safe(row["Close"]),
            "volume":      int(row["Volume"]) if not pd.isna(row["Volume"]) else 0,
            "sma20":       _safe(sma20.iloc[i]),
            "sma50":       _safe(sma50.iloc[i]),
            "bb_upper":    _safe(bb_up.iloc[i]),
            "bb_middle":   _safe(bb_mid.iloc[i]),
            "bb_lower":    _safe(bb_low.iloc[i]),
            "macd":        _safe(macd_line.iloc[i]),
            "macd_signal": _safe(macd_sig.iloc[i]),
            "macd_hist":   _safe(macd_hist.iloc[i]),
        })
    return bars


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe(val) -> float | None:
    """Float cast with NaN/Inf → None."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)
    except (TypeError, ValueError):
        return None


def _empty(ticker: str, error: str) -> dict:
    return {
        "ticker": ticker, "current_price": None, "timestamp": None,
        "rsi": None, "rsi_signal": "Neutral",
        "macd_line": None, "macd_signal_line": None,
        "macd_hist": None, "macd_signal": "Neutral",
        "bb_upper": None, "bb_middle": None, "bb_lower": None,
        "bb_signal": "Inside Bands",
        "vwap": None, "vwap_signal": None,
        "today_volume": None, "avg_volume_20d": None,
        "volume_ratio": None, "volume_signal": "Normal",
        "sma20": None, "sma50": None,
        "price_vs_sma20": "N/A", "price_vs_sma50": "N/A",
        "support_resistance": [], "daily_bars": [],
        "error": error,
    }
