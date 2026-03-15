"""
Market regime detector.

Determines if broad market is risk-on, risk-off, or neutral
by analyzing SPY/QQQ trends, VIX level, and breadth.

Public API:
    get_market_regime() -> dict
"""

import logging
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def get_market_regime() -> dict:
    """
    Analyze SPY, QQQ, and VIX to determine market regime.

    Returns:
        {regime, color, spy_trend, qqq_trend, vix_level, vix_value,
         spy_above_200sma, qqq_above_200sma, spy_above_50sma, qqq_above_50sma,
         warning, detail}
    """
    result = {
        "regime": "Unknown", "color": "gray",
        "spy_trend": "?", "qqq_trend": "?",
        "vix_level": "?", "vix_value": 0,
        "spy_above_200sma": None, "qqq_above_200sma": None,
        "spy_above_50sma": None, "qqq_above_50sma": None,
        "warning": "", "detail": "",
    }

    try:
        batch = yf.download(["SPY", "QQQ", "^VIX"], period="1y", interval="1d",
                            progress=False, threads=True)
        if batch.empty:
            result["warning"] = "Could not fetch market data"
            return result
        # Extract per-ticker DataFrames from multi-level columns
        tickers_in_data = batch.columns.get_level_values(1).unique() if isinstance(
            batch.columns, pd.MultiIndex) else []
        spy = batch.xs("SPY", level=1, axis=1) if "SPY" in tickers_in_data else pd.DataFrame()
        qqq = batch.xs("QQQ", level=1, axis=1) if "QQQ" in tickers_in_data else pd.DataFrame()
        vix = batch.xs("^VIX", level=1, axis=1) if "^VIX" in tickers_in_data else pd.DataFrame()
    except Exception as exc:
        logger.error("market_regime fetch failed: %s", exc)
        result["warning"] = "Could not fetch market data"
        return result

    # SPY analysis
    if not spy.empty and len(spy) > 200:
        spy_close = spy["Close"].iloc[-1]
        spy_sma50 = spy["Close"].rolling(50).mean().iloc[-1]
        spy_sma200 = spy["Close"].rolling(200).mean().iloc[-1]
        spy_5d_chg = (spy_close - spy["Close"].iloc[-6]) / spy["Close"].iloc[-6] * 100 if len(spy) > 5 else 0

        result["spy_above_50sma"] = spy_close > spy_sma50
        result["spy_above_200sma"] = spy_close > spy_sma200
        result["spy_price"] = round(spy_close, 2)
        result["spy_5d_chg"] = round(spy_5d_chg, 2)

        if spy_close > spy_sma50 > spy_sma200:
            result["spy_trend"] = "Strong Uptrend"
        elif spy_close > spy_sma200:
            result["spy_trend"] = "Uptrend"
        elif spy_close > spy_sma50:
            result["spy_trend"] = "Recovering"
        else:
            result["spy_trend"] = "Downtrend"

    # QQQ analysis
    if not qqq.empty and len(qqq) > 200:
        qqq_close = qqq["Close"].iloc[-1]
        qqq_sma50 = qqq["Close"].rolling(50).mean().iloc[-1]
        qqq_sma200 = qqq["Close"].rolling(200).mean().iloc[-1]

        result["qqq_above_50sma"] = qqq_close > qqq_sma50
        result["qqq_above_200sma"] = qqq_close > qqq_sma200
        result["qqq_price"] = round(qqq_close, 2)

        if qqq_close > qqq_sma50 > qqq_sma200:
            result["qqq_trend"] = "Strong Uptrend"
        elif qqq_close > qqq_sma200:
            result["qqq_trend"] = "Uptrend"
        elif qqq_close > qqq_sma50:
            result["qqq_trend"] = "Recovering"
        else:
            result["qqq_trend"] = "Downtrend"

    # VIX analysis
    if not vix.empty:
        vix_val = vix["Close"].iloc[-1]
        result["vix_value"] = round(vix_val, 2)
        if vix_val < 15:
            result["vix_level"] = "Low"
        elif vix_val < 20:
            result["vix_level"] = "Normal"
        elif vix_val < 30:
            result["vix_level"] = "Elevated"
        else:
            result["vix_level"] = "High"

    # Determine overall regime
    bullish_signals = 0
    bearish_signals = 0

    if result["spy_above_200sma"]:
        bullish_signals += 1
    else:
        bearish_signals += 1
    if result["spy_above_50sma"]:
        bullish_signals += 1
    else:
        bearish_signals += 1
    if result["qqq_above_200sma"]:
        bullish_signals += 1
    else:
        bearish_signals += 1
    if result["qqq_above_50sma"]:
        bullish_signals += 1
    else:
        bearish_signals += 1

    vix_val = result["vix_value"]
    if vix_val < 18:
        bullish_signals += 1
    elif vix_val > 25:
        bearish_signals += 1

    if bullish_signals >= 4:
        result["regime"] = "RISK ON"
        result["color"] = "green"
        result["detail"] = "Market trending up — SPY & QQQ above key moving averages, VIX contained."
    elif bearish_signals >= 4:
        result["regime"] = "RISK OFF"
        result["color"] = "red"
        result["warning"] = "Market trending down — long setups have higher failure rate."
        result["detail"] = "SPY & QQQ below key moving averages. Consider reducing position size or waiting."
    elif bearish_signals >= 3:
        result["regime"] = "CAUTION"
        result["color"] = "yellow"
        result["warning"] = "Mixed signals — market not clearly trending. Be selective with entries."
        result["detail"] = "Some indices below moving averages. Tighten stops and be selective."
    else:
        result["regime"] = "NEUTRAL"
        result["color"] = "yellow"
        result["detail"] = "Market mixed — some bullish, some bearish signals."

    return result
