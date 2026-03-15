"""
Anomaly detector — combines technical signals + sentiment scores to score tickers.

A ticker is flagged as 'Watch' when 4+ signals align AND the signal direction
agrees with the 50-day MA trend (Stage 9: raised from 3, direction filter added).

Public API:
    compute_anomaly(ticker, tech_dict, sentiment_dict,
                    check_earnings=False, check_sector=False) -> dict

Returns:
    {
        "ticker":            str,
        "score":             int,        # total signals aligned
        "is_watch":          bool,       # True when score >= WATCH_THRESHOLD AND direction ok
        "reason":            str,        # plain English e.g. "Oversold RSI + High Volume"
        "signals":           list[str],  # individual signal labels
        "quality_score":     int,        # 0-100 quality score
        "direction":         str,        # "Long" or "Short"
        "earnings_proximity": bool,      # True if earnings within 5 days
        "earnings_date":     str|None,   # next earnings date if near
        "sector_momentum":   bool,       # True if sector ETF agrees with direction
    }
"""

import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

WATCH_THRESHOLD = 4   # raised from 3 — higher conviction setups only

# ── Sector ETF mapping (mirrors sector_monitor.py SECTOR_ETFS) ─────────────────
_SECTOR_ETF = {
    'Technology':             'XLK',
    'Health Care':            'XLV',
    'Financials':             'XLF',
    'Consumer Discretionary': 'XLY',
    'Consumer Staples':       'XLP',
    'Industrials':            'XLI',
    'Energy':                 'XLE',
    'Materials':              'XLB',
    'Real Estate':            'XLRE',
    'Communication Services': 'XLC',
    'Utilities':              'XLU',
}

# Signal direction classification
_BULLISH = {
    'Oversold RSI', 'Bullish MACD Cross', 'BB Oversold',
    'Above VWAP', 'Price Above MA20+MA50', 'Bullish News Sentiment',
}
_BEARISH = {
    'Overbought RSI', 'Bearish MACD Cross', 'BB Overbought',
    'Below VWAP', 'Price Below MA20+MA50', 'Bearish News Sentiment',
}


# ── Public entry point ────────────────────────────────────────────────────────

def compute_anomaly(
    ticker: str,
    tech: dict,
    sentiment: dict,
    check_earnings: bool = False,
    check_sector: bool = False,
) -> dict:
    """
    Evaluate technical and sentiment signals for `ticker`.

    Args:
        ticker:         uppercase ticker string
        tech:           dict returned by data.technicals.get_technicals()
        sentiment:      dict returned by sentiment.sentiment_aggregator.get_sentiment()
        check_earnings: if True, fetch yfinance calendar and flag earnings < 5 days
        check_sector:   if True, fetch sector ETF trend and add 'Sector Momentum' signal

    Direction filter (Stage 9):
        Long signals only count as Watch when price > SMA50.
        Short signals only count as Watch when price < SMA50.
    """
    ticker = ticker.upper()
    signals = []

    # ── Technical signals ─────────────────────────────────────────────────────

    rsi_sig = tech.get("rsi_signal", "")
    if rsi_sig == "Oversold":
        signals.append("Oversold RSI")
    elif rsi_sig == "Overbought":
        signals.append("Overbought RSI")

    macd_sig = tech.get("macd_signal", "")
    if macd_sig == "Bullish Cross":
        signals.append("Bullish MACD Cross")
    elif macd_sig == "Bearish Cross":
        signals.append("Bearish MACD Cross")

    bb_sig = tech.get("bb_signal", "")
    if bb_sig == "Below Lower":
        signals.append("BB Oversold")
    elif bb_sig == "Above Upper":
        signals.append("BB Overbought")

    vol_sig = tech.get("volume_signal", "")
    if vol_sig == "High Volume":
        signals.append("High Volume")
    elif vol_sig == "Elevated":
        signals.append("Elevated Volume")

    vwap_sig = tech.get("vwap_signal", "")
    if vwap_sig == "Below VWAP":
        signals.append("Below VWAP")
    elif vwap_sig == "Above VWAP":
        signals.append("Above VWAP")

    sma20 = tech.get("price_vs_sma20", "")
    sma50 = tech.get("price_vs_sma50", "")
    if sma20 == "Above" and sma50 == "Above":
        signals.append("Price Above MA20+MA50")
    elif sma20 == "Below" and sma50 == "Below":
        signals.append("Price Below MA20+MA50")

    # ── Sentiment signals ─────────────────────────────────────────────────────

    sent_label = sentiment.get("sentiment_label", "")
    if sent_label == "Bullish":
        signals.append("Bullish News Sentiment")
    elif sent_label == "Bearish":
        signals.append("Bearish News Sentiment")

    # ── Direction from bull/bear signal balance ───────────────────────────────

    bull_count = sum(1 for s in signals if s in _BULLISH)
    bear_count = sum(1 for s in signals if s in _BEARISH)

    if bull_count > bear_count:
        direction = "Long"
    elif bear_count > bull_count:
        direction = "Short"
    else:
        direction = "Neutral"   # tied — neither side triggers a Watch

    # ── Quality score (0-100) — rewards aligned directional signals ──────────
    total_directional = bull_count + bear_count
    dominant  = max(bull_count, bear_count)
    alignment = dominant / total_directional if total_directional > 0 else 0.0

    vol_pts      = 20 if vol_sig == "High Volume" else (10 if vol_sig == "Elevated" else 0)
    sig_ct_score = min(dominant / 5.0, 1.0) * 40   # 5 directional signals max
    quality_score = int(round(sig_ct_score + alignment * 40 + vol_pts))

    # ── Sector momentum (optional — adds to score) ────────────────────────────
    sector_ok = False
    if check_sector:
        sector_ok = _check_sector_momentum(ticker, direction)
        if sector_ok:
            signals.append("Sector Momentum")

    # ── Final score and Watch flag ────────────────────────────────────────────
    score = len(signals)

    # Watch: total signals >= threshold AND one direction strictly outnumbers the other
    is_watch = score >= WATCH_THRESHOLD and (bull_count > bear_count or bear_count > bull_count)

    reason = " + ".join(signals[:5]) if signals else "No notable signals"

    # ── Earnings proximity (informational — does NOT affect score or is_watch) ─
    earnings_near = False
    earnings_date = None
    if check_earnings:
        earnings_near, earnings_date = _check_earnings_proximity(ticker)

    return {
        "ticker":             ticker,
        "score":              score,
        "is_watch":           is_watch,
        "reason":             reason,
        "signals":            signals,
        "quality_score":      quality_score,
        "direction":          direction,
        "earnings_proximity": earnings_near,
        "earnings_date":      earnings_date,
        "sector_momentum":    sector_ok,
    }


# ── Private helpers ───────────────────────────────────────────────────────────

def _check_earnings_proximity(ticker: str, days: int = 5):
    """
    Return (is_near, date_str) — True if next earnings date is within `days` days.
    Uses yfinance earnings_dates; fails silently if unavailable.
    """
    try:
        ed = yf.Ticker(ticker).earnings_dates
        if ed is None or ed.empty:
            return False, None
        today = pd.Timestamp.now(tz="UTC").normalize()
        for dt in ed.index:
            try:
                dt_utc = dt.tz_convert("UTC") if dt.tzinfo else pd.Timestamp(dt, tz="UTC")
                delta  = (dt_utc.normalize() - today).days
                if 0 <= delta <= days:
                    return True, str(dt_utc.date())
            except Exception:
                continue
    except Exception as exc:
        logger.debug("Earnings check failed for %s: %s", ticker, exc)
    return False, None


def _check_sector_momentum(ticker: str, direction: str) -> bool:
    """
    Return True if the ticker's sector ETF trend agrees with `direction`.
    Sector trend is up when SMA20 > SMA50, down when SMA20 < SMA50.
    Fails silently — returns False if sector cannot be determined.
    """
    try:
        info   = yf.Ticker(ticker).info
        sector = info.get("sector", "")
        etf    = _SECTOR_ETF.get(sector)
        if not etf:
            return False

        df = yf.Ticker(etf).history(period="60d", interval="1d")
        if df.empty or len(df) < 50:
            return False

        close  = df["Close"]
        sma20  = float(close.rolling(20).mean().iloc[-1])
        sma50  = float(close.rolling(50).mean().iloc[-1])
        etf_up = sma20 > sma50

        return (direction == "Long" and etf_up) or (direction == "Short" and not etf_up)
    except Exception as exc:
        logger.debug("Sector momentum check failed for %s: %s", ticker, exc)
    return False
