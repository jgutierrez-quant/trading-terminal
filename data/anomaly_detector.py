"""
Anomaly detector — combines technical signals + sentiment scores to score tickers.

A ticker is flagged as 'Watch' when 3+ signals align AND one direction
strictly outnumbers the other (Stage 9c: threshold back to 3, direction logic preserved).

When check_factor_model=True (Stage 11), the watch flag is driven by a composite
score: factor model (60%) + technical alignment (25%) + sentiment (15%).
Watch triggers when composite > 70 (Long) or < 30 (Short).

Public API:
    compute_anomaly(ticker, tech_dict, sentiment_dict,
                    check_earnings=False, check_sector=False,
                    check_fundamentals=False,
                    check_factor_model=False,
                    check_catalysts=False) -> dict

Returns:
    {
        "ticker":              str,
        "score":               int,        # total signals aligned
        "is_watch":            bool,       # True when score >= WATCH_THRESHOLD AND direction ok
        "reason":              str,        # plain English e.g. "Oversold RSI + High Volume"
        "signals":             list[str],  # individual signal labels
        "quality_score":       int,        # 0-100 quality score
        "direction":           str,        # "Long" or "Short"
        "earnings_proximity":  bool,       # True if earnings within 5 days
        "earnings_date":       str|None,   # next earnings date if near
        "sector_momentum":     bool,       # True if sector ETF agrees with direction
        "fundamental_score":   int|None,   # -100..+100 (None when check_fundamentals=False)
        "fundamental_signal":  str|None,   # "Bullish"/"Bearish"/"Neutral" or None
        "fundamental_reasons": list|None,  # up to 5 reason strings or None
        # Stage 11 — factor model fields (None when check_factor_model=False)
        "factor_model":        dict|None,  # full compute_factor_model() output
        "composite_factor_score": float|None,  # 0-100 weighted composite
        "pead_candidate":      bool|None,  # True if active PEAD opportunity
        # Stage 12 — catalyst detector fields (None when check_catalysts=False)
        "catalyst_result":     dict|None,  # full detect_catalysts() output
        "catalyst_boost":      int,        # net boost -100..+100
        "catalyst_why":        list[str],  # plain English reasons
    }
"""

import logging

import pandas as pd
import yfinance as yf

from data.fundamentals import get_fundamentals, score_fundamentals

logger = logging.getLogger(__name__)

from utils.config import WATCH_THRESHOLD  # 3 aligned signals (Stage 9c)

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
    'Bullish Fundamentals', 'Bullish Catalyst',
}
_BEARISH = {
    'Overbought RSI', 'Bearish MACD Cross', 'BB Overbought',
    'Below VWAP', 'Price Below MA20+MA50', 'Bearish News Sentiment',
    'Bearish Fundamentals', 'Bearish Catalyst',
}


# ── Public entry point ────────────────────────────────────────────────────────

def compute_anomaly(
    ticker: str,
    tech: dict,
    sentiment: dict,
    check_earnings: bool = False,
    check_sector: bool = False,
    check_fundamentals: bool = False,
    check_factor_model: bool = False,
    check_catalysts: bool = False,
) -> dict:
    """
    Evaluate technical and sentiment signals for `ticker`.

    Args:
        ticker:              uppercase ticker string
        tech:                dict returned by data.technicals.get_technicals()
        sentiment:           dict returned by sentiment.sentiment_aggregator.get_sentiment()
        check_earnings:      if True, fetch yfinance calendar and flag earnings < 5 days
        check_sector:        if True, fetch sector ETF trend and add 'Sector Momentum' signal
        check_fundamentals:  if True, fetch fundamentals and fold into quality_score
        check_factor_model:  if True, run the Stage 11 6-factor model and override is_watch
                             with a composite score (factor 60% + tech 25% + sentiment 15%)

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

    # ── Fundamentals (optional) ────────────────────────────────────────────────
    fundamental_score   = None
    fundamental_signal  = None
    fundamental_reasons = None
    fund_dict           = None
    if check_fundamentals:
        try:
            fund_dict    = get_fundamentals(ticker)
            fund_scored  = score_fundamentals(fund_dict)
            fundamental_score   = fund_scored["fundamental_score"]
            fundamental_signal  = fund_scored["fundamental_signal"]
            fundamental_reasons = fund_scored["fundamental_reasons"]
            if fundamental_signal == "Bullish":
                signals.append("Bullish Fundamentals")
            elif fundamental_signal == "Bearish":
                signals.append("Bearish Fundamentals")
        except Exception as exc:
            logger.debug("Fundamentals check failed for %s: %s", ticker, exc)

    # Re-compute bull/bear counts after fundamentals signal may have been added
    bull_count = sum(1 for s in signals if s in _BULLISH)
    bear_count = sum(1 for s in signals if s in _BEARISH)

    if bull_count > bear_count:
        direction = "Long"
    elif bear_count > bull_count:
        direction = "Short"
    else:
        direction = "Neutral"

    # ── Quality score (0-100) — rewards aligned directional signals ──────────
    total_directional = bull_count + bear_count
    dominant  = max(bull_count, bear_count)
    alignment = dominant / total_directional if total_directional > 0 else 0.0

    vol_pts = 20 if vol_sig == "High Volume" else (10 if vol_sig == "Elevated" else 0)

    if fundamental_score is not None:
        # Composite formula when fundamentals available: max 115 → clamped to 100
        tech_score  = min(dominant / 5.0, 1.0) * 35
        align_score = alignment * 35
        fund_pts    = (fundamental_score + 100) / 200 * 25
        quality_score = int(min(tech_score + align_score + vol_pts + fund_pts, 100))
    else:
        sig_ct_score  = min(dominant / 5.0, 1.0) * 40
        quality_score = int(round(sig_ct_score + alignment * 40 + vol_pts))

    # ── Factor model (optional — Stage 11) ────────────────────────────────────
    factor_model_result   = None
    composite_factor_score = None
    pead_candidate        = None
    if check_factor_model:
        try:
            from data.factor_model import compute_factor_model
            # Pass pre-fetched fundamentals dict if available to avoid duplicate call
            fm_fund = fund_dict if check_fundamentals else None
            factor_model_result = compute_factor_model(ticker, fund_dict=fm_fund)

            # Build composite: factor model (60%) + technical alignment (25%) + sentiment (15%)
            fm_score = factor_model_result.get("composite_score", 50.0)

            # Technical alignment: 0-100 (fraction of directional signals in dominant direction,
            # scaled by signal density — rewards both quantity and consistency)
            if total_directional > 0:
                tech_pct = alignment * min(dominant / 5.0, 1.0) * 100
            else:
                tech_pct = 50.0  # neutral when no directional signals

            # Sentiment: map overall_sentiment (-1..+1) to 0..100
            overall_sent = sentiment.get("overall_sentiment", 0.0) or 0.0
            sent_pct = (overall_sent + 1.0) / 2.0 * 100

            composite_factor_score = round(
                fm_score * 0.60 + tech_pct * 0.25 + sent_pct * 0.15, 1
            )

            pead_candidate = factor_model_result.get("pead_candidate", False)
        except Exception as exc:
            logger.debug("Factor model check failed for %s: %s", ticker, exc)

    # ── Catalyst detection (optional — Stage 12) ───────────────────────────────
    catalyst_result = None
    catalyst_boost  = 0
    catalyst_why    = []
    if check_catalysts:
        try:
            from data.catalyst_detector import detect_catalysts
            catalyst_result = detect_catalysts(ticker)
            catalyst_boost  = catalyst_result.get("boost", 0)
            catalyst_why    = catalyst_result.get("why", [])
            cat_dir = catalyst_result.get("direction", "Neutral")
            if cat_dir == "Bullish":
                signals.append("Bullish Catalyst")
            elif cat_dir == "Bearish":
                signals.append("Bearish Catalyst")
        except Exception as exc:
            logger.debug("Catalyst check failed for %s: %s", ticker, exc)

    # ── Sector momentum (optional — adds to score) ────────────────────────────
    sector_ok = False
    if check_sector:
        sector_ok = _check_sector_momentum(ticker, direction)
        if sector_ok:
            signals.append("Sector Momentum")

    # ── Final score and Watch flag ────────────────────────────────────────────
    score = len(signals)

    if composite_factor_score is not None:
        # Stage 11: composite-driven watch flag
        is_watch = composite_factor_score > 70 or composite_factor_score < 30
        # Update direction to match factor model when available
        if composite_factor_score > 70:
            direction = "Long"
        elif composite_factor_score < 30:
            direction = "Short"
        # else: keep direction from technical signals
    else:
        # Legacy: total signals >= threshold AND one direction strictly outnumbers the other
        is_watch = score >= WATCH_THRESHOLD and (bull_count > bear_count or bear_count > bull_count)

    reason = " + ".join(signals[:5]) if signals else "No notable signals"

    # ── Earnings proximity (informational — does NOT affect score or is_watch) ─
    earnings_near = False
    earnings_date = None
    if check_earnings:
        earnings_near, earnings_date = _check_earnings_proximity(ticker)

    return {
        "ticker":                 ticker,
        "score":                  score,
        "is_watch":               is_watch,
        "reason":                 reason,
        "signals":                signals,
        "quality_score":          quality_score,
        "direction":              direction,
        "earnings_proximity":     earnings_near,
        "earnings_date":          earnings_date,
        "sector_momentum":        sector_ok,
        "fundamental_score":      fundamental_score,
        "fundamental_signal":     fundamental_signal,
        "fundamental_reasons":    fundamental_reasons,
        # Stage 11 — factor model fields
        "factor_model":           factor_model_result,
        "composite_factor_score": composite_factor_score,
        "pead_candidate":         pead_candidate,
        # Stage 12 — catalyst fields
        "catalyst_result":        catalyst_result,
        "catalyst_boost":         catalyst_boost,
        "catalyst_why":           catalyst_why,
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
