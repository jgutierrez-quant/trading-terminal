"""
Trade coach — real-time trade setup analysis with plain English coaching.

Combines all terminal signals (technicals, anomaly, fundamentals, factor model,
catalysts, whale activity) into a single actionable trade recommendation.

For each ticker, produces:
    - Trade or No-Trade verdict
    - Direction (Long / Short / Stay Out)
    - Entry, stop-loss, take-profit levels
    - Position size (based on account value + risk)
    - Risk/reward ratio
    - Confidence score (0-100)
    - Plain English coaching — WHY to take the trade, what to watch for

Public API:
    analyze_setup(ticker, account_value=100000, risk_pct=1.0) -> dict
"""

import logging
import math
from datetime import datetime

import yfinance as yf

logger = logging.getLogger(__name__)


def analyze_setup(
    ticker: str,
    account_value: float = 100_000,
    risk_pct: float = 1.0,
    technicals: dict = None,
    anomaly: dict = None,
    whale: dict = None,
) -> dict:
    """
    Full trade setup analysis with coaching.

    Args:
        ticker:        Stock symbol.
        account_value: Portfolio value for position sizing.
        risk_pct:      Percent of account to risk per trade (default 1%).
        technicals:    Pre-fetched technicals dict (optional, fetches if None).
        anomaly:       Pre-fetched anomaly dict (optional).
        whale:         Pre-fetched whale dict (optional).

    Returns dict with:
        ticker, verdict, direction, confidence, entry, stop_loss, take_profit,
        risk_reward, position_size, dollar_risk, signals_for, signals_against,
        coaching (list of plain English coaching lines), grade (A/B/C/D/F)
    """
    ticker = ticker.upper().strip()

    # ── Fetch data if not provided ───────────────────────────────────────────
    if technicals is None:
        from data.technicals import get_technicals
        technicals = get_technicals(ticker)

    if anomaly is None:
        from data.anomaly_detector import compute_anomaly
        sent = {"sentiment_label": "Neutral", "overall_sentiment": 0.0}
        anomaly = compute_anomaly(
            ticker, technicals, sent,
            check_earnings=True, check_sector=True,
            check_fundamentals=True, check_factor_model=True,
            check_catalysts=True,
        )

    if whale is None:
        try:
            from data.whale_detector import detect_whales
            whale = detect_whales(ticker)
        except Exception:
            whale = {"whale_score": 0, "whale_direction": "Neutral", "whale_signals": []}

    # ── Extract key data ─────────────────────────────────────────────────────
    price = technicals.get("current_price") or 0
    rsi = technicals.get("rsi")
    macd_sig = technicals.get("macd_signal", "Neutral")
    bb_sig = technicals.get("bb_signal", "Neutral")
    vwap = technicals.get("vwap")
    vwap_sig = technicals.get("vwap_signal", "Neutral")
    vol_ratio = technicals.get("volume_ratio") or 1.0
    sma20 = technicals.get("sma20")
    sma50 = technicals.get("sma50")
    support_resistance = technicals.get("support_resistance") or []

    an_direction = anomaly.get("direction", "Neutral")
    an_score = anomaly.get("score", 0)
    quality = anomaly.get("quality_score", 0)
    factor_score = anomaly.get("composite_factor_score")
    cat_boost = anomaly.get("catalyst_boost", 0)
    cat_why = anomaly.get("catalyst_why", [])
    fund_result = anomaly.get("fundamental_result")

    whale_score = whale.get("whale_score", 0)
    whale_dir = whale.get("whale_direction", "Neutral")
    whale_signals = whale.get("whale_signals", [])
    whale_alert = whale.get("alert_level", "None")

    if price <= 0:
        return _no_trade(ticker, "No price data available")

    # ── Count signals for and against ────────────────────────────────────────
    signals_for = []
    signals_against = []
    coaching = []

    # RSI
    if rsi is not None:
        if rsi < 30:
            signals_for.append(f"RSI oversold ({rsi:.0f}) — stock is beaten down, bounce likely")
        elif rsi > 70:
            signals_against.append(f"RSI overbought ({rsi:.0f}) — risky to chase here")
        elif rsi < 40:
            signals_for.append(f"RSI approaching oversold ({rsi:.0f})")
        elif rsi > 60:
            signals_against.append(f"RSI elevated ({rsi:.0f}) — less upside room")

    # MACD
    if "Bullish" in macd_sig:
        signals_for.append("MACD bullish — momentum shifting up")
    elif "Bearish" in macd_sig:
        signals_against.append("MACD bearish — momentum fading")

    # Bollinger Bands
    if bb_sig == "Oversold":
        signals_for.append("Price below lower Bollinger Band — stretched to downside")
    elif bb_sig == "Overbought":
        signals_against.append("Price above upper Bollinger Band — stretched to upside")

    # VWAP
    if vwap and price:
        if vwap_sig == "Above VWAP":
            signals_for.append(f"Trading above VWAP (${vwap:.2f}) — institutional buyers in control")
        elif vwap_sig == "Below VWAP":
            signals_against.append(f"Trading below VWAP (${vwap:.2f}) — sellers in control")

    # Moving averages
    if sma20 and sma50 and price:
        if price > sma20 > sma50:
            signals_for.append("Price > SMA20 > SMA50 — strong uptrend structure")
        elif price < sma20 < sma50:
            signals_against.append("Price < SMA20 < SMA50 — downtrend structure")
        elif price > sma20 and price < sma50:
            signals_for.append("Price reclaimed SMA20 — potential trend reversal starting")

    # Volume
    if vol_ratio >= 2.0:
        signals_for.append(f"Volume {vol_ratio:.1f}x average — strong participation")
    elif vol_ratio < 0.5:
        signals_against.append("Low volume — weak conviction, moves may not hold")

    # Quality score
    if quality >= 80:
        signals_for.append(f"Signal quality {quality}/100 — high confidence setup")
    elif quality >= 60:
        signals_for.append(f"Signal quality {quality}/100 — decent setup")
    elif quality < 40:
        signals_against.append(f"Signal quality {quality}/100 — weak/noisy signals")

    # Factor model
    if factor_score is not None:
        if factor_score >= 70:
            signals_for.append(f"Factor model score {factor_score:.0f}/100 — quant factors align")
        elif factor_score <= 30:
            signals_against.append(f"Factor model score {factor_score:.0f}/100 — quant headwinds")

    # Catalysts
    if cat_boost > 10:
        signals_for.append(f"Active bullish catalyst (+{cat_boost})")
        for c in cat_why[:2]:
            signals_for.append(f"  -> {c}")
    elif cat_boost < -10:
        signals_against.append(f"Active bearish catalyst ({cat_boost})")
        for c in cat_why[:2]:
            signals_against.append(f"  -> {c}")

    # Whale activity
    if whale_alert in ("Whale Alert", "Alert"):
        if whale_dir == "Bullish":
            signals_for.append(f"Whale activity: {whale_dir} (score {whale_score:+d})")
            for ws in whale_signals[:2]:
                signals_for.append(f"  -> {ws}")
        elif whale_dir == "Bearish":
            signals_against.append(f"Whale activity: {whale_dir} (score {whale_score:+d})")
            for ws in whale_signals[:2]:
                signals_against.append(f"  -> {ws}")
    elif whale_alert == "Watch":
        note = f"Whale watch: {whale_dir} (score {whale_score:+d})"
        if whale_dir == "Bullish":
            signals_for.append(note)
        elif whale_dir == "Bearish":
            signals_against.append(note)

    # ── Determine direction ──────────────────────────────────────────────────
    bull_count = len(signals_for)
    bear_count = len(signals_against)

    # Weighted score: anomaly direction + whale + catalyst + signal count
    direction_score = 0
    if an_direction == "Long":
        direction_score += 30
    elif an_direction == "Short":
        direction_score -= 30

    direction_score += whale_score * 0.3
    direction_score += cat_boost * 0.2
    direction_score += (bull_count - bear_count) * 8

    if direction_score > 15:
        direction = "Long"
    elif direction_score < -15:
        direction = "Short"
    else:
        direction = "Neutral"

    # ── Confidence score ─────────────────────────────────────────────────────
    # Scale 0-100: signal alignment, quality, whale confirmation
    raw_confidence = 30  # base

    # Signal alignment
    if bull_count + bear_count > 0:
        alignment = abs(bull_count - bear_count) / (bull_count + bear_count)
        raw_confidence += alignment * 25

    # Quality boost
    raw_confidence += min(quality / 100 * 20, 20)

    # Whale confirmation (same direction as trade)
    if (direction == "Long" and whale_dir == "Bullish") or \
       (direction == "Short" and whale_dir == "Bearish"):
        raw_confidence += 15
    elif (direction == "Long" and whale_dir == "Bearish") or \
         (direction == "Short" and whale_dir == "Bullish"):
        raw_confidence -= 10

    # Volume confirmation
    if vol_ratio >= 1.5:
        raw_confidence += 5
    elif vol_ratio < 0.7:
        raw_confidence -= 5

    # Factor model alignment
    if factor_score is not None:
        if (direction == "Long" and factor_score >= 60) or \
           (direction == "Short" and factor_score <= 40):
            raw_confidence += 5

    confidence = max(0, min(100, round(raw_confidence)))

    # ── Calculate levels ─────────────────────────────────────────────────────
    # ATR for stop calculation
    atr = _get_atr(ticker)
    if atr is None or atr <= 0:
        atr = price * 0.02  # fallback: 2% of price

    # Find nearest support/resistance for smarter stops
    supports = sorted([lv for lv in support_resistance if lv < price], reverse=True)
    resistances = sorted([lv for lv in support_resistance if lv > price])

    if direction == "Long":
        # Entry: current price (market) or slightly below at support
        entry = price
        if supports:
            # If nearest support is within 1 ATR, use it as entry for limit order
            nearest_support = supports[0]
            if price - nearest_support < atr:
                entry = round(nearest_support + 0.01, 2)

        # Stop: below nearest support or 2x ATR below entry
        if supports:
            stop = round(supports[0] - atr * 0.3, 2)
        else:
            stop = round(entry - atr * 2, 2)

        # Take profit: nearest resistance or 2:1 R:R, whichever is better
        risk = abs(entry - stop)
        tp_rr = round(entry + risk * 2, 2)  # 2:1 minimum
        if resistances:
            tp_level = resistances[0]
            # Use resistance if it gives at least 1.5:1
            if risk > 0 and (tp_level - entry) / risk >= 1.5:
                take_profit = tp_level
            else:
                take_profit = tp_rr
        else:
            take_profit = tp_rr

    elif direction == "Short":
        entry = price
        if resistances:
            nearest_resistance = resistances[0]
            if nearest_resistance - price < atr:
                entry = round(nearest_resistance - 0.01, 2)

        if resistances:
            stop = round(resistances[0] + atr * 0.3, 2)
        else:
            stop = round(entry + atr * 2, 2)

        risk = abs(stop - entry)
        tp_rr = round(entry - risk * 2, 2)
        if supports:
            tp_level = supports[0]
            if risk > 0 and (entry - tp_level) / risk >= 1.5:
                take_profit = tp_level
            else:
                take_profit = tp_rr
        else:
            take_profit = tp_rr

    else:  # Neutral — no trade
        return _no_trade(ticker, "Mixed signals — no clear edge", signals_for, signals_against, confidence)

    # ── Risk/reward ratio ────────────────────────────────────────────────────
    risk_amount = abs(entry - stop)
    reward_amount = abs(take_profit - entry)
    rr_ratio = round(reward_amount / risk_amount, 2) if risk_amount > 0 else 0

    # ── Position sizing ──────────────────────────────────────────────────────
    dollar_risk = account_value * (risk_pct / 100)
    shares = int(math.floor(dollar_risk / risk_amount)) if risk_amount > 0 else 0
    max_by_cap = int(math.floor(account_value * 0.20 / entry)) if entry > 0 else 0
    shares = min(shares, max_by_cap)
    actual_dollar_risk = shares * risk_amount

    # ── Verdict ──────────────────────────────────────────────────────────────
    # Need: R:R >= 1.5, confidence >= 40, and clear direction
    if rr_ratio >= 2.0 and confidence >= 55:
        verdict = "TAKE THE TRADE"
        grade = "A"
    elif rr_ratio >= 1.5 and confidence >= 45:
        verdict = "TAKE THE TRADE"
        grade = "B"
    elif rr_ratio >= 1.5 and confidence >= 35:
        verdict = "PROCEED WITH CAUTION"
        grade = "C"
    elif rr_ratio >= 1.0:
        verdict = "RISKY — SMALL SIZE ONLY"
        grade = "D"
    else:
        verdict = "SKIP — BAD RISK/REWARD"
        grade = "F"

    # ── Build coaching lines ─────────────────────────────────────────────────
    coaching = _build_coaching(
        ticker, direction, entry, stop, take_profit, rr_ratio,
        shares, actual_dollar_risk, confidence, grade,
        signals_for, signals_against, whale, technicals, anomaly,
    )

    return {
        "ticker": ticker,
        "verdict": verdict,
        "direction": direction,
        "confidence": confidence,
        "grade": grade,
        "entry": round(entry, 2),
        "stop_loss": round(stop, 2),
        "take_profit": round(take_profit, 2),
        "risk_reward": rr_ratio,
        "risk_per_share": round(risk_amount, 2),
        "reward_per_share": round(reward_amount, 2),
        "position_size": shares,
        "dollar_risk": round(actual_dollar_risk, 2),
        "position_value": round(shares * entry, 2),
        "signals_for": signals_for,
        "signals_against": signals_against,
        "coaching": coaching,
        "whale_alert": whale.get("alert_level", "None"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def _no_trade(ticker, reason, signals_for=None, signals_against=None, confidence=0):
    """Return a no-trade result."""
    coaching = [
        f"NO TRADE on {ticker}",
        f"Reason: {reason}",
        "",
        "When signals conflict, the best trade is NO trade.",
        "Wait for a cleaner setup where most indicators agree.",
    ]
    if signals_for:
        coaching.append("")
        coaching.append("What's working:")
        coaching.extend(f"  + {s}" for s in signals_for[:3])
    if signals_against:
        coaching.append("")
        coaching.append("What's against you:")
        coaching.extend(f"  - {s}" for s in signals_against[:3])

    return {
        "ticker": ticker,
        "verdict": "NO TRADE",
        "direction": "Neutral",
        "confidence": confidence,
        "grade": "F",
        "entry": None,
        "stop_loss": None,
        "take_profit": None,
        "risk_reward": 0,
        "risk_per_share": 0,
        "reward_per_share": 0,
        "position_size": 0,
        "dollar_risk": 0,
        "position_value": 0,
        "signals_for": signals_for or [],
        "signals_against": signals_against or [],
        "coaching": coaching,
        "whale_alert": "None",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def _get_atr(ticker: str, period: int = 14) -> float | None:
    """Calculate ATR for stop-loss placement."""
    try:
        hist = yf.Ticker(ticker).history(period="30d", interval="1d")
        if hist is None or len(hist) < period + 1:
            return None
        high = hist["High"]
        low = hist["Low"]
        close = hist["Close"]
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1])
    except Exception:
        return None


# Need pandas for ATR calc
import pandas as pd


def _build_coaching(
    ticker, direction, entry, stop, take_profit, rr_ratio,
    shares, dollar_risk, confidence, grade,
    signals_for, signals_against, whale, technicals, anomaly,
) -> list[str]:
    """Build plain English coaching lines — like having a mentor next to you."""
    lines = []

    # Header
    dir_word = "BUY" if direction == "Long" else "SHORT"
    lines.append(f"=== TRADE COACH: {ticker} ===")
    lines.append("")

    # The setup
    lines.append(f"SETUP: {dir_word} {ticker} at ${entry:.2f}")
    lines.append(f"  Stop loss:   ${stop:.2f} (you lose ${abs(entry - stop):.2f}/share)")
    lines.append(f"  Take profit: ${take_profit:.2f} (you gain ${abs(take_profit - entry):.2f}/share)")
    lines.append(f"  Risk/Reward: 1:{rr_ratio:.1f}")
    lines.append("")

    # Position sizing
    lines.append(f"POSITION SIZE: {shares} shares (${shares * entry:,.0f} total)")
    lines.append(f"  If you're wrong, you lose ${dollar_risk:,.0f}")
    lines.append(f"  If you're right, you make ${shares * abs(take_profit - entry):,.0f}")
    lines.append("")

    # Why take this trade
    lines.append("WHY THIS TRADE:")
    if signals_for:
        for s in signals_for[:5]:
            lines.append(f"  + {s}")
    else:
        lines.append("  (no strong signals in favor)")
    lines.append("")

    # What to watch out for
    if signals_against:
        lines.append("WATCH OUT FOR:")
        for s in signals_against[:4]:
            lines.append(f"  - {s}")
        lines.append("")

    # Whale activity coaching
    whale_alert = whale.get("alert_level", "None") if whale else "None"
    if whale_alert in ("Whale Alert", "Alert"):
        whale_dir = whale.get("whale_direction", "Neutral")
        lines.append("WHALE ACTIVITY:")
        if whale_dir == direction.replace("Long", "Bullish").replace("Short", "Bearish"):
            lines.append("  Big money is moving in YOUR direction — good confirmation.")
        elif whale_dir != "Neutral":
            lines.append("  WARNING: Big money is moving AGAINST your trade direction.")
            lines.append("  Consider reducing size or waiting.")
        for ws in whale.get("whale_signals", [])[:3]:
            lines.append(f"  -> {ws}")
        lines.append("")

    # Trade management rules
    lines.append("TRADE MANAGEMENT RULES:")
    if direction == "Long":
        mid_target = round(entry + abs(take_profit - entry) * 0.5, 2)
        lines.append(f"  1. Set your stop at ${stop:.2f} IMMEDIATELY — never move it lower")
        lines.append(f"  2. At ${mid_target:.2f} (halfway), move stop to breakeven (${entry:.2f})")
        lines.append(f"  3. At ${take_profit:.2f}, take profit or trail stop to ${mid_target:.2f}")
        lines.append(f"  4. If it drops to ${stop:.2f}, EXIT. No hoping, no averaging down.")
    else:
        mid_target = round(entry - abs(entry - take_profit) * 0.5, 2)
        lines.append(f"  1. Set your stop at ${stop:.2f} IMMEDIATELY — never move it higher")
        lines.append(f"  2. At ${mid_target:.2f} (halfway), move stop to breakeven (${entry:.2f})")
        lines.append(f"  3. At ${take_profit:.2f}, cover or trail stop to ${mid_target:.2f}")
        lines.append(f"  4. If it hits ${stop:.2f}, EXIT. No hoping, no adding to losers.")
    lines.append("")

    # Confidence and grade
    lines.append(f"CONFIDENCE: {confidence}/100 | GRADE: {grade}")
    if grade == "A":
        lines.append("  This is a high-quality setup. All major signals align.")
        lines.append("  Full position size appropriate.")
    elif grade == "B":
        lines.append("  Good setup with most signals agreeing.")
        lines.append("  Full position size OK, but watch the against-signals closely.")
    elif grade == "C":
        lines.append("  Decent setup but some conflict.")
        lines.append("  Consider HALF position size. Be quick to exit if it goes wrong.")
    elif grade == "D":
        lines.append("  Weak setup. Only trade if you have a strong personal conviction.")
        lines.append("  Use QUARTER position size maximum.")
    else:
        lines.append("  Bad risk/reward. Skip this trade entirely.")

    lines.append("")
    lines.append("REMEMBER: The goal is not to win every trade.")
    lines.append("The goal is to win MORE than you lose, and lose SMALL when you're wrong.")

    return lines
