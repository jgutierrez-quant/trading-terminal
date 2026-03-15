"""
Anomaly detector — combines technical signals + sentiment scores to score tickers.

A ticker is flagged as 'Watch' when 3+ signals align simultaneously.
The output is a plain-English reason string suitable for display in the sidebar.

Public API:
    compute_anomaly(ticker, tech_dict, sentiment_dict) -> dict
"""

import logging

logger = logging.getLogger(__name__)

WATCH_THRESHOLD = 3


def compute_anomaly(ticker: str, tech: dict, sentiment: dict) -> dict:
    """
    Evaluate technical and sentiment signals for `ticker`.

    Args:
        ticker:    uppercase ticker string
        tech:      dict returned by data.technicals.get_technicals()
        sentiment: dict returned by sentiment.sentiment_aggregator.get_sentiment()

    Returns:
        {
            "ticker":   str,
            "score":    int,       # total signals aligned
            "is_watch": bool,      # True when score >= WATCH_THRESHOLD
            "reason":   str,       # plain English e.g. "Oversold RSI + High Volume + Bullish News"
            "signals":  list[str], # individual signal labels
        }
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

    # ── Score and Watch flag ──────────────────────────────────────────────────

    score    = len(signals)
    is_watch = score >= WATCH_THRESHOLD
    reason   = " + ".join(signals[:5]) if signals else "No notable signals"

    return {
        "ticker":   ticker,
        "score":    score,
        "is_watch": is_watch,
        "reason":   reason,
        "signals":  signals,
    }
