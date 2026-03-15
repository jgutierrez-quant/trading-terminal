"""
Stage 6 smoke test — technical indicators and anomaly flags for AAPL, NVDA, TSLA.
Run from project root:
    venv/Scripts/python tests/test_technicals.py
"""

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.WARNING)

from data.technicals       import get_technicals
from data.anomaly_detector import compute_anomaly

DIVIDER  = "=" * 64
DIVIDER2 = "-" * 64


def _v(val, fmt=None, prefix="", suffix="") -> str:
    """Format a value for display, returning 'N/A' for None."""
    if val is None:
        return "N/A"
    if fmt:
        return prefix + format(val, fmt) + suffix
    return prefix + str(val) + suffix


def _signal_line(label: str, signal: str) -> str:
    tag = {"Overbought": "[!]", "Oversold": "[!]",
           "Bullish Cross": "[+]", "Bearish Cross": "[-]",
           "Bullish": "[+]", "Bearish": "[-]",
           "High Volume": "[!]", "Elevated": "[~]",
           "Above Upper": "[!]", "Below Lower": "[!]",
           "Above VWAP": "[+]", "Below VWAP": "[-]"}.get(signal, "[ ]")
    return f"    {tag} {label}: {signal}"


def print_tech_report(ticker: str):
    print(f"\n{DIVIDER}")
    print(f"  {ticker}")
    print(DIVIDER)

    tech = get_technicals(ticker)

    if tech.get("error"):
        print(f"  ERROR: {tech['error']}")
        return

    cp = tech.get("current_price")
    print(f"\n  Current Price : {_v(cp, '.2f', '$')}")
    print(f"  Timestamp     : {tech.get('timestamp', 'N/A')}")

    print(f"\n  RSI (14)")
    print(f"    Value  : {_v(tech.get('rsi'), '.2f')}")
    print(_signal_line("Signal", tech.get("rsi_signal", "N/A")))

    print(f"\n  MACD (12/26/9)")
    print(f"    MACD Line    : {_v(tech.get('macd_line'), '.4f')}")
    print(f"    Signal Line  : {_v(tech.get('macd_signal_line'), '.4f')}")
    print(f"    Histogram    : {_v(tech.get('macd_hist'), '.4f')}")
    print(_signal_line("Signal", tech.get("macd_signal", "N/A")))

    print(f"\n  Bollinger Bands (20, 2σ)")
    print(f"    Upper  : {_v(tech.get('bb_upper'), '.2f', '$')}")
    print(f"    Middle : {_v(tech.get('bb_middle'), '.2f', '$')}")
    print(f"    Lower  : {_v(tech.get('bb_lower'), '.2f', '$')}")
    print(_signal_line("Signal", tech.get("bb_signal", "N/A")))

    print(f"\n  VWAP (intraday)")
    print(f"    Value  : {_v(tech.get('vwap'), '.2f', '$')}")
    vwap_sig = tech.get("vwap_signal") or "N/A"
    print(_signal_line("Signal", vwap_sig))

    print(f"\n  Volume")
    tv = tech.get("today_volume")
    av = tech.get("avg_volume_20d")
    vr = tech.get("volume_ratio")
    print(f"    Today (so far) : {_v(tv, ',d') if tv else 'N/A'}")
    print(f"    20d Avg        : {_v(av, ',.0f') if av else 'N/A'}")
    print(f"    Ratio          : {_v(vr, '.2f')}x")
    print(_signal_line("Signal", tech.get("volume_signal", "N/A")))

    print(f"\n  Moving Averages")
    print(f"    SMA 20 : {_v(tech.get('sma20'), '.2f', '$')}  "
          f"— Price is {tech.get('price_vs_sma20', 'N/A')}")
    print(f"    SMA 50 : {_v(tech.get('sma50'), '.2f', '$')}  "
          f"— Price is {tech.get('price_vs_sma50', 'N/A')}")

    sr = tech.get("support_resistance", [])
    print(f"\n  Support / Resistance Levels")
    if sr:
        for i, lv in enumerate(sr, 1):
            dist = round((cp - lv) / cp * 100, 1) if cp else None
            dist_str = f"  ({dist:+.1f}% from price)" if dist is not None else ""
            print(f"    Level {i}: ${lv:.2f}{dist_str}")
    else:
        print("    No levels found")

    nb = len(tech.get("daily_bars", []))
    print(f"\n  Daily Bars    : {nb} bars returned")

    # Anomaly (using neutral mock sentiment to isolate technical signals)
    mock_sent = {"sentiment_label": "Neutral", "overall_sentiment": 0.0}
    anomaly = compute_anomaly(ticker, tech, mock_sent)
    print(f"\n  Anomaly Score : {anomaly['score']} signals")
    watch_tag = "  [WATCH]" if anomaly["is_watch"] else ""
    print(f"  Watch Flag    : {'YES' if anomaly['is_watch'] else 'No'}{watch_tag}")
    print(f"  Reason        : {anomaly['reason']}")
    if anomaly["signals"]:
        for sig in anomaly["signals"]:
            print(f"                  · {sig}")


if __name__ == "__main__":
    tickers = ["AAPL", "NVDA", "TSLA"]
    print(f"\nTrading Terminal — Stage 6 Technical Indicator Check")
    print(f"Tickers: {', '.join(tickers)}")

    for t in tickers:
        print_tech_report(t)

    print(f"\n{DIVIDER}")
    print("  All checks complete.")
    print(DIVIDER)
