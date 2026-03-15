"""
Stage 3 smoke test — prints sentiment data for AAPL, NVDA, and TSLA.
Run from the project root:
    venv/Scripts/python tests/test_sentiment.py
"""

import sys
import os
import logging
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentiment.sentiment_aggregator import get_sentiment

logging.basicConfig(level=logging.WARNING)

DIVIDER  = "=" * 62
DIVIDER2 = "-" * 62


def _score_bar(score: float | None, width: int = 20) -> str:
    """ASCII bar: negative left, positive right, center = 0."""
    if score is None:
        return "[     N/A     ]"
    filled = int(abs(score) * (width // 2))
    filled = min(filled, width // 2)
    half   = width // 2
    if score >= 0:
        bar = " " * half + "#" * filled + " " * (half - filled)
    else:
        bar = " " * (half - filled) + "#" * filled + " " * half
    return f"[{bar}]"


def _fmt_score(score: float | None) -> str:
    if score is None:
        return "N/A"
    sign = "+" if score >= 0 else ""
    return f"{sign}{score:.4f}"


def print_sentiment_report(ticker: str):
    print(f"\n{DIVIDER}")
    print(f"  {ticker}")
    print(DIVIDER)

    data = get_sentiment(ticker)

    # --- Overall ---
    overall = data.get("overall_sentiment")
    label   = data.get("sentiment_label", "Unknown")
    bar     = _score_bar(overall)
    print(f"\n  OVERALL SENTIMENT")
    print(f"    Score  : {_fmt_score(overall)}  {bar}")
    print(f"    Signal : {label}")

    # --- Source health ---
    src = data.get("sources_ok", {})
    print(f"\n  SOURCES")
    print(f"    Yahoo News    : {'OK' if src.get('yahoo')         else 'FAILED/NO DATA'}"
          f"  (score: {_fmt_score(data.get('yahoo_score'))})")
    print(f"    Finviz        : {'OK' if src.get('finviz')        else 'FAILED/NO DATA'}"
          f"  (score: {_fmt_score(data.get('finviz_score'))})")
    print(f"    Google Trends : {'OK' if src.get('google_trends') else 'FAILED/NO DATA'}"
          f"  (direction: {data.get('google_trend_direction', 'N/A')}"
          f", value: {data.get('google_trend_value', 'N/A')}/100"
          f", proxy score: {_fmt_score(data.get('google_trend_score'))})")

    # --- Yahoo headlines ---
    yahoo_hl = data.get("yahoo_headlines", [])
    print(f"\n  YAHOO NEWS HEADLINES  ({len(yahoo_hl)} returned)")
    if yahoo_hl:
        for h in yahoo_hl[:5]:
            sign = "+" if h["score"] >= 0 else ""
            print(f"    [{sign}{h['score']:+.3f}] {h['title'][:72]}")
        if len(yahoo_hl) > 5:
            print(f"    ... and {len(yahoo_hl) - 5} more")
    else:
        print("    No headlines returned")

    # --- Finviz headlines ---
    fv_hl = data.get("finviz_headlines", [])
    print(f"\n  FINVIZ HEADLINES  ({len(fv_hl)} returned)")
    if fv_hl:
        for h in fv_hl[:5]:
            score = h.get("score")
            sign  = "+" if score is not None and score >= 0 else ""
            score_str = f"{sign}{score:+.3f}" if score is not None else "N/A"
            ts = h.get("timestamp", "")
            print(f"    [{score_str}] {h['title'][:65]}  ({ts})")
        if len(fv_hl) > 5:
            print(f"    ... and {len(fv_hl) - 5} more")
    else:
        print("    No headlines returned")


if __name__ == "__main__":
    tickers = ["AAPL", "NVDA", "TSLA"]
    print(f"\nTrading Terminal — Stage 3 Sentiment Check")
    print(f"Tickers: {', '.join(tickers)}")
    print("(Google Trends adds ~5s delay per ticker to avoid rate limiting)\n")

    for i, t in enumerate(tickers):
        print_sentiment_report(t)
        if i < len(tickers) - 1:
            time.sleep(5)   # polite pause between Google Trends calls

    print(f"\n{DIVIDER}")
    print("  All sentiment checks complete.")
    print(DIVIDER)
