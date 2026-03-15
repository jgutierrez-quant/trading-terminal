"""
Sentiment aggregator.
Calls Yahoo News, Finviz, and Google Trends independently and combines
their signals into a single weighted overall_sentiment score per ticker.

Weights:
    Yahoo News    35%
    Finviz News   35%
    Google Trends 30%  (trend direction converted to a -0.5 / 0 / +0.5 proxy)

If a source fails its weight is redistributed proportionally among
the sources that did return data, so the score is always meaningful.
"""

import logging

from sentiment.yahoo_news_client   import get_news_sentiment   as yahoo_sentiment
from sentiment.finviz_client        import get_news_sentiment   as finviz_sentiment
from sentiment.google_trends_client import get_trend

logger = logging.getLogger(__name__)

_WEIGHTS = {
    "yahoo":  0.35,
    "finviz": 0.35,
    "trends": 0.30,
}

# Map Google Trends direction to a sentiment-proxy numeric score
_DIRECTION_SCORE = {
    "rising":  0.5,
    "stable":  0.0,
    "falling": -0.5,
    None:       None,
}


def get_sentiment(ticker: str) -> dict:
    """
    Pull sentiment from all three sources for `ticker` and return a
    clean unified dictionary.

    Returns:
        {
            "ticker":               str,
            "overall_sentiment":    float | None,   # -1 to +1
            "sentiment_label":      str,             # Bullish / Neutral / Bearish
            "yahoo_score":          float | None,
            "yahoo_headlines":      list[dict],
            "finviz_score":         float | None,
            "finviz_headlines":     list[dict],
            "google_trend_direction": str | None,
            "google_trend_value":   int | None,
            "google_trend_score":   float | None,   # direction converted to numeric
            "sources_ok":           dict[str, bool],
        }
    """
    ticker = ticker.upper().strip()
    logger.info("Fetching sentiment for %s", ticker)

    # --- Call all sources independently ---
    yahoo  = _safe(yahoo_sentiment,  ticker, label="yahoo")
    finviz = _safe(finviz_sentiment, ticker, label="finviz")
    trends = _safe(get_trend,        ticker, label="google_trends")

    yahoo_score  = yahoo.get("score")
    finviz_score = finviz.get("score")
    trend_dir    = trends.get("direction")
    trend_score  = _DIRECTION_SCORE.get(trend_dir)

    overall = _weighted_average(yahoo_score, finviz_score, trend_score)

    return {
        "ticker":                 ticker,
        "overall_sentiment":      overall,
        "sentiment_label":        _label(overall),
        "yahoo_score":            yahoo_score,
        "yahoo_headlines":        yahoo.get("headlines", []),
        "finviz_score":           finviz_score,
        "finviz_headlines":       finviz.get("headlines", []),
        "google_trend_direction": trend_dir,
        "google_trend_value":     trends.get("last_value"),
        "google_trend_score":     trend_score,
        "sources_ok": {
            "yahoo":         yahoo_score  is not None,
            "finviz":        finviz_score is not None,
            "google_trends": trend_dir    is not None,
        },
    }


# --- helpers ---

def _weighted_average(yahoo: float | None,
                      finviz: float | None,
                      trends: float | None) -> float | None:
    """
    Compute weighted average, redistributing weight from failed sources.
    Returns None only if ALL sources failed.
    """
    available = {
        "yahoo":  (yahoo,  _WEIGHTS["yahoo"]),
        "finviz": (finviz, _WEIGHTS["finviz"]),
        "trends": (trends, _WEIGHTS["trends"]),
    }
    pairs = [(score, w) for score, w in available.values() if score is not None]
    if not pairs:
        return None

    total_weight = sum(w for _, w in pairs)
    weighted_sum = sum(score * w for score, w in pairs)
    return round(weighted_sum / total_weight, 4)


def _label(score: float | None) -> str:
    if score is None:
        return "Unknown"
    if score >= 0.05:
        return "Bullish"
    if score <= -0.05:
        return "Bearish"
    return "Neutral"


def _safe(fn, ticker: str, label: str) -> dict:
    try:
        return fn(ticker)
    except Exception as e:
        logger.error("Unexpected error in %s for %s: %s", label, ticker, e)
        return {"ticker": ticker, "score": None, "headlines": [],
                "direction": None, "error": str(e)}
