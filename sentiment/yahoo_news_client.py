"""
Yahoo News sentiment client.
Pulls the latest news headlines for a ticker via yfinance and scores
each one with VADER. Returns an aggregate sentiment score and headline list.
"""

import logging
from datetime import datetime

import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)
_vader = SentimentIntensityAnalyzer()


def get_news_sentiment(ticker: str, max_headlines: int = 10) -> dict:
    """
    Fetch up to `max_headlines` Yahoo Finance news headlines for `ticker`,
    score each with VADER, and return an aggregate score + headline list.

    Returns:
        {
            "ticker":     str,
            "score":      float | None,   # -1 (bearish) to +1 (bullish)
            "headlines":  list[dict],
            "source":     "yahoo_news",
        }
    """
    ticker = ticker.upper()
    try:
        news = yf.Ticker(ticker).news or []

        headlines = []
        scores = []

        for item in news[:max_headlines]:
            # yfinance news item structure varies by version —
            # try both flat and nested ('content') formats
            content = item.get("content", item)
            title = (
                content.get("title")
                or item.get("title")
                or ""
            ).strip()

            if not title:
                continue

            pub_time = (
                content.get("pubDate")
                or item.get("providerPublishTime")
            )
            if isinstance(pub_time, int):
                pub_time = datetime.utcfromtimestamp(pub_time).strftime("%Y-%m-%d %H:%M")

            compound = _vader.polarity_scores(title)["compound"]
            scores.append(compound)

            headlines.append({
                "title":     title,
                "score":     round(compound, 4),
                "publisher": content.get("provider", {}).get("displayName")
                             or item.get("publisher", ""),
                "published": pub_time,
                "url":       content.get("canonicalUrl", {}).get("url")
                             or item.get("link", ""),
            })

        overall = round(sum(scores) / len(scores), 4) if scores else None

        return {
            "ticker":    ticker,
            "score":     overall,
            "headlines": headlines,
            "source":    "yahoo_news",
        }

    except Exception as e:
        logger.error("Yahoo News sentiment failed for %s: %s", ticker, e)
        return {"ticker": ticker, "score": None, "headlines": [], "source": "yahoo_news",
                "error": str(e)}
