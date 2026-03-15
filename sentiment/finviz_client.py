"""
Finviz news scraper.
Pulls the latest headlines from finviz.com/quote.ashx?t=TICKER,
scores each with VADER, and returns an aggregate sentiment score.

Uses a browser-like User-Agent to avoid 403 blocks.
Finviz does not require an API key for basic quote/news pages.
"""

import logging
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)
_vader = SentimentIntensityAnalyzer()

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://finviz.com/",
}
_BASE_URL = "https://finviz.com/quote.ashx?t={ticker}&p=d"
_TIMEOUT = 15  # seconds


def get_news_sentiment(ticker: str, max_headlines: int = 10) -> dict:
    """
    Scrape Finviz quote page for `ticker`, extract the news table,
    run VADER on each headline, and return aggregate score + list.

    Returns:
        {
            "ticker":     str,
            "score":      float | None,
            "headlines":  list[dict],
            "source":     "finviz",
        }
    """
    ticker = ticker.upper()
    try:
        url = _BASE_URL.format(ticker=ticker)
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)

        if resp.status_code == 403:
            logger.warning("Finviz returned 403 for %s — rate limited or blocked", ticker)
            return _empty(ticker, error="403 Forbidden")
        if resp.status_code == 404:
            logger.warning("Finviz: ticker %s not found (404)", ticker)
            return _empty(ticker, error="404 Not Found")
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "lxml")
        table = soup.find("table", id="news-table")

        if table is None:
            logger.warning("Finviz: news-table not found for %s", ticker)
            return _empty(ticker, error="news-table not in HTML")

        headlines = _parse_news_table(table, max_headlines)
        scores = [h["score"] for h in headlines if h["score"] is not None]
        overall = round(sum(scores) / len(scores), 4) if scores else None

        return {
            "ticker":    ticker,
            "score":     overall,
            "headlines": headlines,
            "source":    "finviz",
        }

    except requests.exceptions.Timeout:
        logger.error("Finviz request timed out for %s", ticker)
        return _empty(ticker, error="timeout")
    except Exception as e:
        logger.error("Finviz scrape failed for %s: %s", ticker, e)
        return _empty(ticker, error=str(e))


def _parse_news_table(table, max_rows: int) -> list[dict]:
    """
    Parse the Finviz #news-table into a list of headline dicts.

    Finviz date column format:
      - First row of a new day:  "Feb-28-26 09:30AM"
      - Subsequent same-day rows: "09:15AM"
    """
    rows = table.find_all("tr")
    results = []
    last_date_str = ""

    for row in rows:
        if len(results) >= max_rows:
            break

        cells = row.find_all("td")
        if len(cells) < 2:
            continue

        # --- Date / time cell ---
        raw_dt = cells[0].get_text(strip=True)
        if " " in raw_dt:
            # Has full date, e.g. "Feb-28-26 09:30AM"
            last_date_str = raw_dt.split()[0]
            time_str = raw_dt.split()[1]
        else:
            # Time only, e.g. "09:30AM" — reuse last seen date
            time_str = raw_dt

        timestamp = f"{last_date_str} {time_str}".strip()

        # --- Headline cell ---
        link = cells[1].find("a")
        if link is None:
            continue
        title = link.get_text(strip=True)
        url   = link.get("href", "")

        compound = _vader.polarity_scores(title)["compound"]

        results.append({
            "title":     title,
            "score":     round(compound, 4),
            "timestamp": timestamp,
            "url":       url,
        })

    return results


def _empty(ticker: str, error: str = "") -> dict:
    result = {
        "ticker":    ticker,
        "score":     None,
        "headlines": [],
        "source":    "finviz",
    }
    if error:
        result["error"] = error
    return result
