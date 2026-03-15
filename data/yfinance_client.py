"""
yfinance client.
Provides current price + daily change %, key fundamentals, and earnings dates.
No API key required — yfinance scrapes Yahoo Finance.
"""

import logging
import yfinance as yf

logger = logging.getLogger(__name__)


def get_price_and_change(ticker: str) -> dict:
    """
    Current price and daily change % from yfinance.
    Uses 5-day history to ensure we always have a prev-close even on thin days.
    """
    ticker = ticker.upper()
    try:
        hist = yf.Ticker(ticker).history(period="5d")

        if hist.empty:
            return {"ticker": ticker, "error": "No price history returned"}

        current  = float(hist["Close"].iloc[-1])
        prev     = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else current
        chg_pct  = round(((current - prev) / prev) * 100, 2) if prev else None

        return {
            "ticker":        ticker,
            "current_price": round(current, 2),
            "prev_close":    round(prev, 2),
            "change_pct":    chg_pct,
        }

    except Exception as e:
        logger.error("yfinance get_price_and_change failed for %s: %s", ticker, e)
        return {"ticker": ticker, "error": str(e)}


def get_fundamentals(ticker: str) -> dict:
    """
    Key fundamentals from yfinance: P/E, market cap, 52-week range, sector, etc.
    """
    ticker = ticker.upper()
    try:
        info = yf.Ticker(ticker).info

        def safe(key):
            val = info.get(key)
            return val if val not in (None, "None", "N/A", "") else None

        return {
            "ticker":       ticker,
            "short_name":   safe("shortName"),
            "sector":       safe("sector"),
            "industry":     safe("industry"),
            "market_cap":   safe("marketCap"),
            "pe_ratio":     safe("trailingPE"),
            "forward_pe":   safe("forwardPE"),
            "52w_high":     safe("fiftyTwoWeekHigh"),
            "52w_low":      safe("fiftyTwoWeekLow"),
            "avg_volume":   safe("averageVolume"),
            "beta":         safe("beta"),
            "dividend_yield": safe("dividendYield"),
        }

    except Exception as e:
        logger.error("yfinance get_fundamentals failed for %s: %s", ticker, e)
        return {"ticker": ticker, "error": str(e)}


def get_earnings_dates(ticker: str) -> list[dict]:
    """
    Recent and upcoming earnings dates with EPS estimate vs. actual.
    Returns up to 6 entries (mix of past + future).
    """
    ticker = ticker.upper()
    result = []
    try:
        t = yf.Ticker(ticker)
        dates = t.earnings_dates

        if dates is not None and not dates.empty:
            for dt, row in dates.head(6).iterrows():
                result.append({
                    "date":         str(dt.date()),
                    "eps_estimate": _safe_float(row.get("EPS Estimate")),
                    "reported_eps": _safe_float(row.get("Reported EPS")),
                    "surprise_pct": _safe_float(row.get("Surprise(%)")),
                })

    except Exception as e:
        logger.warning("yfinance get_earnings_dates failed for %s: %s", ticker, e)

    return result


def _safe_float(val) -> float | None:
    try:
        f = float(val)
        import math
        return None if math.isnan(f) else round(f, 4)
    except (TypeError, ValueError):
        return None
