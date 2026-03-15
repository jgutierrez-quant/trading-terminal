"""
Macro indicator fetcher.

4 indicators from FRED (requires FRED_API_KEY in .env):
    FEDFUNDS  — Federal Funds Rate
    GS10      — 10-Year Treasury Constant Maturity
    CPIAUCSL  — Consumer Price Index
    UNRATE    — Unemployment Rate

1 indicator always from yfinance:
    ^GSPC     — S&P 500 (30-day % change)

Each indicator is returned as a dict:
    {name, display_value, value, change, change_display, unit}

change_display is a pre-formatted string ("+0.25" or "-0.03") ready to
drop straight into st.metric(delta=...) — no f-string formatting needed
in the calling code.
"""

import os
import logging
from datetime import datetime, timedelta

import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
_FRED_SERIES = [
    ("FEDFUNDS", "Fed Funds",    "%"),
    ("GS10",     "10Y Yield",   "%"),
    ("CPIAUCSL", "CPI",         ""),
    ("UNRATE",   "Unemployment", "%"),
]


def get_macro_data() -> list[dict]:
    """
    Fetch all 5 macro indicators. Each source failure returns a degraded
    entry (value=None, display_value='N/A') — never raises.
    """
    results = []
    api_key = os.getenv("FRED_API_KEY", "")

    for series_id, name, unit in _FRED_SERIES:
        results.append(_fetch_fred(series_id, name, unit, api_key))

    results.append(_fetch_sp500())
    return results


def _fetch_fred(series_id: str, name: str, unit: str, api_key: str) -> dict:
    base = {"name": name, "display_value": "N/A", "value": None,
            "change": None, "change_display": None, "unit": unit}

    if not api_key or api_key.startswith("your_"):
        base["display_value"] = "No key"
        return base

    try:
        cutoff = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        resp = requests.get(_FRED_BASE, params={
            "series_id":         series_id,
            "api_key":           api_key,
            "file_type":         "json",
            "sort_order":        "desc",
            "limit":             2,
            "observation_start": cutoff,
        }, timeout=10)
        resp.raise_for_status()
        obs = [o for o in resp.json().get("observations", [])
               if o.get("value", ".") not in (".", "", None)]

        if not obs:
            return base

        latest = float(obs[0]["value"])
        change = None
        change_display = None

        if len(obs) >= 2:
            prev = float(obs[1]["value"])
            change = round(latest - prev, 4)
            sign = "+" if change >= 0 else ""
            change_display = sign + str(round(change, 3)) + unit

        # Pre-compute display_value — unit appended here, not in f-string
        if unit == "%":
            display_value = str(round(latest, 2)) + "%"
        else:
            display_value = str(round(latest, 2))

        return {
            "name":           name,
            "display_value":  display_value,
            "value":          latest,
            "change":         change,
            "change_display": change_display,
            "unit":           unit,
        }

    except Exception as e:
        logger.error("FRED fetch failed for %s: %s", series_id, e)
        return base


def _fetch_sp500() -> dict:
    base = {"name": "S&P 500", "display_value": "N/A", "value": None,
            "change": None, "change_display": None, "unit": "pts"}
    try:
        hist = yf.Ticker("^GSPC").history(period="35d")
        if hist.empty or len(hist) < 2:
            return base

        latest = float(hist["Close"].iloc[-1])
        # compare to earliest bar in the window (~30 days ago)
        prev_30 = float(hist["Close"].iloc[0])
        change_pct = round(((latest - prev_30) / prev_30) * 100, 2) if prev_30 else None

        display_value = str(round(latest, 0)).replace(".0", "")

        change_display = None
        if change_pct is not None:
            sign = "+" if change_pct >= 0 else ""
            change_display = sign + str(change_pct) + "% (30d)"

        return {
            "name":           "S&P 500",
            "display_value":  display_value,
            "value":          latest,
            "change":         change_pct,
            "change_display": change_display,
            "unit":           "pts",
        }

    except Exception as e:
        logger.error("S&P 500 fetch failed: %s", e)
        return base
