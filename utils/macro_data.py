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
from concurrent.futures import ThreadPoolExecutor
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
    ("CPIAUCSL", "CPI",         "%"),
    ("UNRATE",   "Unemployment", "%"),
]


def get_macro_data() -> list[dict]:
    """
    Fetch all 5 macro indicators. Each source failure returns a degraded
    entry (value=None, display_value='N/A') — never raises.
    """
    api_key = os.getenv("FRED_API_KEY", "")

    with ThreadPoolExecutor(max_workers=5) as pool:
        fred_futures = [
            pool.submit(_fetch_fred, sid, name, unit, api_key)
            for sid, name, unit in _FRED_SERIES
        ]
        sp_future = pool.submit(_fetch_sp500)

    results = [f.result() for f in fred_futures]
    results.append(sp_future.result())
    return results


def _fetch_fred(series_id: str, name: str, unit: str, api_key: str) -> dict:
    base = {"name": name, "display_value": "N/A", "value": None,
            "change": None, "change_display": None, "unit": unit}

    if not api_key or api_key.startswith("your_"):
        base["display_value"] = "No key"
        return base

    try:
        # CPI needs 13 months of data to compute YoY% change
        obs_limit = 13 if series_id == "CPIAUCSL" else 2
        cutoff_days = 450 if series_id == "CPIAUCSL" else 90
        cutoff = (datetime.now() - timedelta(days=cutoff_days)).strftime("%Y-%m-%d")
        resp = requests.get(_FRED_BASE, params={
            "series_id":         series_id,
            "api_key":           api_key,
            "file_type":         "json",
            "sort_order":        "desc",
            "limit":             obs_limit,
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

        # CPI: convert raw index to YoY% inflation rate
        if series_id == "CPIAUCSL" and len(obs) >= 13:
            year_ago = float(obs[12]["value"])
            yoy_pct = round(((latest - year_ago) / year_ago) * 100, 2)
            # month-over-month delta for the change display
            if len(obs) >= 2:
                prev_index = float(obs[1]["value"])
                prev_year_ago = float(obs[12]["value"])  # approximate
                if len(obs) >= 13:
                    # compute previous month's YoY for a proper delta
                    prev_yoy = round(((prev_index - year_ago) / year_ago) * 100, 2)
                    change = round(yoy_pct - prev_yoy, 4)
                    sign = "+" if change >= 0 else ""
                    change_display = sign + str(round(change, 3)) + unit
            latest = yoy_pct
        elif len(obs) >= 2:
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
