"""
Watchlist persistence — reads and writes watchlist.json at the project root.
All paths are resolved relative to this file's location so they work
regardless of the cwd when Streamlit runs.
"""

import json
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATH = os.path.join(_ROOT, "watchlist.json")
_DEFAULT = ["AAPL", "NVDA", "TSLA", "SPY"]


def load_watchlist() -> list[str]:
    try:
        with open(_PATH) as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(t).upper().strip() for t in data if t]
        return list(_DEFAULT)
    except (FileNotFoundError, json.JSONDecodeError):
        return list(_DEFAULT)


def save_watchlist(tickers: list[str]) -> None:
    with open(_PATH, "w") as f:
        json.dump([t.upper().strip() for t in tickers if t], f, indent=2)


def add_ticker(ticker: str) -> list[str]:
    ticker = ticker.upper().strip()
    wl = load_watchlist()
    if ticker and ticker not in wl:
        wl.append(ticker)
        save_watchlist(wl)
    return load_watchlist()


def remove_ticker(ticker: str) -> list[str]:
    ticker = ticker.upper().strip()
    wl = [t for t in load_watchlist() if t != ticker]
    save_watchlist(wl)
    return wl
