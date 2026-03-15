"""
Stage 2 smoke test — prints live market data for AAPL, SPY, and NVDA.
Run from the project root:
    venv/Scripts/python tests/test_market_data.py
"""

import sys
import os
import logging

# Ensure project root is on the path regardless of where we run from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.market_data import get_ticker_data

logging.basicConfig(level=logging.WARNING)  # suppress info noise during tests

DIVIDER = "=" * 62


def _fmt_price(val) -> str:
    return f"${val:,.2f}" if val is not None else "N/A"


def _fmt_pct(val) -> str:
    if val is None:
        return "N/A"
    arrow = "+" if val >= 0 else "-"
    return f"{arrow}{abs(val):.2f}%"


def _fmt_cap(val) -> str:
    if val is None:
        return "N/A"
    if val >= 1e12:
        return f"${val/1e12:.2f}T"
    if val >= 1e9:
        return f"${val/1e9:.2f}B"
    if val >= 1e6:
        return f"${val/1e6:.2f}M"
    return f"${val:,.0f}"


def print_ticker_report(ticker: str):
    print(f"\n{DIVIDER}")
    print(f"  {ticker}")
    print(DIVIDER)

    data = get_ticker_data(ticker)
    fund = data.get("fundamentals", {})
    src  = data.get("sources", {})

    # --- Price block ---
    print(f"\n  PRICE")
    print(f"    Live price   : {_fmt_price(data.get('live_price'))}")
    print(f"    Change       : {_fmt_pct(data.get('change_pct'))}")
    print(f"    Prev close   : {_fmt_price(data.get('price_yf', {}).get('prev_close'))}")

    # --- Fundamentals block ---
    print(f"\n  FUNDAMENTALS")
    print(f"    Name         : {fund.get('short_name', 'N/A')}")
    print(f"    Sector       : {fund.get('sector', 'N/A')}")
    print(f"    Market cap   : {_fmt_cap(fund.get('market_cap'))}")
    print(f"    P/E (trail.) : {fund.get('pe_ratio', 'N/A')}")
    print(f"    P/E (fwd)    : {fund.get('forward_pe', 'N/A')}")
    print(f"    52w High     : {_fmt_price(fund.get('52w_high'))}")
    print(f"    52w Low      : {_fmt_price(fund.get('52w_low'))}")
    print(f"    Beta         : {fund.get('beta', 'N/A')}")
    print(f"    Div. yield   : {fund.get('dividend_yield', 'N/A')}")

    # --- Intraday bars block ---
    bars = data.get("intraday_bars", [])
    print(f"\n  INTRADAY BARS  ({len(bars)} total — showing last 3)")
    if bars:
        for bar in bars[-3:]:
            ts = bar.get("timestamp", "?")
            # Polygon timestamps are ms epoch — convert for readability
            if isinstance(ts, int):
                from datetime import datetime, timezone
                ts = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%H:%M UTC")
            print(f"    {ts}  O:{bar.get('open')}  H:{bar.get('high')}  "
                  f"L:{bar.get('low')}  C:{bar.get('close')}  V:{bar.get('volume')}")
    else:
        print("    No intraday bars returned")

    # --- Options block ---
    opts = data.get("options_chain", [])
    print(f"\n  OPTIONS CHAIN  ({len(opts)} contracts — showing first 3)")
    if opts:
        for opt in opts[:3]:
            print(f"    {opt.get('contract_type','?').upper():4s}  "
                  f"Strike:{opt.get('strike_price')}  "
                  f"Exp:{opt.get('expiration_date')}  "
                  f"IV:{opt.get('implied_vol')}  "
                  f"Delta:{opt.get('delta')}")
    else:
        print("    No options data (free Polygon tier or market closed)")

    # --- Earnings block ---
    earnings = data.get("earnings", [])
    print(f"\n  EARNINGS  (showing up to 6 dates)")
    if earnings:
        for e in earnings:
            est = e.get("eps_estimate", "N/A")
            rep = e.get("reported_eps", "N/A")
            surp = e.get("surprise_pct", "N/A")
            print(f"    {e['date']}  Est:{est}  Actual:{rep}  Surprise:{surp}%")
    else:
        print("    No earnings data returned")

    # --- Source health ---
    print(f"\n  DATA SOURCES")
    print(f"    Polygon quote OK : {src.get('polygon_quote_ok')}")
    print(f"    yfinance OK      : {src.get('yfinance_ok')}")
    print(f"    Intraday bars    : {src.get('intraday_bars_count')} bars")
    print(f"    Options contracts: {src.get('options_count')}")


if __name__ == "__main__":
    tickers = ["AAPL", "SPY", "NVDA"]
    print(f"\nTrading Terminal — Stage 2 Data Check")
    print(f"Tickers: {', '.join(tickers)}\n")

    for t in tickers:
        print_ticker_report(t)

    print(f"\n{DIVIDER}")
    print("  All checks complete.")
    print(DIVIDER)
