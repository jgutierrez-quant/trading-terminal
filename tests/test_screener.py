"""
Stage 7 smoke test — market screener and sector monitor.
Run from project root:
    venv/Scripts/python tests/test_screener.py
"""

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.WARNING)

from data.screener       import get_top_movers, get_unusual_volume, run_market_scan
from data.sector_monitor import get_sector_data, SECTOR_ETFS

DIVIDER  = "=" * 64
DIVIDER2 = "-" * 64


def _fmt_vol(v) -> str:
    if v is None:
        return "N/A"
    if v >= 1_000_000_000:
        return f"{v/1_000_000_000:.1f}B"
    if v >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    if v >= 1_000:
        return f"{v/1_000:.0f}K"
    return str(v)


def test_top_movers():
    print(f"\n{DIVIDER}")
    print("  TOP MOVERS")
    print(DIVIDER)

    result  = get_top_movers(n=15)
    gainers = result.get("gainers", [])
    losers  = result.get("losers",  [])
    error   = result.get("error")

    if error:
        print(f"  NOTE: {error}")

    print(f"\n  TOP GAINERS ({len(gainers)} returned)")
    print(f"  {'Ticker':<8} {'Price':>9} {'Chg%':>8} {'Volume':>10}  Company")
    print(f"  {DIVIDER2}")
    for r in gainers:
        chg     = r.get("change_pct")
        chg_str = f"{chg:+.2f}%" if chg is not None else "N/A"
        p_str   = f"${r['price']:.2f}" if r.get("price") else "N/A"
        print(f"  {r['ticker']:<8} {p_str:>9} {chg_str:>8} "
              f"{_fmt_vol(r.get('volume')):>10}  {r.get('company','')[:28]}")

    print(f"\n  TOP LOSERS ({len(losers)} returned)")
    print(f"  {'Ticker':<8} {'Price':>9} {'Chg%':>8} {'Volume':>10}  Company")
    print(f"  {DIVIDER2}")
    for r in losers:
        chg     = r.get("change_pct")
        chg_str = f"{chg:+.2f}%" if chg is not None else "N/A"
        p_str   = f"${r['price']:.2f}" if r.get("price") else "N/A"
        print(f"  {r['ticker']:<8} {p_str:>9} {chg_str:>8} "
              f"{_fmt_vol(r.get('volume')):>10}  {r.get('company','')[:28]}")

    assert isinstance(gainers, list), "gainers must be a list"
    assert isinstance(losers,  list), "losers must be a list"
    if gainers:
        g = gainers[0]
        assert "ticker"     in g, "missing ticker"
        assert "price"      in g, "missing price"
        assert "change_pct" in g, "missing change_pct"
        assert "volume"     in g, "missing volume"
    print("\n  [PASS] get_top_movers() OK")


def test_unusual_volume():
    print(f"\n{DIVIDER}")
    print("  UNUSUAL VOLUME")
    print(DIVIDER)

    result = get_unusual_volume(n=15)
    print(f"\n  Unusual volume tickers ({len(result)} returned)")
    print(f"  {'Ticker':<8} {'Price':>9} {'Chg%':>8} {'Volume':>10}  Company")
    print(f"  {DIVIDER2}")
    for r in result:
        chg     = r.get("change_pct")
        chg_str = f"{chg:+.2f}%" if chg is not None else "N/A"
        p_str   = f"${r['price']:.2f}" if r.get("price") else "N/A"
        print(f"  {r['ticker']:<8} {p_str:>9} {chg_str:>8} "
              f"{_fmt_vol(r.get('volume')):>10}  {r.get('company','')[:28]}")

    assert isinstance(result, list), "result must be a list"
    print("\n  [PASS] get_unusual_volume() OK")


def test_sector_data():
    print(f"\n{DIVIDER}")
    print("  SECTOR MONITOR")
    print(DIVIDER)

    result = get_sector_data()

    print(f"\n  All {len(result)} sector ETFs (sorted best → worst)")
    print(f"  {'ETF':<5} {'Sector':<22} {'Price':>9} {'Chg%':>8}  Chart")
    print(f"  {DIVIDER2}")
    for s in result:
        chg     = s.get("change_pct")
        chg_str = f"{chg:+.2f}%" if chg is not None else "N/A"
        p_str   = f"${s['price']:.2f}" if s.get("price") else "N/A"
        if chg and chg > 0:
            bar = ("+" * min(int(abs(chg) * 3), 20)).ljust(20)
        elif chg and chg < 0:
            bar = ("-" * min(int(abs(chg) * 3), 20)).ljust(20)
        else:
            bar = " " * 20
        print(f"  {s['ticker']:<5} {s['name']:<22} {p_str:>9} {chg_str:>8}  {bar}")

    assert len(result) == len(SECTOR_ETFS), (
        f"Expected {len(SECTOR_ETFS)} sectors, got {len(result)}"
    )
    if result:
        r = result[0]
        assert "ticker"     in r, "missing ticker"
        assert "name"       in r, "missing name"
        assert "change_pct" in r, "missing change_pct"
    print("\n  [PASS] get_sector_data() OK")


def test_market_scan():
    print(f"\n{DIVIDER}")
    print("  MARKET SCAN  (mini — top movers only, max 20 tickers)")
    print(DIVIDER)
    print("  Fetching movers and running anomaly scan...")

    movers  = get_top_movers(n=10)
    unusual = get_unusual_volume(n=5)
    results = run_market_scan(movers=movers, unusual=unusual, max_tickers=20)

    print(f"\n  Scan result: {len(results)} Watch candidate(s)")
    print(f"  {'Ticker':<7} {'Score':>5} {'Chg%':>8}  Reason")
    print(f"  {DIVIDER2}")
    for r in results:
        chg_str = f"{r.get('change_pct'):+.2f}%" if r.get("change_pct") is not None else "N/A"
        print(f"  {r['ticker']:<7} {r['score']:>5} {chg_str:>8}  {r['reason'][:55]}")
    if not results:
        print("  (No Watch-flagged tickers in this 20-ticker universe)")

    assert isinstance(results, list), "results must be a list"
    for r in results:
        assert "ticker"   in r, "missing ticker"
        assert "score"    in r, "missing score"
        assert "is_watch" in r, "missing is_watch"
        assert r["is_watch"] is True, "scan must only return Watch=True entries"
    print("\n  [PASS] run_market_scan() OK")


if __name__ == "__main__":
    print(f"\nTrading Terminal — Stage 7 Screener Check")
    print(f"Tests: top_movers, unusual_volume, sector_data, market_scan\n")

    test_top_movers()
    test_unusual_volume()
    test_sector_data()
    test_market_scan()

    print(f"\n{DIVIDER}")
    print("  All Stage 7 checks complete.")
    print(DIVIDER)
