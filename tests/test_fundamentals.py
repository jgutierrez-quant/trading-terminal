"""
Stage 10 smoke tests — fundamental analysis layer.
Run from project root:
    venv/Scripts/python tests/test_fundamentals.py

Tests:
    1. get_fundamentals (AAPL) — all expected keys present, no crash
    2. score_fundamentals (AAPL) — score in [-100, 100], valid signal label
    3. get_short_squeeze_score — GME (high short): keys present, score >= 0
                                  AAPL (low short): is_squeeze_candidate == False
    4. Integration — compute_anomaly("AAPL", check_fundamentals=True) has fundamental_score
"""

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.WARNING)

from data.fundamentals import get_fundamentals, score_fundamentals, get_short_squeeze_score

DIVIDER  = "=" * 64
DIVIDER2 = "-" * 64

EXPECTED_KEYS = [
    "ticker", "pe_ratio", "forward_pe", "peg_ratio", "ps_ratio",
    "pb_ratio", "ev_ebitda", "revenue_growth", "earnings_growth",
    "debt_to_equity", "current_ratio", "free_cashflow", "profit_margins",
    "roe", "inst_ownership_pct", "short_ratio", "short_pct_float",
    "recommendation", "target_price", "earnings_beat_rate",
    "recent_upgrade", "recent_downgrade",
    "insider_buy_recent", "insider_sell_recent",
    "market_cap", "sector", "industry", "error",
]


# ── Test 1: get_fundamentals ──────────────────────────────────────────────────

def test_get_fundamentals():
    print(f"\n{DIVIDER}")
    print("  TEST 1: get_fundamentals (AAPL)")
    print(DIVIDER)

    fund = get_fundamentals("AAPL")

    print(f"\n  Ticker:    {fund.get('ticker')}")
    print(f"  Sector:    {fund.get('sector')}")
    print(f"  Industry:  {fund.get('industry')}")
    print(f"  Error:     {fund.get('error')}")
    print(f"\n  Key values:")
    for k in ["pe_ratio", "forward_pe", "peg_ratio", "pb_ratio", "ev_ebitda",
              "revenue_growth", "earnings_growth", "debt_to_equity",
              "current_ratio", "roe", "inst_ownership_pct",
              "short_ratio", "short_pct_float",
              "recommendation", "target_price", "earnings_beat_rate",
              "market_cap"]:
        print(f"    {k:<24} {fund.get(k)}")

    print(f"\n  Boolean flags:")
    for k in ["recent_upgrade", "recent_downgrade", "insider_buy_recent", "insider_sell_recent"]:
        print(f"    {k:<24} {fund.get(k)}")

    assert isinstance(fund, dict), "get_fundamentals must return a dict"
    assert fund.get("ticker") == "AAPL", "ticker must be AAPL"
    for key in EXPECTED_KEYS:
        assert key in fund, f"Missing key: {key}"

    print("\n  [PASS] get_fundamentals() OK")


# ── Test 2: score_fundamentals ────────────────────────────────────────────────

def test_score_fundamentals():
    print(f"\n{DIVIDER}")
    print("  TEST 2: score_fundamentals (AAPL)")
    print(DIVIDER)

    fund   = get_fundamentals("AAPL")
    result = score_fundamentals(fund)

    fs  = result["fundamental_score"]
    sig = result["fundamental_signal"]
    rsn = result["fundamental_reasons"]

    print(f"\n  fundamental_score:   {fs}")
    print(f"  fundamental_signal:  {sig}")
    print(f"  fundamental_reasons: {rsn}")

    assert isinstance(result, dict), "score_fundamentals must return a dict"
    assert "fundamental_score"   in result
    assert "fundamental_signal"  in result
    assert "fundamental_reasons" in result
    assert -100 <= fs <= 100, f"Score {fs} out of [-100, 100] range"
    assert sig in ("Bullish", "Bearish", "Neutral"), f"Invalid signal: {sig}"
    assert isinstance(rsn, list), "fundamental_reasons must be a list"
    assert len(rsn) <= 5, "fundamental_reasons must have at most 5 items"

    print("\n  [PASS] score_fundamentals() OK")


# ── Test 3: get_short_squeeze_score ──────────────────────────────────────────

def test_get_short_squeeze_score():
    print(f"\n{DIVIDER}")
    print("  TEST 3: get_short_squeeze_score (GME + AAPL)")
    print(DIVIDER)

    # GME — historically high short interest
    gme_fund = get_fundamentals("GME")
    gme_sq   = get_short_squeeze_score(gme_fund)
    print(f"\n  GME:")
    print(f"    short_pct_float:    {gme_fund.get('short_pct_float')}")
    print(f"    short_ratio:        {gme_fund.get('short_ratio')}")
    print(f"    squeeze_score:      {gme_sq['squeeze_score']}")
    print(f"    is_squeeze_candidate: {gme_sq['is_squeeze_candidate']}")

    assert "squeeze_score"       in gme_sq
    assert "is_squeeze_candidate" in gme_sq
    assert gme_sq["squeeze_score"] >= 0, "squeeze_score must be >= 0"
    assert isinstance(gme_sq["is_squeeze_candidate"], bool)

    # AAPL — large cap, low short interest
    aapl_fund = get_fundamentals("AAPL")
    aapl_sq   = get_short_squeeze_score(aapl_fund)
    print(f"\n  AAPL:")
    print(f"    short_pct_float:    {aapl_fund.get('short_pct_float')}")
    print(f"    short_ratio:        {aapl_fund.get('short_ratio')}")
    print(f"    squeeze_score:      {aapl_sq['squeeze_score']}")
    print(f"    is_squeeze_candidate: {aapl_sq['is_squeeze_candidate']}")

    assert aapl_sq["is_squeeze_candidate"] == False, \
        "AAPL should NOT be a squeeze candidate"

    print("\n  [PASS] get_short_squeeze_score() OK")


# ── Test 4: Integration — compute_anomaly with check_fundamentals ─────────────

def test_integration():
    print(f"\n{DIVIDER}")
    print("  TEST 4: Integration — compute_anomaly(AAPL, check_fundamentals=True)")
    print(DIVIDER)

    from data.technicals        import get_technicals
    from data.anomaly_detector  import compute_anomaly

    tech = get_technicals("AAPL")
    sent = {"sentiment_label": "Neutral", "overall_sentiment": 0.0}

    result = compute_anomaly("AAPL", tech, sent, check_fundamentals=True)

    print(f"\n  ticker:              {result.get('ticker')}")
    print(f"  score:               {result.get('score')}")
    print(f"  is_watch:            {result.get('is_watch')}")
    print(f"  quality_score:       {result.get('quality_score')}")
    print(f"  direction:           {result.get('direction')}")
    print(f"  fundamental_score:   {result.get('fundamental_score')}")
    print(f"  fundamental_signal:  {result.get('fundamental_signal')}")
    print(f"  fundamental_reasons: {result.get('fundamental_reasons')}")

    assert result.get("ticker") == "AAPL"
    assert "fundamental_score"   in result, "fundamental_score missing from return"
    assert "fundamental_signal"  in result, "fundamental_signal missing from return"
    assert "fundamental_reasons" in result, "fundamental_reasons missing from return"
    fs = result.get("fundamental_score")
    assert fs is not None, "fundamental_score should not be None when check_fundamentals=True"
    assert -100 <= fs <= 100, f"fundamental_score {fs} out of range"

    # Without check_fundamentals — fundamental fields should be None
    result_no_fund = compute_anomaly("AAPL", tech, sent, check_fundamentals=False)
    assert result_no_fund.get("fundamental_score") is None, \
        "fundamental_score should be None when check_fundamentals=False"

    print("\n  [PASS] integration test OK")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nTrading Terminal — Stage 10 Fundamental Analysis Layer")
    print(f"Tests: get_fundamentals, score_fundamentals, squeeze_score, integration\n")

    test_get_fundamentals()
    test_score_fundamentals()
    test_get_short_squeeze_score()
    test_integration()

    print(f"\n{DIVIDER}")
    print("  All Stage 10 checks complete.")
    print(DIVIDER)
