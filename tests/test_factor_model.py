"""
Stage 11 smoke tests — 6-factor quantitative model + PEAD tracker.
Run from project root:
    venv/Scripts/python tests/test_factor_model.py

Tests:
    1. compute_factor_model (AAPL) — all expected keys present, score in [0,100]
    2. All 6 individual factors (NVDA) — available/unavailable, percentile range
    3. compute_factor_model (TSLA) — composite signal is Long/Short/Neutral
    4. compute_factor_model (JNJ)  — print full breakdown, verify structure
    5. PEAD tracker (AAPL)         — get_pead_status keys present, types correct
    6. Integration — compare old heuristic score vs new factor model score for AAPL
"""

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.WARNING)

from data.factor_model  import compute_factor_model, FACTOR_WEIGHTS
from data.pead_tracker  import get_pead_status

DIVIDER  = "=" * 64
DIVIDER2 = "-" * 64

EXPECTED_FM_KEYS = [
    "ticker", "composite_score", "composite_signal", "factors",
    "weights_used", "data_completeness", "pead_candidate",
]
FACTOR_NAMES = [
    "momentum", "earnings_momentum", "value",
    "quality", "short_interest", "institutional",
]
EXPECTED_PEAD_KEYS = [
    "ticker", "earnings_date", "days_since", "trading_days_since",
    "surprise_pct", "sue", "drift_pct", "est_total_drift",
    "est_remaining_drift", "pct_drift_realized", "is_active",
    "signal_strength", "error",
]


# ── Test 1: compute_factor_model (AAPL) ──────────────────────────────────────

def test_compute_factor_model_aapl():
    print(f"\n{DIVIDER}")
    print("  TEST 1: compute_factor_model (AAPL) — structure check")
    print(DIVIDER)

    result = compute_factor_model("AAPL")

    print(f"\n  Ticker:            {result.get('ticker')}")
    print(f"  Composite Score:   {result.get('composite_score'):.1f} / 100")
    print(f"  Composite Signal:  {result.get('composite_signal')}")
    print(f"  Data Completeness: {result.get('data_completeness', 0)*100:.0f}%")
    print(f"  PEAD Candidate:    {result.get('pead_candidate')}")

    assert isinstance(result, dict), "compute_factor_model must return a dict"
    assert result.get("ticker") == "AAPL"
    for key in EXPECTED_FM_KEYS:
        assert key in result, f"Missing key: {key}"

    score = result["composite_score"]
    assert 0 <= score <= 100, f"composite_score {score} out of [0, 100]"
    assert result["composite_signal"] in ("Long", "Short", "Neutral"), \
        f"Invalid signal: {result['composite_signal']}"

    factors = result.get("factors", {})
    for name in FACTOR_NAMES:
        assert name in factors, f"Missing factor: {name}"
        f = factors[name]
        assert "available" in f, f"{name}: missing 'available' key"
        assert "percentile" in f, f"{name}: missing 'percentile' key"
        pct = f["percentile"]
        assert 0 <= pct <= 100, f"{name}: percentile {pct} out of range"

    print("\n  [PASS] compute_factor_model (AAPL) OK")


# ── Test 2: All 6 factors on NVDA ────────────────────────────────────────────

def test_all_factors_nvda():
    print(f"\n{DIVIDER}")
    print("  TEST 2: All 6 factors (NVDA) — individual factor check")
    print(DIVIDER)

    result = compute_factor_model("NVDA")
    factors = result.get("factors", {})

    print(f"\n  {'Factor':<22} {'Available':<12} {'Percentile':<12} {'Notes'}")
    print(f"  {DIVIDER2}")
    for name in FACTOR_NAMES:
        f = factors.get(name, {})
        avail = f.get("available", False)
        pct   = f.get("percentile", 50.0)
        notes = f.get("notes", "—")
        print(f"  {name:<22} {str(avail):<12} {pct:<12.1f} {str(notes)[:40]}")
        assert "available" in f, f"{name}: missing 'available' key"
        assert "percentile" in f, f"{name}: missing 'percentile' key"
        assert 0 <= f["percentile"] <= 100, f"{name}: out of range"

    weights_used = result.get("weights_used", {})
    total_w = sum(weights_used.values())
    if weights_used:
        assert abs(total_w - 1.0) < 0.01, f"Weights don't sum to 1.0: {total_w:.3f}"
        print(f"\n  Weights sum: {total_w:.3f} (OK)")

    print("\n  [PASS] All 6 factors (NVDA) OK")


# ── Test 3: TSLA — composite signal validation ────────────────────────────────

def test_composite_signal_tsla():
    print(f"\n{DIVIDER}")
    print("  TEST 3: compute_factor_model (TSLA) — composite signal")
    print(DIVIDER)

    result = compute_factor_model("TSLA")
    score  = result.get("composite_score", 50.0)
    signal = result.get("composite_signal", "")

    print(f"\n  TSLA Composite Score:  {score:.1f}")
    print(f"  TSLA Composite Signal: {signal}")

    if score > 70:
        assert signal == "Long",    f"Expected Long for score {score:.1f}"
    elif score < 30:
        assert signal == "Short",   f"Expected Short for score {score:.1f}"
    else:
        assert signal == "Neutral", f"Expected Neutral for score {score:.1f}"

    print("\n  [PASS] TSLA composite signal consistent with score OK")


# ── Test 4: JNJ — full breakdown print ───────────────────────────────────────

def test_full_breakdown_jnj():
    print(f"\n{DIVIDER}")
    print("  TEST 4: compute_factor_model (JNJ) — full breakdown")
    print(DIVIDER)

    result = compute_factor_model("JNJ")
    factors = result.get("factors", {})

    print(f"\n  JNJ Composite: {result.get('composite_score'):.1f} / 100"
          f"  [{result.get('composite_signal')}]")
    print(f"  Data completeness: {result.get('data_completeness', 0)*100:.0f}%\n")

    for name in FACTOR_NAMES:
        f    = factors.get(name, {})
        wt   = FACTOR_WEIGHTS.get(name, 0)
        pct  = f.get("percentile", 50.0)
        avail = f.get("available", False)
        notes = f.get("notes", "") or ""
        bar   = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
        print(f"  {name:<22} {pct:>5.1f}th  wt={wt*100:.0f}%  [{bar}]")
        print(f"    notes: {str(notes)[:60]}")

    assert result.get("ticker") == "JNJ"
    print("\n  [PASS] JNJ full breakdown OK")


# ── Test 5: PEAD tracker — get_pead_status ────────────────────────────────────

def test_pead_status():
    print(f"\n{DIVIDER}")
    print("  TEST 5: get_pead_status (AAPL) — PEAD tracker structure")
    print(DIVIDER)

    status = get_pead_status("AAPL")

    print(f"\n  Ticker:              {status.get('ticker')}")
    print(f"  Earnings Date:       {status.get('earnings_date')}")
    print(f"  Days Since:          {status.get('days_since')}")
    print(f"  Trading Days Since:  {status.get('trading_days_since')}")
    print(f"  Surprise %:          {status.get('surprise_pct')}")
    print(f"  SUE:                 {status.get('sue')}")
    print(f"  Drift %:             {status.get('drift_pct')}")
    print(f"  Est. Total Drift:    {status.get('est_total_drift')}")
    print(f"  Est. Remaining:      {status.get('est_remaining_drift')}")
    print(f"  % Drift Realized:    {status.get('pct_drift_realized')}")
    print(f"  Is Active:           {status.get('is_active')}")
    print(f"  Signal Strength:     {status.get('signal_strength')}")
    print(f"  Error:               {status.get('error')}")

    assert isinstance(status, dict), "get_pead_status must return a dict"
    assert status.get("ticker") == "AAPL"
    for key in EXPECTED_PEAD_KEYS:
        assert key in status, f"Missing key: {key}"

    assert isinstance(status.get("is_active"), bool), "is_active must be bool"
    assert isinstance(status.get("signal_strength"), float), "signal_strength must be float"
    assert status.get("signal_strength") >= 0, "signal_strength must be >= 0"

    print("\n  [PASS] get_pead_status (AAPL) OK")


# ── Test 6: Integration — old heuristic vs new factor model ──────────────────

def test_integration_comparison():
    print(f"\n{DIVIDER}")
    print("  TEST 6: Integration — heuristic vs factor model (AAPL)")
    print(DIVIDER)

    from data.technicals       import get_technicals
    from data.anomaly_detector import compute_anomaly

    tech = get_technicals("AAPL")
    sent = {"sentiment_label": "Neutral", "overall_sentiment": 0.0}

    # Old heuristic path
    old_result = compute_anomaly("AAPL", tech, sent,
                                 check_fundamentals=True,
                                 check_factor_model=False)
    # New factor model path
    new_result = compute_anomaly("AAPL", tech, sent,
                                 check_fundamentals=True,
                                 check_factor_model=True)

    print(f"\n  === Heuristic (check_factor_model=False) ===")
    print(f"  score:          {old_result.get('score')}")
    print(f"  quality_score:  {old_result.get('quality_score')}")
    print(f"  is_watch:       {old_result.get('is_watch')}")
    print(f"  direction:      {old_result.get('direction')}")
    print(f"  factor_model:   {old_result.get('factor_model')}  (should be None)")

    print(f"\n  === Factor Model (check_factor_model=True) ===")
    print(f"  score:                  {new_result.get('score')}")
    print(f"  quality_score:          {new_result.get('quality_score')}")
    print(f"  is_watch:               {new_result.get('is_watch')}")
    print(f"  direction:              {new_result.get('direction')}")
    print(f"  composite_factor_score: {new_result.get('composite_factor_score')}")
    print(f"  pead_candidate:         {new_result.get('pead_candidate')}")
    fm = new_result.get("factor_model") or {}
    print(f"  fm composite_score:     {fm.get('composite_score')}")
    print(f"  fm composite_signal:    {fm.get('composite_signal')}")
    print(f"  fm data_completeness:   {fm.get('data_completeness', 0)*100:.0f}%")

    # Structural assertions
    assert old_result.get("factor_model") is None, \
        "factor_model should be None when check_factor_model=False"
    assert old_result.get("composite_factor_score") is None, \
        "composite_factor_score should be None when check_factor_model=False"

    assert new_result.get("factor_model") is not None, \
        "factor_model should not be None when check_factor_model=True"
    cfs = new_result.get("composite_factor_score")
    assert cfs is not None, "composite_factor_score should not be None"
    assert 0 <= cfs <= 100, f"composite_factor_score {cfs} out of [0, 100]"

    print("\n  [PASS] Integration comparison OK")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nTrading Terminal — Stage 11 Factor Model Tests")
    print(f"Tests: factor model, 6 factors, PEAD tracker, integration comparison\n")

    test_compute_factor_model_aapl()
    test_all_factors_nvda()
    test_composite_signal_tsla()
    test_full_breakdown_jnj()
    test_pead_status()
    test_integration_comparison()

    print(f"\n{DIVIDER}")
    print("  All Stage 11 checks complete.")
    print(DIVIDER)
