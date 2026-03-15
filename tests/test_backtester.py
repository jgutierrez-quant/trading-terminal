"""
Stage 9 smoke test — backtesting engine with signal quality improvements.
Run from project root:
    venv/Scripts/python tests/test_backtester.py

Tests:
    1. get_historical_data (AAPL) — shape, columns, no NaN on key indicators
    2. generate_signals (AAPL)    — quality_score, direction filter, watch_flag
    3. run_backtest (AAPL, NVDA, TSLA) — new metrics: avg_hold_days, exit_breakdown
    4. run_multi_backtest (all 3)  — comparison DataFrame
    5. Threshold 3 vs 4 side-by-side — win rate and profit factor
"""

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.WARNING)

from data.backtester import (
    get_historical_data,
    generate_signals,
    run_backtest,
    run_multi_backtest,
)

DIVIDER  = "=" * 64
DIVIDER2 = "-" * 64

REQUIRED_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'rsi', 'macd_hist', 'bb_upper', 'bb_lower',
    'sma20', 'sma50', 'vol_ratio', 'atr', 'vwap_proxy',
]


# ── Test 1: get_historical_data ───────────────────────────────────────────────

def test_get_historical_data():
    print(f"\n{DIVIDER}")
    print("  TEST 1: get_historical_data (AAPL, 2y)")
    print(DIVIDER)

    df = get_historical_data('AAPL', period='2y')

    print(f"\n  Shape:      {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Date range: {str(df.index[0])[:10]}  ->  {str(df.index[-1])[:10]}")
    print(f"\n  Last-row indicator values:")

    last = df.iloc[-1]
    for col in REQUIRED_COLS:
        val   = last.get(col, None)
        is_ok = val is not None and str(val) != 'nan'
        flag  = "OK" if is_ok else "NaN"
        val_s = f"{float(val):.4f}" if is_ok else "NaN"
        print(f"    {col:<15} {val_s:>12}  [{flag}]")

    assert df.shape[0] >= 100, f"Expected >= 100 rows, got {df.shape[0]}"
    for col in REQUIRED_COLS:
        assert col in df.columns, f"Missing column: {col}"
    last_row = df.iloc[-1]
    for col in ['rsi', 'macd_hist', 'bb_upper', 'bb_lower', 'sma20', 'atr']:
        v = last_row[col]
        assert str(v) != 'nan', f"NaN in last row for {col}"

    print("\n  [PASS] get_historical_data() OK")


# ── Test 2: generate_signals (Stage 9) ───────────────────────────────────────

def test_generate_signals():
    print(f"\n{DIVIDER}")
    print("  TEST 2: generate_signals (AAPL, 2y) — Stage 9 quality + direction filter")
    print(DIVIDER)

    df = generate_signals(get_historical_data('AAPL', period='2y'))

    watch_days   = int(df['watch_flag'].sum())
    watch_thr3   = int((df['signal_count'] >= 3).sum())   # old threshold count
    watch_thr4   = int((df['signal_count'] >= 4).sum())   # raw new threshold count
    print(f"\n  Total rows:            {len(df)}")
    print(f"  Raw score >= 3 days:   {watch_thr3}")
    print(f"  Raw score >= 4 days:   {watch_thr4}")
    print(f"  Watch-flag days (thr4 + direction filter): {watch_days}")
    print(f"  Max quality score:     {int(df['quality_score'].max())}")
    print(f"  Mean quality score:    {df['quality_score'].mean():.1f}")
    print(f"  Direction breakdown:   {df['direction'].value_counts().to_dict()}")

    watch_rows = df[df['watch_flag']].tail(5)
    if not watch_rows.empty:
        print(f"\n  Last 5 watch-flag rows (thr=4 + direction filter):")
        for dt, row in watch_rows.iterrows():
            print(
                f"    {str(dt)[:10]}  score={row['signal_count']}  "
                f"Q={row['quality_score']}  dir={row['direction']:<8}  "
                f"{row['signal_list'][:50]}"
            )

    sig_cols = ['signal_count', 'signal_list', 'quality_score', 'direction', 'watch_flag']
    for col in sig_cols:
        assert col in df.columns, f"Missing signal column: {col}"
    assert (df['signal_count'] >= 0).all(), "signal_count must be >= 0"
    assert (df['quality_score'] >= 0).all(), "quality_score must be >= 0"
    assert (df['quality_score'] <= 100).all(), "quality_score must be <= 100"
    valid_dirs = {'Bullish', 'Bearish', 'Neutral'}
    bad_dirs   = set(df['direction'].unique()) - valid_dirs
    assert not bad_dirs, f"Unexpected direction values: {bad_dirs}"

    print("\n  [PASS] generate_signals() OK")


# ── Test 3: run_backtest ──────────────────────────────────────────────────────

def _print_result(result: dict) -> None:
    ticker = result['ticker']
    error  = result.get('error')
    trades = result.get('trades', [])

    print(f"\n  Ticker:           {ticker}")
    if error and not trades:
        print(f"  ERROR: {error}")
        return

    print(f"  Total Trades:     {result['total_trades']}")
    print(f"  Win Rate:         {result['win_rate']:.1f}%")
    print(f"  Profit Factor:    {result['profit_factor']:.2f}")
    print(f"  Total Return:     {result['total_return_pct']:+.2f}%")
    print(f"  Max Drawdown:     {result['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe Ratio:     {result['sharpe_ratio']:.2f}")
    print(f"  Avg Hold Days:    {result['avg_hold_days']:.1f}d")
    print(f"  Initial Capital:  ${result['initial_capital']:,.2f}")
    print(f"  Final Capital:    ${result['final_capital']:,.2f}")

    eb = result.get('exit_breakdown', {})
    if eb:
        parts = ', '.join(f"{k}: {v:.1f}%" for k, v in eb.items())
        print(f"  Exit Breakdown:   {parts}")

    bt = result.get('best_trade')
    wt = result.get('worst_trade')
    if bt:
        print(f"  Best Trade:       {bt['entry_date']}  {bt['side']:<5}  "
              f"{bt['pnl_pct']:+.2f}%  Q={bt.get('quality_score',0)}  ({bt['outcome']})")
    if wt:
        print(f"  Worst Trade:      {wt['entry_date']}  {wt['side']:<5}  "
              f"{wt['pnl_pct']:+.2f}%  Q={wt.get('quality_score',0)}  ({wt['outcome']})")

    if trades:
        print(f"\n  {'Entry':>10} {'Exit':>10} {'Side':>5} {'Entry$':>9} {'Exit$':>9} "
              f"{'Shrs':>5} {'P&L':>8} {'Pct%':>7} {'Q':>3} {'Out':>8}  Signals")
        print(f"  {DIVIDER2}")
        for t in (trades[-20:] if len(trades) > 20 else trades):
            print(
                f"  {t['entry_date']:>10} {t['exit_date']:>10} {t['side']:>5} "
                f"{t['entry_price']:>9.2f} {t['exit_price']:>9.2f} {t['shares']:>5} "
                f"{t['pnl']:>+8.2f} {t['pnl_pct']:>+6.1f}% "
                f"{t.get('quality_score',0):>3} {t['outcome']:>8}  "
                f"{t['signals'][:38]}"
            )
        if len(trades) > 20:
            print(f"  ... ({len(trades) - 20} earlier trades not shown)")


def test_run_backtest():
    print(f"\n{DIVIDER}")
    print("  TEST 3: run_backtest (Stage 9) — AAPL, NVDA, TSLA")
    print(DIVIDER)

    for ticker in ['AAPL', 'NVDA', 'TSLA']:
        result = run_backtest(
            ticker,
            initial_capital=1000,
            risk_percent=2.0,
            risk_reward=2.0,
            hold_days=10,
            quality_threshold=60,
            check_earnings=True,
        )
        _print_result(result)

        # Assertions
        for key in ['trades', 'equity_curve', 'win_rate', 'total_return_pct',
                    'avg_hold_days', 'exit_breakdown', 'final_capital']:
            assert key in result, f"Missing result key: {key}"
        assert isinstance(result['trades'], list),       "trades must be a list"
        assert isinstance(result['equity_curve'], list), "equity_curve must be a list"
        assert isinstance(result['exit_breakdown'], dict), "exit_breakdown must be a dict"
        assert set(result['exit_breakdown'].keys()) == {'Target', 'Stop', 'Timeout', 'Trend'}, \
            "exit_breakdown must have all 4 keys"

        if result['trades']:
            t = result['trades'][0]
            for k in ['entry_date', 'exit_date', 'entry_price', 'exit_price',
                      'shares', 'side', 'pnl', 'pnl_pct', 'outcome',
                      'signals', 'signal_count', 'quality_score']:
                assert k in t, f"Trade missing key: {k}"

        print(f"\n  [PASS] run_backtest({ticker}) OK")


# ── Test 4: run_multi_backtest ────────────────────────────────────────────────

def test_run_multi_backtest():
    print(f"\n{DIVIDER}")
    print("  TEST 4: run_multi_backtest (Stage 9) — AAPL, NVDA, TSLA")
    print(DIVIDER)

    df = run_multi_backtest(
        ['AAPL', 'NVDA', 'TSLA'],
        initial_capital=1000,
        quality_threshold=60,
        hold_days=10,
    )

    print(f"\n  Multi-ticker comparison ({len(df)} rows, sorted by Total Return %):")
    print(f"  {DIVIDER2}")
    if not df.empty:
        print(df.to_string(index=False))

    expected_cols = [
        'Ticker', 'Total Return %', 'Win Rate %', 'Profit Factor',
        'Max Drawdown %', 'Sharpe Ratio', 'Avg Hold Days', 'Total Trades', 'Final Capital',
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"
    assert len(df) == 3, f"Expected 3 rows, got {len(df)}"

    print("\n  [PASS] run_multi_backtest() OK")


# ── Test 5: Threshold 3 vs 4 comparison ──────────────────────────────────────

def test_threshold_comparison():
    print(f"\n{DIVIDER}")
    print("  TEST 5: Threshold 3 vs 4 comparison — AAPL, NVDA, TSLA")
    print("          (quality_threshold=0, check_earnings=False to isolate effect)")
    print(DIVIDER)

    tickers = ['AAPL', 'NVDA', 'TSLA']
    rows = []

    for t in tickers:
        r3 = run_backtest(t, 1000, watch_threshold=3, quality_threshold=0, check_earnings=False)
        r4 = run_backtest(t, 1000, watch_threshold=4, quality_threshold=0, check_earnings=False)
        rows.append({
            'Ticker':          t,
            'Trades (thr=3)':  r3['total_trades'],
            'WinRate% (thr=3)': r3['win_rate'],
            'PF (thr=3)':      r3['profit_factor'],
            'Return% (thr=3)': r3['total_return_pct'],
            'Trades (thr=4)':  r4['total_trades'],
            'WinRate% (thr=4)': r4['win_rate'],
            'PF (thr=4)':      r4['profit_factor'],
            'Return% (thr=4)': r4['total_return_pct'],
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    print(f"\n  Side-by-side comparison:")
    print(f"  {DIVIDER2}")
    print(df.to_string(index=False))

    print("\n  Key observations:")
    for r in rows:
        t   = r['Ticker']
        dwr = r['WinRate% (thr=4)'] - r['WinRate% (thr=3)']
        dpf = r['PF (thr=4)'] - r['PF (thr=3)']
        dtr = r['Trades (thr=4)'] - r['Trades (thr=3)']
        print(f"    {t}: trades {r['Trades (thr=3)']} -> {r['Trades (thr=4)']} ({dtr:+d})"
              f"  |  Win Rate {dwr:+.1f}%  |  PF {dpf:+.2f}")

    assert len(rows) == 3, "Expected 3 rows"
    print("\n  [PASS] threshold_comparison() OK")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nTrading Terminal — Stage 9 Signal Quality Improvements")
    print(f"Tests: historical_data, signals (quality/direction), backtest x3, multi, thr3vs4\n")

    test_get_historical_data()
    test_generate_signals()
    test_run_backtest()
    test_run_multi_backtest()
    test_threshold_comparison()

    print(f"\n{DIVIDER}")
    print("  All Stage 9 checks complete.")
    print(DIVIDER)
