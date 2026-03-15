"""
Stage 8B smoke test — backtesting engine.
Run from project root:
    venv/Scripts/python tests/test_backtester.py
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
        val  = last.get(col, None)
        is_ok = val is not None and str(val) != 'nan'
        flag  = "OK" if is_ok else "NaN"
        val_s = f"{float(val):.4f}" if is_ok else "NaN"
        print(f"    {col:<15} {val_s:>12}  [{flag}]")

    # Assertions
    assert df.shape[0] >= 100, f"Expected >= 100 rows, got {df.shape[0]}"
    for col in REQUIRED_COLS:
        assert col in df.columns, f"Missing column: {col}"
    last_row = df.iloc[-1]
    for col in ['rsi', 'macd_hist', 'bb_upper', 'bb_lower', 'sma20', 'atr']:
        v = last_row[col]
        assert str(v) != 'nan', f"NaN in last row for {col}"

    print("\n  [PASS] get_historical_data() OK")


# ── Test 2: generate_signals ──────────────────────────────────────────────────

def test_generate_signals():
    print(f"\n{DIVIDER}")
    print("  TEST 2: generate_signals (AAPL, 2y)")
    print(DIVIDER)

    df = generate_signals(get_historical_data('AAPL', period='2y'))

    watch_days = int(df['watch_flag'].sum())
    print(f"\n  Total rows:       {len(df)}")
    print(f"  Watch-flag days:  {watch_days}")
    print(f"  Max signal count: {int(df['signal_count'].max())}")
    print(f"  Direction breakdown: {df['direction'].value_counts().to_dict()}")

    # Show last 5 watch-flag rows
    watch_rows = df[df['watch_flag']].tail(5)
    if not watch_rows.empty:
        print(f"\n  Last 5 watch-flag rows:")
        for dt, row in watch_rows.iterrows():
            date_s = str(dt)[:10]
            print(
                f"    {date_s}  score={row['signal_count']}  "
                f"dir={row['direction']:<8}  {row['signal_list'][:55]}"
            )

    # Assertions
    sig_cols = [
        'rsi_signal', 'macd_signal', 'bb_signal', 'vol_signal',
        'vwap_signal', 'price_vs_sma20', 'price_vs_sma50',
        'signal_count', 'signal_list', 'watch_flag', 'direction',
    ]
    for col in sig_cols:
        assert col in df.columns, f"Missing signal column: {col}"
    assert (df['signal_count'] >= 0).all(), "signal_count must be >= 0"
    valid_dirs = {'Bullish', 'Bearish', 'Mixed', 'Neutral'}
    bad_dirs = set(df['direction'].unique()) - valid_dirs
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
    print(f"  Avg Win:          {result['avg_win_pct']:+.2f}%")
    print(f"  Avg Loss:         {result['avg_loss_pct']:+.2f}%")
    print(f"  Initial Capital:  ${result['initial_capital']:,.2f}")
    print(f"  Final Capital:    ${result['final_capital']:,.2f}")

    bt = result.get('best_trade')
    wt = result.get('worst_trade')
    if bt:
        print(f"  Best Trade:       {bt['entry_date']}  {bt['side']:<5}  {bt['pnl_pct']:+.2f}%  ({bt['outcome']})")
    if wt:
        print(f"  Worst Trade:      {wt['entry_date']}  {wt['side']:<5}  {wt['pnl_pct']:+.2f}%  ({wt['outcome']})")

    if trades:
        print(f"\n  {'Entry':>10} {'Exit':>10} {'Side':>5} {'Entry$':>9} {'Exit$':>9} "
              f"{'Shrs':>5} {'P&L':>8} {'Pct%':>7} {'Out':>8}  Signals")
        print(f"  {DIVIDER2}")
        display = trades[-20:] if len(trades) > 20 else trades
        for t in display:
            print(
                f"  {t['entry_date']:>10} {t['exit_date']:>10} {t['side']:>5} "
                f"{t['entry_price']:>9.2f} {t['exit_price']:>9.2f} {t['shares']:>5} "
                f"{t['pnl']:>+8.2f} {t['pnl_pct']:>+6.1f}% {t['outcome']:>8}  "
                f"{t['signals'][:40]}"
            )
        if len(trades) > 20:
            print(f"  ... ({len(trades) - 20} earlier trades not shown)")


def test_run_backtest():
    print(f"\n{DIVIDER}")
    print("  TEST 3: run_backtest — AAPL, NVDA, TSLA")
    print(DIVIDER)

    for ticker in ['AAPL', 'NVDA', 'TSLA']:
        result = run_backtest(
            ticker,
            initial_capital=1000,
            risk_percent=2.0,
            risk_reward=2.0,
            hold_days=5,
        )
        _print_result(result)

        # Assertions
        for key in ['trades', 'equity_curve', 'win_rate', 'total_return_pct',
                    'max_drawdown_pct', 'sharpe_ratio', 'final_capital']:
            assert key in result, f"Missing result key: {key}"
        assert isinstance(result['trades'],       list), "trades must be a list"
        assert isinstance(result['equity_curve'], list), "equity_curve must be a list"
        if result['trades']:
            t = result['trades'][0]
            for k in ['entry_date', 'exit_date', 'entry_price', 'exit_price',
                      'shares', 'side', 'pnl', 'pnl_pct', 'outcome',
                      'signals', 'signal_count']:
                assert k in t, f"Trade missing key: {k}"

        print(f"\n  [PASS] run_backtest({ticker}) OK")


# ── Test 4: run_multi_backtest ────────────────────────────────────────────────

def test_run_multi_backtest():
    print(f"\n{DIVIDER}")
    print("  TEST 4: run_multi_backtest — AAPL, NVDA, TSLA")
    print(DIVIDER)

    df = run_multi_backtest(['AAPL', 'NVDA', 'TSLA'], initial_capital=1000)

    print(f"\n  Multi-ticker comparison ({len(df)} rows, sorted by Total Return %):")
    print(f"  {DIVIDER2}")
    if not df.empty:
        print(df.to_string(index=False))

    # Assertions
    assert hasattr(df, 'columns'), "Result must be a DataFrame"
    expected_cols = [
        'Ticker', 'Total Return %', 'Win Rate %', 'Profit Factor',
        'Max Drawdown %', 'Sharpe Ratio', 'Total Trades', 'Final Capital',
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"
    assert len(df) == 3, f"Expected 3 rows, got {len(df)}"

    print("\n  [PASS] run_multi_backtest() OK")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nTrading Terminal — Stage 8B Backtesting Engine")
    print(f"Tests: historical_data, generate_signals, run_backtest (x3), multi_backtest\n")

    test_get_historical_data()
    test_generate_signals()
    test_run_backtest()
    test_run_multi_backtest()

    print(f"\n{DIVIDER}")
    print("  All Stage 8B checks complete.")
    print(DIVIDER)
