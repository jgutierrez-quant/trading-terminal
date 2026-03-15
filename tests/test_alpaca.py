"""
Stage 8A smoke test — Alpaca client, risk manager, and trade logger.
Run from project root:
    venv/Scripts/python tests/test_alpaca.py
"""

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.WARNING)

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from data.alpaca_client  import (get_account, get_positions, get_orders,
                                  place_order, close_position, cancel_order,
                                  get_real_time_quote)
from utils.risk_manager  import (calculate_position_size, calculate_stop_loss,
                                  calculate_take_profit, validate_trade,
                                  get_market_hours)
from utils.trade_logger  import (log_signal, log_trade, close_trade,
                                  get_performance_summary)

DIVIDER  = "=" * 64
DIVIDER2 = "-" * 64


# ── Alpaca Client Tests ───────────────────────────────────────────────────────

def test_get_account():
    print(f"\n{DIVIDER}")
    print("  ALPACA ACCOUNT")
    print(DIVIDER)

    acct = get_account()
    if acct.get("error"):
        print(f"  ERROR: {acct['error']}")
    else:
        print(f"  Portfolio Value : ${acct.get('portfolio_value', 0):>12,.2f}")
        print(f"  Equity          : ${acct.get('equity', 0):>12,.2f}")
        print(f"  Cash            : ${acct.get('cash', 0):>12,.2f}")
        print(f"  Buying Power    : ${acct.get('buying_power', 0):>12,.2f}")
        daily_pnl = acct.get('daily_pnl', 0)
        daily_pct = acct.get('daily_pnl_pct', 0)
        sign = "+" if daily_pnl >= 0 else ""
        print(f"  Daily P&L       :  {sign}${daily_pnl:,.2f}  ({sign}{daily_pct:.2f}%)")

    assert isinstance(acct, dict), "get_account must return a dict"
    print("\n  [PASS] get_account() OK")


def test_get_positions():
    print(f"\n{DIVIDER}")
    print("  ALPACA POSITIONS")
    print(DIVIDER)

    positions = get_positions()
    print(f"\n  Open positions: {len(positions)}")
    if positions:
        print(f"  {'Ticker':<8} {'Qty':>6} {'Avg Entry':>10} {'Current':>10} "
              f"{'P&L':>10} {'P&L%':>7}")
        print(f"  {DIVIDER2}")
        for p in positions:
            pl   = p.get("unrealized_pl", 0)
            plpc = p.get("unrealized_plpc", 0)
            sign = "+" if pl >= 0 else ""
            print(f"  {p['ticker']:<8} {p['qty']:>6.0f} "
                  f"${p.get('avg_entry_price',0):>9.2f} "
                  f"${p.get('current_price',0):>9.2f} "
                  f"  {sign}${pl:>8.2f} {sign}{plpc:>6.2f}%")
    else:
        print("  (No open positions)")

    assert isinstance(positions, list), "get_positions must return a list"
    print("\n  [PASS] get_positions() OK")


def test_get_orders():
    print(f"\n{DIVIDER}")
    print("  ALPACA RECENT ORDERS")
    print(DIVIDER)

    orders = get_orders(status="all", limit=10)
    print(f"\n  Recent orders: {len(orders)}")
    if orders:
        print(f"  {'Ticker':<8} {'Qty':>5} {'Side':<5} {'Type':<10} {'Status':<12} "
              f"{'Fill':>8}  Submitted")
        print(f"  {DIVIDER2}")
        for o in orders[:10]:
            fp  = o.get("filled_price")
            fp_str = f"${fp:.2f}" if fp else "—"
            print(f"  {o['ticker']:<8} {o['qty']:>5.0f} {o['side']:<5} "
                  f"{o['order_type']:<10} {o['status']:<12} {fp_str:>8}  "
                  f"{o.get('submitted_at','')[:16]}")
    else:
        print("  (No orders found)")

    assert isinstance(orders, list), "get_orders must return a list"
    print("\n  [PASS] get_orders() OK")


def test_get_real_time_quote():
    print(f"\n{DIVIDER}")
    print("  REAL-TIME QUOTE (AAPL)")
    print(DIVIDER)

    quote = get_real_time_quote("AAPL")
    if quote.get("error"):
        print(f"  NOTE: {quote['error']} (may require funded IEX subscription)")
    else:
        print(f"  Ticker : {quote.get('ticker')}")
        print(f"  Bid    : ${quote.get('bid', 'N/A')}")
        print(f"  Ask    : ${quote.get('ask', 'N/A')}")
        print(f"  Mid    : ${quote.get('mid', 'N/A')}")
        print(f"  Time   : {quote.get('timestamp', 'N/A')}")

    assert isinstance(quote, dict), "get_real_time_quote must return a dict"
    assert "ticker" in quote, "missing ticker"
    print("\n  [PASS] get_real_time_quote() OK")


# ── Risk Manager Tests ────────────────────────────────────────────────────────

def test_risk_manager():
    print(f"\n{DIVIDER}")
    print("  RISK MANAGER")
    print(DIVIDER)

    account_value = 100_000.0
    entry         = 150.0
    stop          = 145.0   # $5 stop distance

    shares = calculate_position_size(account_value, 1.0, entry, stop)
    print(f"\n  Position Sizing (${account_value:,.0f} acct, 1% risk, "
          f"${entry} entry, ${stop} stop)")
    print(f"    Stop distance : ${abs(entry - stop):.2f}")
    print(f"    Dollar risk   : ${account_value * 0.01:,.2f}")
    print(f"    Shares        : {shares}")
    # 20% max-position cap: 100,000 × 20% / 150 = 133 shares (not raw 200)
    assert shares == 133, f"Expected 133 shares (20% cap applied), got {shares}"

    stop_l = calculate_stop_loss("AAPL", entry, "buy", atr_multiplier=2.0)
    print(f"\n  Stop Loss (AAPL, entry=${entry}, long, 2x ATR)")
    print(f"    Stop price    : ${stop_l}")
    assert stop_l is not None and stop_l < entry, "Long stop must be below entry"

    target = calculate_take_profit(entry, stop_l, "buy", risk_reward=2.0)
    print(f"\n  Take Profit (2:1 R:R)")
    print(f"    Entry         : ${entry}")
    print(f"    Stop          : ${stop_l}")
    print(f"    Target        : ${target}")
    assert target is not None and target > entry, "Long target must be above entry"

    mh = get_market_hours()
    print(f"\n  Market Hours")
    print(f"    Session       : {mh.get('market_session')}")
    print(f"    Is Open       : {mh.get('is_open')}")
    print(f"    Current ET    : {mh.get('current_time_et')}")
    assert "is_open" in mh, "missing is_open"

    val = validate_trade("AAPL", "buy", account_value, account_value * 0.5)
    print(f"\n  Trade Validation (AAPL buy, 50% buying power)")
    print(f"    Valid    : {val['valid']}")
    for w in val.get("warnings", []):
        print(f"    Warning  : {w}")
    for e in val.get("errors", []):
        print(f"    Error    : {e}")
    assert isinstance(val, dict) and "valid" in val, "validate_trade must return dict with 'valid'"

    print("\n  [PASS] risk_manager OK")


# ── Trade Logger Tests ────────────────────────────────────────────────────────

def test_trade_logger():
    print(f"\n{DIVIDER}")
    print("  TRADE LOGGER")
    print(DIVIDER)

    # Log a signal
    log_signal(
        ticker             = "TEST",
        anomaly_score      = 4,
        signals_triggered  = ["Bearish MACD Cross", "High Volume", "Below VWAP"],
        reason             = "Test signal for smoke test",
        price_at_signal    = 99.99,
    )
    print("\n  Logged test signal for TEST @ $99.99")

    # Log a trade and close it
    trade_id = log_trade(
        ticker         = "TEST",
        entry_price    = 100.00,
        qty            = 10,
        side           = "buy",
        strategy_notes = "Stage 8A smoke test trade",
    )
    print(f"  Logged test buy trade, id={trade_id}")
    assert trade_id > 0, f"log_trade should return positive id, got {trade_id}"

    close_trade(trade_id, exit_price=105.00)
    print(f"  Closed trade {trade_id} @ $105.00 (expected +$50.00 P&L)")

    # Check performance summary
    perf = get_performance_summary()
    print(f"\n  Performance Summary")
    print(f"    Total trades  : {perf.get('total_trades')}")
    print(f"    Open trades   : {perf.get('open_trades')}")
    print(f"    Closed trades : {perf.get('closed_trades')}")
    print(f"    Wins          : {perf.get('wins')}")
    print(f"    Losses        : {perf.get('losses')}")
    win_rate = perf.get('win_rate')
    print(f"    Win Rate      : {f'{win_rate:.1f}%' if win_rate is not None else 'N/A'}")
    total_pnl = perf.get('total_pnl', 0)
    print(f"    Total P&L     : ${total_pnl:+,.2f}")
    best  = perf.get('best_trade')
    worst = perf.get('worst_trade')
    print(f"    Best trade    : {f'${best:+,.2f}' if best is not None else 'N/A'}")
    print(f"    Worst trade   : {f'${worst:+,.2f}' if worst is not None else 'N/A'}")

    assert perf.get("error") is None, f"get_performance_summary returned error: {perf.get('error')}"
    assert perf.get("total_trades", 0) >= 1, "Should have at least 1 trade after logging"
    print("\n  [PASS] trade_logger OK")


if __name__ == "__main__":
    print(f"\nTrading Terminal — Stage 8A Alpaca + Risk + Logger Check")
    print(f"Tests: account, positions, orders, quote, risk_manager, trade_logger\n")

    test_get_account()
    test_get_positions()
    test_get_orders()
    test_get_real_time_quote()
    test_risk_manager()
    test_trade_logger()

    print(f"\n{DIVIDER}")
    print("  All Stage 8A checks complete.")
    print(DIVIDER)
