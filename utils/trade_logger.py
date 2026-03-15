"""
SQLite trade and signal logger.

Database: trades.db (created at project root on first import).

Tables:
    signals — auto-logged each time an anomaly scan flags a Watch ticker.
    trades  — manually logged when a paper trade is placed / closed.

Public API:
    log_signal(ticker, anomaly_score, signals_triggered, reason, price_at_signal)
    log_trade(ticker, entry_price, qty, side, entry_date=None, strategy_notes="") -> int
    close_trade(trade_id, exit_price, exit_date=None)
    get_performance_summary() -> dict
"""

import logging
import os
import sqlite3
from datetime import datetime

logger = logging.getLogger(__name__)

# Resolve project root (two levels up from utils/)
_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DB_PATH = os.path.join(_ROOT, "trades.db")

_CREATE_SIGNALS = """
CREATE TABLE IF NOT EXISTS signals (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker            TEXT    NOT NULL,
    timestamp         TEXT    NOT NULL,
    anomaly_score     INTEGER NOT NULL,
    signals_triggered TEXT    NOT NULL,
    reason            TEXT,
    price_at_signal   REAL,
    action_taken      TEXT    DEFAULT 'watched'
)
"""

_CREATE_TRADES = """
CREATE TABLE IF NOT EXISTS trades (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker           TEXT    NOT NULL,
    entry_date       TEXT    NOT NULL,
    entry_price      REAL    NOT NULL,
    exit_date        TEXT,
    exit_price       REAL,
    quantity         INTEGER NOT NULL,
    side             TEXT    NOT NULL,
    pnl              REAL,
    pnl_percent      REAL,
    strategy_notes   TEXT,
    outcome          TEXT    DEFAULT 'open'
)
"""


def _conn() -> sqlite3.Connection:
    return sqlite3.connect(_DB_PATH)


def _init_db() -> None:
    try:
        with _conn() as con:
            con.execute(_CREATE_SIGNALS)
            con.execute(_CREATE_TRADES)
            con.commit()
    except Exception as exc:
        logger.error("trade_logger _init_db failed: %s", exc)


# Auto-initialize on import
_init_db()


# ── Public API ─────────────────────────────────────────────────────────────────

def log_signal(
    ticker: str,
    anomaly_score: int,
    signals_triggered: list,
    reason: str,
    price_at_signal: float,
    action_taken: str = "watched",
) -> None:
    """
    Record an anomaly scan Watch hit to the signals table.

    Args:
        ticker:             Stock symbol.
        anomaly_score:      Number of signals triggered.
        signals_triggered:  List of signal description strings.
        reason:             Short summary reason from anomaly detector.
        price_at_signal:    Current price at time of scan.
        action_taken:       'watched', 'bought', 'passed', etc.
    """
    try:
        sigs = (
            "; ".join(signals_triggered)
            if isinstance(signals_triggered, list)
            else str(signals_triggered)
        )
        with _conn() as con:
            con.execute(
                """INSERT INTO signals
                   (ticker, timestamp, anomaly_score, signals_triggered,
                    reason, price_at_signal, action_taken)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    ticker.upper(),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    int(anomaly_score),
                    sigs,
                    reason,
                    price_at_signal,
                    action_taken,
                ),
            )
            con.commit()
    except Exception as exc:
        logger.error("log_signal %s failed: %s", ticker, exc)


def log_trade(
    ticker: str,
    entry_price: float,
    qty: int,
    side: str,
    entry_date: str = None,
    strategy_notes: str = "",
) -> int:
    """
    Insert a new open trade record into the trades table.

    Args:
        ticker:         Stock symbol.
        entry_price:    Fill price per share.
        qty:            Number of shares.
        side:           'buy' or 'sell'.
        entry_date:     ISO date/datetime string; defaults to now.
        strategy_notes: Optional free-text notes.

    Returns:
        New trade's row id (int), or -1 on error.
    """
    try:
        date = entry_date or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with _conn() as con:
            cur = con.execute(
                """INSERT INTO trades
                   (ticker, entry_date, entry_price, quantity, side, strategy_notes, outcome)
                   VALUES (?, ?, ?, ?, ?, ?, 'open')""",
                (ticker.upper(), date, float(entry_price), int(qty), side.lower(), strategy_notes),
            )
            con.commit()
            return cur.lastrowid
    except Exception as exc:
        logger.error("log_trade %s failed: %s", ticker, exc)
        return -1


def close_trade(
    trade_id: int,
    exit_price: float,
    exit_date: str = None,
) -> None:
    """
    Record exit details for an open trade and compute P&L.

    Args:
        trade_id:   Row id returned by log_trade().
        exit_price: Fill price at exit.
        exit_date:  ISO date/datetime string; defaults to now.
    """
    try:
        date = exit_date or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with _conn() as con:
            row = con.execute(
                "SELECT entry_price, quantity, side FROM trades WHERE id=?",
                (trade_id,),
            ).fetchone()
            if not row:
                logger.warning("close_trade: trade_id %d not found", trade_id)
                return
            entry_price, qty, side = row
            if side == "buy":
                pnl = round((exit_price - entry_price) * qty, 2)
            else:
                pnl = round((entry_price - exit_price) * qty, 2)
            cost        = entry_price * qty
            pnl_pct     = round(pnl / cost * 100, 2) if cost else 0
            outcome     = "win" if pnl > 0 else ("loss" if pnl < 0 else "breakeven")
            con.execute(
                """UPDATE trades
                   SET exit_date=?, exit_price=?, pnl=?, pnl_percent=?, outcome=?
                   WHERE id=?""",
                (date, exit_price, pnl, pnl_pct, outcome, trade_id),
            )
            con.commit()
    except Exception as exc:
        logger.error("close_trade %d failed: %s", trade_id, exc)


def get_performance_summary() -> dict:
    """
    Compute aggregate performance stats from the trades table.

    Returns:
        {total_trades, open_trades, closed_trades, wins, losses, breakeven,
         win_rate, total_pnl, avg_pnl, best_trade, worst_trade, error}
    """
    try:
        with _conn() as con:
            rows = con.execute(
                "SELECT outcome, pnl FROM trades"
            ).fetchall()

        if not rows:
            return {
                "total_trades": 0, "open_trades": 0, "closed_trades": 0,
                "wins": 0, "losses": 0, "breakeven": 0,
                "win_rate": None, "total_pnl": 0.0,
                "avg_pnl": None, "best_trade": None, "worst_trade": None,
                "error": None,
            }

        total    = len(rows)
        open_t   = sum(1 for r in rows if r[0] == "open")
        wins     = sum(1 for r in rows if r[0] == "win")
        losses   = sum(1 for r in rows if r[0] == "loss")
        be       = sum(1 for r in rows if r[0] == "breakeven")
        closed   = total - open_t

        closed_pnl = [r[1] for r in rows if r[0] != "open" and r[1] is not None]
        total_pnl  = round(sum(closed_pnl), 2)
        avg_pnl    = round(total_pnl / len(closed_pnl), 2) if closed_pnl else None
        win_rate   = round(wins / closed * 100, 1) if closed else None
        best       = round(max(closed_pnl), 2) if closed_pnl else None
        worst      = round(min(closed_pnl), 2) if closed_pnl else None

        return {
            "total_trades":  total,
            "open_trades":   open_t,
            "closed_trades": closed,
            "wins":          wins,
            "losses":        losses,
            "breakeven":     be,
            "win_rate":      win_rate,
            "total_pnl":     total_pnl,
            "avg_pnl":       avg_pnl,
            "best_trade":    best,
            "worst_trade":   worst,
            "error":         None,
        }
    except Exception as exc:
        logger.error("get_performance_summary failed: %s", exc)
        return {"error": str(exc)}
