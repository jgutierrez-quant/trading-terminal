"""
Price alerts and notification system.

Database: alerts.db (project root).

Tables:
    price_alerts — user-set price level alerts per ticker.
    alert_log    — auto-logged alerts from scans (squeeze, whale, anomaly).

Public API:
    add_price_alert(ticker, target_price, direction, note="")
    remove_price_alert(alert_id)
    get_price_alerts()                -> list[dict]
    check_price_alerts(prices: dict)  -> list[dict]   # prices = {ticker: current_price}
    log_alert(ticker, alert_type, message, score=0)
    get_alert_log(limit=50)           -> list[dict]
    clear_alert_log()
"""

import logging
import os
import sqlite3
from datetime import datetime

logger = logging.getLogger(__name__)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DB_PATH = os.path.join(_ROOT, "alerts.db")

_CREATE_PRICE_ALERTS = """
CREATE TABLE IF NOT EXISTS price_alerts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT    NOT NULL,
    target_price REAL   NOT NULL,
    direction   TEXT    NOT NULL,  -- 'above' or 'below'
    note        TEXT    DEFAULT '',
    created_at  TEXT    NOT NULL,
    triggered   INTEGER DEFAULT 0,
    triggered_at TEXT
)
"""

_CREATE_ALERT_LOG = """
CREATE TABLE IF NOT EXISTS alert_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT    NOT NULL,
    alert_type  TEXT    NOT NULL,  -- 'squeeze', 'whale', 'anomaly', 'price', 'regime'
    message     TEXT    NOT NULL,
    score       INTEGER DEFAULT 0,
    timestamp   TEXT    NOT NULL
)
"""


def _conn() -> sqlite3.Connection:
    return sqlite3.connect(_DB_PATH)


def _init_db() -> None:
    try:
        with _conn() as con:
            con.execute(_CREATE_PRICE_ALERTS)
            con.execute(_CREATE_ALERT_LOG)
            con.commit()
    except Exception as exc:
        logger.error("alerts _init_db failed: %s", exc)


_init_db()


# ── Price Alerts ──────────────────────────────────────────────────────────────

def add_price_alert(ticker: str, target_price: float, direction: str, note: str = "") -> int:
    try:
        with _conn() as con:
            cur = con.execute(
                """INSERT INTO price_alerts (ticker, target_price, direction, note, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (ticker.upper(), float(target_price), direction.lower(),
                 note, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            )
            con.commit()
            return cur.lastrowid
    except Exception as exc:
        logger.error("add_price_alert failed: %s", exc)
        return -1


def remove_price_alert(alert_id: int) -> None:
    try:
        with _conn() as con:
            con.execute("DELETE FROM price_alerts WHERE id=?", (alert_id,))
            con.commit()
    except Exception as exc:
        logger.error("remove_price_alert failed: %s", exc)


def get_price_alerts() -> list:
    try:
        with _conn() as con:
            rows = con.execute(
                "SELECT id, ticker, target_price, direction, note, created_at, triggered "
                "FROM price_alerts WHERE triggered=0 ORDER BY ticker, target_price"
            ).fetchall()
        return [
            {"id": r[0], "ticker": r[1], "target_price": r[2], "direction": r[3],
             "note": r[4], "created_at": r[5], "triggered": r[6]}
            for r in rows
        ]
    except Exception as exc:
        logger.error("get_price_alerts failed: %s", exc)
        return []


def check_price_alerts(prices: dict) -> list:
    """Check active alerts against current prices. Returns list of triggered alerts."""
    triggered = []
    try:
        alerts = get_price_alerts()
        for a in alerts:
            ticker = a["ticker"]
            price = prices.get(ticker)
            if price is None:
                continue
            hit = False
            if a["direction"] == "above" and price >= a["target_price"]:
                hit = True
            elif a["direction"] == "below" and price <= a["target_price"]:
                hit = True
            if hit:
                a["current_price"] = price
                triggered.append(a)
                with _conn() as con:
                    con.execute(
                        "UPDATE price_alerts SET triggered=1, triggered_at=? WHERE id=?",
                        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), a["id"]),
                    )
                    con.commit()
                log_alert(ticker, "price",
                          f"{ticker} hit ${price:.2f} ({a['direction']} ${a['target_price']:.2f})")
    except Exception as exc:
        logger.error("check_price_alerts failed: %s", exc)
    return triggered


# ── Alert Log ─────────────────────────────────────────────────────────────────

def log_alert(ticker: str, alert_type: str, message: str, score: int = 0) -> None:
    try:
        with _conn() as con:
            con.execute(
                """INSERT INTO alert_log (ticker, alert_type, message, score, timestamp)
                   VALUES (?, ?, ?, ?, ?)""",
                (ticker.upper(), alert_type, message, score,
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            )
            con.commit()
    except Exception as exc:
        logger.error("log_alert failed: %s", exc)


def get_alert_log(limit: int = 50) -> list:
    try:
        with _conn() as con:
            rows = con.execute(
                "SELECT id, ticker, alert_type, message, score, timestamp "
                "FROM alert_log ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {"id": r[0], "ticker": r[1], "alert_type": r[2], "message": r[3],
             "score": r[4], "timestamp": r[5]}
            for r in rows
        ]
    except Exception as exc:
        logger.error("get_alert_log failed: %s", exc)
        return []


def clear_alert_log() -> None:
    try:
        with _conn() as con:
            con.execute("DELETE FROM alert_log")
            con.commit()
    except Exception as exc:
        logger.error("clear_alert_log failed: %s", exc)
