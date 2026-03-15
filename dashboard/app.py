"""
Trading Terminal — Bloomberg-style Streamlit dashboard.
Launch from project root:  streamlit run dashboard/app.py
Or double-click:           run.bat
"""

import sys
import os
import time
import logging
from datetime import datetime

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv(os.path.join(_ROOT, ".env"))

# Import sentiment package (applies urllib3 v2 compat patch for pytrends)
import sentiment  # noqa: F401

from data.market_data               import get_ticker_data
from data.technicals                import get_technicals
from data.anomaly_detector          import compute_anomaly
from sentiment.sentiment_aggregator  import get_sentiment
from sentiment.yahoo_news_client    import get_news_sentiment as _yahoo_fast
from utils.watchlist                import load_watchlist, add_ticker, remove_ticker
from utils.macro_data               import get_macro_data
from data.alpaca_client             import (get_account, get_positions, get_orders,
                                            place_order, place_bracket_order,
                                            close_position, cancel_order)
from utils.risk_manager             import (calculate_position_size, calculate_stop_loss,
                                            calculate_take_profit, validate_trade,
                                            get_market_hours)
from utils.trade_logger             import log_signal, log_trade, get_performance_summary, get_all_trades, close_trade
from utils.alerts                   import (add_price_alert, remove_price_alert, get_price_alerts,
                                            check_price_alerts, log_alert, get_alert_log, clear_alert_log)
from utils.market_regime            import get_market_regime
from dashboard.backtest_tab         import render_backtest_tab
from dashboard.performance_tab     import render_performance_tab
from data.fundamentals              import get_fundamentals, score_fundamentals, get_short_squeeze_score
from data.catalyst_detector         import detect_catalysts, scan_catalysts
from data.whale_detector            import detect_whales, scan_whales
from data.trade_coach               import analyze_setup

logging.basicConfig(level=logging.WARNING)

# ── Colors ────────────────────────────────────────────────────────────────────
from utils.config import GREEN, RED, YELLOW, BLUE, NEUTRAL, BG, BG2

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trading Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    #MainMenu, footer {visibility: hidden;}
    .block-container {padding-top: 0.75rem; padding-bottom: 0;}

    [data-testid="metric-container"] {
        background: #1a1d24;
        border: 1px solid #2a2d35;
        border-radius: 5px;
        padding: 8px 12px;
    }
    section[data-testid="stSidebar"] .stButton > button {
        background: #1a1d24;
        border: 1px solid #2a2d35;
        color: #fafafa;
        font-family: monospace;
        font-size: 0.78rem;
        text-align: left;
        width: 100%;
        padding: 5px 8px;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        border-color: #00ff88;
        color: #00ff88;
    }
    section[data-testid="stSidebar"] > div:first-child {padding-top: 1rem;}
    div[data-testid="stHorizontalBlock"] {gap: 0.5rem;}
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {gap: 4px;}
    .stTabs [data-baseweb="tab"] {
        background: #1a1d24;
        border-radius: 4px 4px 0 0;
        color: #888;
        font-size: 0.82rem;
    }
    .stTabs [aria-selected="true"] {
        background: #2a2d35 !important;
        color: #00ff88 !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = "AAPL"
if "next_refresh" not in st.session_state:
    st.session_state.next_refresh = time.time() + 60
if "pending_order" not in st.session_state:
    st.session_state.pending_order = None     # dict when an order is awaiting confirmation
if "logged_signals_today" not in st.session_state:
    st.session_state.logged_signals_today = set()  # "TICKER_YYYY-MM-DD" keys
if "logged_alerts_today" not in st.session_state:
    st.session_state.logged_alerts_today = set()   # dedup alert logging per session
if "_scan_score_map" not in st.session_state:
    st.session_state._scan_score_map = {}          # {ticker: quality_score} from last scan


# ── Cached fetchers — Market View ─────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def cached_market_data(ticker: str) -> dict:
    return get_ticker_data(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def cached_sentiment(ticker: str) -> dict:
    return get_sentiment(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def cached_technicals(ticker: str) -> dict:
    return get_technicals(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def cached_anomaly(ticker: str) -> dict:
    """Calls cached_technicals + cached_sentiment — avoids duplicate API calls.
    Enables earnings proximity, sector momentum, and fundamentals checks for the dashboard view."""
    tech = cached_technicals(ticker)
    sent = cached_sentiment(ticker)
    return compute_anomaly(ticker, tech, sent, check_earnings=True,
                           check_sector=True, check_fundamentals=True,
                           check_factor_model=True,
                           check_catalysts=True)


@st.cache_data(ttl=300, show_spinner=False)
def cached_yahoo_fast(ticker: str) -> dict:
    return _yahoo_fast(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_fundamentals(ticker: str) -> dict:
    return get_fundamentals(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_factor_model(ticker: str) -> dict:
    from data.factor_model import compute_factor_model
    return compute_factor_model(ticker)


@st.cache_data(ttl=900, show_spinner=False)
def cached_catalysts(ticker: str) -> dict:
    return detect_catalysts(ticker)


@st.cache_data(ttl=900, show_spinner=False)
def cached_whales(ticker: str) -> dict:
    return detect_whales(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def cached_trade_coach(ticker: str, account_value: float = 100_000, risk_pct: float = 1.0) -> dict:
    return analyze_setup(ticker, account_value=account_value, risk_pct=risk_pct)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_macro() -> list:
    return get_macro_data()


@st.cache_data(ttl=60, show_spinner=False)
def cached_daily_bars(ticker: str) -> list:
    """Fallback: 1-month daily bars from yfinance (when Polygon intraday unavailable)."""
    try:
        hist = yf.Ticker(ticker).history(period="1mo")
        if hist.empty:
            return []
        bars = []
        for dt, row in hist.iterrows():
            bars.append({
                "timestamp": int(dt.timestamp() * 1000),
                "open":   float(row["Open"]),
                "high":   float(row["High"]),
                "low":    float(row["Low"]),
                "close":  float(row["Close"]),
                "volume": int(row["Volume"]),
            })
        return bars
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def cached_weekly_bars(ticker: str) -> list:
    """1-year weekly bars from yfinance for the Weekly chart tab."""
    try:
        hist = yf.Ticker(ticker).history(period="1y", interval="1wk")
        if hist.empty:
            return []
        bars = []
        for dt, row in hist.iterrows():
            bars.append({
                "date":   str(dt.date()) if hasattr(dt, "date") else str(dt)[:10],
                "open":   float(row["Open"]),
                "high":   float(row["High"]),
                "low":    float(row["Low"]),
                "close":  float(row["Close"]),
                "volume": int(row["Volume"]),
            })
        return bars
    except Exception:
        return []


# ── Cached fetchers — Screener ────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def cached_screener_movers() -> dict:
    """Single Finviz call for top gainers/losers. TTL=300s."""
    from data.screener import get_top_movers
    return get_top_movers(n=20)


@st.cache_data(ttl=300, show_spinner=False)
def cached_unusual_volume() -> list:
    """Single Finviz call for unusual volume tickers. TTL=300s."""
    from data.screener import get_unusual_volume
    return get_unusual_volume(n=20)


@st.cache_data(ttl=300, show_spinner=False)
def cached_sector_data() -> list:
    """11 sector ETFs via single yfinance batch download. TTL=300s."""
    from data.sector_monitor import get_sector_data
    return get_sector_data()


@st.cache_data(ttl=300, show_spinner=False)
def cached_market_scan() -> list:
    """
    Anomaly scan across the full S&P 500 universe.

    Reuses cached_screener_movers() and cached_unusual_volume() so Finviz
    is never called twice within the same TTL window.
    Uses cached_technicals() + mock-neutral sentiment (no sentiment API calls
    for 100+ tickers — keeps first-run time manageable).
    Returns only Watch-flagged tickers, sorted by anomaly score descending.
    """
    from data.screener import get_stock_universe

    movers  = cached_screener_movers()
    unusual = cached_unusual_volume()

    # Price lookup: movers/unusual rows carry price data; base universe tickers
    # get price from get_technicals() directly (done in the loop below).
    price_map = {
        r["ticker"]: {"price": r.get("price"), "change_pct": r.get("change_pct")}
        for r in (movers.get("gainers", []) + movers.get("losers", []) + unusual)
    }

    # Build deduplicated universe: mover tickers first (most likely to be interesting),
    # then unusual volume, then broad S&P 500 base — capped at 75 tickers.
    mover_tickers   = [r["ticker"] for r in (movers.get("gainers", []) + movers.get("losers", []))]
    unusual_tickers = [r["ticker"] for r in unusual]
    seen: set = set()
    universe: list = []
    for t in mover_tickers + unusual_tickers + get_stock_universe():
        if t not in seen:
            seen.add(t)
            universe.append(t)
        if len(universe) >= 75:
            break

    mock_sent = {"sentiment_label": "Neutral", "overall_sentiment": 0.0}
    results = []
    for ticker in universe:
        try:
            tech = cached_technicals(ticker)   # uses Streamlit cache
            if tech.get("error"):
                continue
            # check_earnings/check_sector disabled — too slow for 75-ticker scan
            anomaly = compute_anomaly(ticker, tech, mock_sent)
            if not anomaly.get("is_watch"):
                continue
            if anomaly.get("quality_score", 0) < 50:
                continue
            pi = price_map.get(ticker, {})
            row = dict(anomaly)
            row["price"]      = pi.get("price") or tech.get("current_price")
            row["change_pct"] = pi.get("change_pct")
            results.append(row)
        except Exception:
            continue

    # Stage 10-12: augment top 15 with fundamentals, top 10 with factor model + catalysts
    # Parallelized — all functions are stateless and thread-safe
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from data.factor_model import compute_factor_model

    def _augment_row(row, idx):
        ticker = row["ticker"]
        try:
            fund = get_fundamentals(ticker)
            scored = score_fundamentals(fund)
            row["fundamental_score"]  = scored["fundamental_score"]
            row["fundamental_signal"] = scored["fundamental_signal"]
        except Exception:
            row["fundamental_score"]  = 0
            row["fundamental_signal"] = "Neutral"
        if idx < 10:
            try:
                fm = compute_factor_model(ticker)
                row["composite_factor_score"] = fm.get("composite_score", 50.0)
                row["factor_signal"]          = fm.get("composite_signal", "Neutral")
                row["pead_candidate"]         = fm.get("pead_candidate", False)
            except Exception:
                row["composite_factor_score"] = 50.0
                row["factor_signal"]          = "Neutral"
                row["pead_candidate"]         = False
            try:
                cat = detect_catalysts(ticker)
                row["catalyst_boost"]     = cat.get("boost", 0)
                row["catalyst_direction"] = cat.get("direction", "Neutral")
                row["catalyst_why"]       = cat.get("why", [])
            except Exception:
                row["catalyst_boost"]     = 0
                row["catalyst_direction"] = "Neutral"
                row["catalyst_why"]       = []
        return row

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_augment_row, row, i) for i, row in enumerate(results[:15])]
        for f in as_completed(futures):
            f.result()  # rows mutated in-place; propagate exceptions if any

    def _composite(r):
        anomaly_norm  = min((r.get("score") or 0) / 6.0, 1.0) * 25
        fund_norm     = ((r.get("fundamental_score") or 0) + 100) / 200 * 22
        qual_norm     = (r.get("quality_score") or 0) / 100 * 18
        factor_norm   = (r.get("composite_factor_score") or 50.0) / 100 * 22
        catalyst_norm = ((r.get("catalyst_boost") or 0) + 100) / 200 * 13
        return anomaly_norm + fund_norm + qual_norm + factor_norm + catalyst_norm

    results.sort(key=_composite, reverse=True)
    return results


# ── Cached fetchers — Portfolio ───────────────────────────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def cached_account() -> dict:
    """Alpaca paper account summary. TTL=30s."""
    return get_account()


@st.cache_data(ttl=30, show_spinner=False)
def cached_positions() -> list:
    """All open Alpaca positions. TTL=30s."""
    return get_positions()


@st.cache_data(ttl=60, show_spinner=False)
def cached_orders(status: str = "all", limit: int = 20) -> list:
    """Recent Alpaca orders. TTL=60s."""
    return get_orders(status=status, limit=limit)


@st.cache_data(ttl=900, show_spinner=False)
def cached_market_regime() -> dict:
    """Market regime (SPY/QQQ/VIX). TTL=15min."""
    return get_market_regime()


# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt_price(val) -> str:
    return "N/A" if val is None else f"${val:,.2f}"

def _fmt_pct(val) -> str:
    if val is None:
        return "N/A"
    sign = "+" if val >= 0 else ""
    return sign + f"{val:.2f}%"

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

def _fmt_float(val, decimals: int = 2, suffix: str = "") -> str:
    if val is None:
        return "N/A"
    return str(round(val, decimals)) + suffix

def _fmt_volume(val) -> str:
    if val is None:
        return "N/A"
    if val >= 1_000_000_000:
        return f"{val/1_000_000_000:.1f}B"
    if val >= 1_000_000:
        return f"{val/1_000_000:.1f}M"
    if val >= 1_000:
        return f"{val/1_000:.0f}K"
    return str(val)

def _score_color(score) -> str:
    if score is None:
        return NEUTRAL
    return GREEN if score >= 0.05 else (RED if score <= -0.05 else NEUTRAL)

def _label_color(label: str) -> str:
    return GREEN if label == "Bullish" else (RED if label == "Bearish" else NEUTRAL)

def _chg_color(val) -> str:
    if val is None:
        return NEUTRAL
    return GREEN if val >= 0 else RED

def _quick_sentiment_label(score) -> str:
    if score is None:
        return "Neutral"
    return "Bullish" if score >= 0.05 else ("Bearish" if score <= -0.05 else "Neutral")

def _signal_color(signal: str) -> str:
    bull = {"Bullish Cross", "Bullish", "Oversold", "Below Lower",
            "Above VWAP", "High Volume", "Elevated", "Above"}
    bear = {"Bearish Cross", "Bearish", "Overbought", "Above Upper",
            "Below VWAP", "Below"}
    if signal in bull:
        return GREEN
    if signal in bear:
        return RED
    return NEUTRAL


# ── HTML helpers ──────────────────────────────────────────────────────────────

def _card(inner_html: str, border_color: str = "#2a2d35") -> str:
    return (
        f'<div style="background:{BG2};border:1px solid {border_color};'
        f'border-radius:6px;padding:12px;margin-bottom:6px">'
        f'{inner_html}</div>'
    )

def _headline_row(title: str, url: str, score, byline: str) -> str:
    c = _score_color(score)
    score_str = "N/A" if score is None else str(round(score, 3))
    safe_title = title[:95]
    return (
        f'<div style="border-left:3px solid {c};padding:4px 10px;margin-bottom:8px">'
        f'<a href="{url}" target="_blank" '
        f'style="color:#fafafa;text-decoration:none;font-size:0.85rem">{safe_title}</a><br>'
        f'<span style="color:{NEUTRAL};font-size:0.72rem">{byline} &nbsp;·&nbsp; {score_str}</span>'
        f'</div>'
    )

def _signal_badge(label: str, value_str: str, color: str) -> str:
    """Colored chip for the TA signal badges row."""
    return (
        f'<div style="display:inline-block;background:{color}18;'
        f'border:1px solid {color};border-radius:5px;'
        f'padding:6px 12px;margin:3px;text-align:center;min-width:90px">'
        f'<div style="color:{NEUTRAL};font-size:0.62rem;letter-spacing:0.5px">{label}</div>'
        f'<div style="color:{color};font-size:0.82rem;font-weight:bold;margin-top:2px">'
        f'{value_str}</div>'
        f'</div>'
    )

def _movers_table_html(rows: list, accent: str, score_map: dict = None) -> str:
    """Pure-HTML table for gainers/losers — no interactive elements needed."""
    score_map = score_map or {}
    html = (
        '<table style="width:100%;border-collapse:collapse;'
        'font-size:0.8rem;font-family:monospace">'
        '<tr style="color:#666;border-bottom:1px solid #2a2d35">'
        '<th style="text-align:left;padding:4px 8px">Ticker</th>'
        '<th style="text-align:right;padding:4px 8px">Price</th>'
        '<th style="text-align:right;padding:4px 8px">Chg%</th>'
        '<th style="text-align:right;padding:4px 8px">Vol</th>'
        '<th style="text-align:right;padding:4px 8px">Score</th>'
        '</tr>'
    )
    for row in rows:
        chg       = row.get("change_pct")
        chg_c     = GREEN if (chg or 0) >= 0 else RED
        chg_str   = f"{chg:+.2f}%" if chg is not None else "N/A"
        price_str = f'${row["price"]:.2f}' if row.get("price") else "N/A"
        vol_str   = _fmt_volume(row.get("volume"))
        qs        = score_map.get(row["ticker"])
        if qs is not None:
            qs_color = GREEN if qs >= 70 else (YELLOW if qs >= 50 else NEUTRAL)
            qs_str   = f'{qs:.0f}'
        else:
            qs_color = NEUTRAL
            qs_str   = '—'
        html += (
            f'<tr style="border-bottom:1px solid #1e2128">'
            f'<td style="padding:4px 8px;color:{accent};font-weight:bold">'
            f'{row["ticker"]}</td>'
            f'<td style="padding:4px 8px;text-align:right">{price_str}</td>'
            f'<td style="padding:4px 8px;text-align:right;color:{chg_c}">{chg_str}</td>'
            f'<td style="padding:4px 8px;text-align:right;color:{NEUTRAL}">{vol_str}</td>'
            f'<td style="padding:4px 8px;text-align:right;color:{qs_color};font-weight:bold">{qs_str}</td>'
            f'</tr>'
        )
    html += '</table>'
    return html


# ── Chart builders ────────────────────────────────────────────────────────────

def _build_intraday_chart(bars: list, title: str) -> go.Figure:
    """Intraday 1-min candlestick + volume."""
    if not bars:
        fig = go.Figure()
        fig.update_layout(title="No intraday data available",
                          paper_bgcolor=BG, plot_bgcolor=BG2,
                          font=dict(color="#fafafa"), height=400)
        return fig

    ts     = [b["timestamp"] for b in bars]
    opens  = [b.get("open",   0) for b in bars]
    highs  = [b.get("high",   0) for b in bars]
    lows   = [b.get("low",    0) for b in bars]
    closes = [b.get("close",  0) for b in bars]
    vols   = [b.get("volume", 0) or 0 for b in bars]

    dts = pd.to_datetime(ts, unit="ms", utc=True)
    try:
        dts = dts.tz_convert("America/New_York")
    except Exception:
        pass

    vol_colors = [GREEN if c >= o else RED for c, o in zip(closes, opens)]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.78, 0.22], vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(
        x=dts, open=opens, high=highs, low=lows, close=closes,
        name="Price", increasing_line_color=GREEN, decreasing_line_color=RED,
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=dts, y=vols, name="Volume",
        marker_color=vol_colors, opacity=0.5, showlegend=False,
    ), row=2, col=1)

    fig.update_layout(
        title=dict(text=title, font=dict(color=NEUTRAL, size=13)),
        paper_bgcolor=BG, plot_bgcolor=BG2,
        font=dict(color="#fafafa"), height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis_rangeslider_visible=False, hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#2a2d35", showgrid=True)
    fig.update_yaxes(gridcolor="#2a2d35", showgrid=True)
    fig.update_yaxes(side="right", row=1, col=1)
    fig.update_yaxes(showticklabels=False, showgrid=False, row=2, col=1)
    return fig


def _build_daily_ta_chart(tech: dict, ticker: str) -> go.Figure:
    """
    3-panel daily chart (90 days):
      Row 1 (60%): Candlestick + SMA20 + SMA50 + Bollinger Bands + S/R lines
      Row 2 (15%): Volume bars
      Row 3 (25%): MACD line + Signal line + Histogram
    """
    bars      = tech.get("daily_bars", [])
    sr_levels = tech.get("support_resistance", [])

    if not bars:
        fig = go.Figure()
        fig.update_layout(title="No daily data available",
                          paper_bgcolor=BG, plot_bgcolor=BG2,
                          font=dict(color="#fafafa"), height=520)
        return fig

    dates  = [b["date"]          for b in bars]
    opens  = [b.get("open")      for b in bars]
    highs  = [b.get("high")      for b in bars]
    lows   = [b.get("low")       for b in bars]
    closes = [b.get("close")     for b in bars]
    vols   = [b.get("volume", 0) or 0 for b in bars]
    sma20s = [b.get("sma20")     for b in bars]
    sma50s = [b.get("sma50")     for b in bars]
    bb_ups = [b.get("bb_upper")  for b in bars]
    bb_mds = [b.get("bb_middle") for b in bars]
    bb_lws = [b.get("bb_lower")  for b in bars]
    macds  = [b.get("macd")         for b in bars]
    msigs  = [b.get("macd_signal")  for b in bars]
    mhists = [b.get("macd_hist")    for b in bars]

    vol_colors  = [GREEN if c is not None and o is not None and c >= o else RED
                   for c, o in zip(closes, opens)]
    hist_colors = [GREEN if (h or 0) >= 0 else RED for h in mhists]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.60, 0.15, 0.25],
        vertical_spacing=0.02,
    )

    # Row 1: Candlestick
    fig.add_trace(go.Candlestick(
        x=dates, open=opens, high=highs, low=lows, close=closes,
        name="Price",
        increasing_line_color=GREEN, decreasing_line_color=RED,
        showlegend=False,
    ), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=dates, y=bb_ups, name="BB Upper",
        line=dict(color="#666688", width=0.7, dash="dot"),
        fill=None, showlegend=True,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=bb_lws, name="BB Lower",
        line=dict(color="#666688", width=0.7, dash="dot"),
        fill="tonexty", fillcolor="rgba(100,100,150,0.08)",
        showlegend=True,
    ), row=1, col=1)

    # SMA lines
    fig.add_trace(go.Scatter(
        x=dates, y=sma20s, name="SMA 20",
        line=dict(color=YELLOW, width=1.3), showlegend=True,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=sma50s, name="SMA 50",
        line=dict(color=BLUE, width=1.3), showlegend=True,
    ), row=1, col=1)

    # Support / Resistance horizontal dashed lines
    for sr in sr_levels:
        fig.add_hline(
            y=sr, row=1, col=1,
            line_dash="dot", line_color="#aaaaaa", line_width=0.8,
            annotation_text=f"  ${sr:.2f}",
            annotation_position="right",
            annotation_font=dict(size=9, color="#aaaaaa"),
        )

    # Row 2: Volume
    fig.add_trace(go.Bar(
        x=dates, y=vols, name="Volume",
        marker_color=vol_colors, opacity=0.5, showlegend=False,
    ), row=2, col=1)

    # Row 3: MACD
    fig.add_trace(go.Scatter(
        x=dates, y=macds, name="MACD",
        line=dict(color=GREEN, width=1.3), showlegend=True,
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=msigs, name="Signal",
        line=dict(color="#ff9900", width=1.3), showlegend=True,
    ), row=3, col=1)
    fig.add_trace(go.Bar(
        x=dates, y=mhists, name="Histogram",
        marker_color=hist_colors, opacity=0.7, showlegend=False,
    ), row=3, col=1)

    fig.update_layout(
        title=dict(text=f"{ticker} — Daily Chart (90 Days)",
                   font=dict(color=NEUTRAL, size=13)),
        paper_bgcolor=BG, plot_bgcolor=BG2,
        font=dict(color="#fafafa"), height=540,
        margin=dict(l=0, r=60, t=40, b=0),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="left", x=0, bgcolor="rgba(0,0,0,0)",
            font=dict(size=10),
        ),
    )
    fig.update_xaxes(gridcolor="#2a2d35", showgrid=True)
    fig.update_yaxes(gridcolor="#2a2d35", showgrid=True)
    fig.update_yaxes(side="right", row=1, col=1)
    fig.update_yaxes(showticklabels=False, showgrid=False, row=2, col=1)
    fig.update_yaxes(title_text="MACD", title_font=dict(size=9),
                     side="right", row=3, col=1)
    return fig


def _build_weekly_chart(bars: list, ticker: str) -> go.Figure:
    """Simple weekly candlestick + volume (1 year)."""
    if not bars:
        fig = go.Figure()
        fig.update_layout(title="No weekly data available",
                          paper_bgcolor=BG, plot_bgcolor=BG2,
                          font=dict(color="#fafafa"), height=400)
        return fig

    dates  = [b["date"]  for b in bars]
    opens  = [b["open"]  for b in bars]
    highs  = [b["high"]  for b in bars]
    lows   = [b["low"]   for b in bars]
    closes = [b["close"] for b in bars]
    vols   = [b["volume"] for b in bars]
    vol_colors = [GREEN if c >= o else RED for c, o in zip(closes, opens)]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.78, 0.22], vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(
        x=dates, open=opens, high=highs, low=lows, close=closes,
        name="Price", increasing_line_color=GREEN, decreasing_line_color=RED,
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=dates, y=vols, marker_color=vol_colors, opacity=0.5, showlegend=False,
    ), row=2, col=1)

    fig.update_layout(
        title=dict(text=f"{ticker} — Weekly Chart (1 Year)",
                   font=dict(color=NEUTRAL, size=13)),
        paper_bgcolor=BG, plot_bgcolor=BG2,
        font=dict(color="#fafafa"), height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis_rangeslider_visible=False, hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#2a2d35", showgrid=True)
    fig.update_yaxes(gridcolor="#2a2d35", showgrid=True)
    fig.update_yaxes(side="right", row=1, col=1)
    fig.update_yaxes(showticklabels=False, showgrid=False, row=2, col=1)
    return fig


def _build_gauge(score, label: str) -> go.Figure:
    val       = score if score is not None else 0.0
    gauge_val = round((val + 1) * 50, 1)
    color     = _label_color(label)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=gauge_val,
        number=dict(font=dict(color=color, size=32), valueformat=".1f"),
        title=dict(text=label, font=dict(color=color, size=14)),
        gauge=dict(
            axis=dict(range=[0, 100],
                      tickvals=[0, 25, 50, 75, 100],
                      ticktext=["Bearish", "", "Neutral", "", "Bullish"],
                      tickcolor="#aaaaaa", tickfont=dict(size=10)),
            bar=dict(color=color, thickness=0.25),
            bgcolor=BG2, borderwidth=1, bordercolor="#333",
            steps=[
                dict(range=[0,  40], color="#2a1515"),
                dict(range=[40, 60], color="#1a1d24"),
                dict(range=[60, 100], color="#152a1e"),
            ],
            threshold=dict(line=dict(color="white", width=2),
                           thickness=0.8, value=gauge_val),
        ),
    ))
    fig.update_layout(paper_bgcolor=BG, font=dict(color="#fafafa"),
                      height=250, margin=dict(l=40, r=40, t=50, b=10))
    return fig


def _build_factor_radar(factor_model_dict: dict) -> go.Figure:
    """
    Plotly polar radar chart showing all 6 factor percentile scores (0-100).
    50 = market average. Green fill for strong factors, neutral otherwise.
    """
    factors = factor_model_dict.get("factors", {})
    labels  = []
    values  = []
    factor_order = [
        ("momentum",          "Momentum"),
        ("earnings_momentum", "Earnings"),
        ("value",             "Value"),
        ("quality",           "Quality"),
        ("short_interest",    "Short Int."),
        ("institutional",     "Inst. Flow"),
        ("dcf",               "DCF Value"),
    ]
    for key, display in factor_order:
        f = factors.get(key, {})
        pct = f.get("percentile", 50.0) if f.get("available") else 50.0
        labels.append(display)
        values.append(round(pct, 1))

    # Close the polygon
    labels_closed = labels + [labels[0]]
    values_closed = values + [values[0]]

    composite = factor_model_dict.get("composite_score", 50.0)
    fill_color = "rgba(0,255,136,0.15)" if composite >= 65 else (
                 "rgba(255,68,68,0.15)"  if composite <= 35 else
                 "rgba(136,136,136,0.1)"
    )
    line_color = GREEN if composite >= 65 else (RED if composite <= 35 else NEUTRAL)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor=fill_color,
        line=dict(color=line_color, width=2),
        name="Factor Score",
        hovertemplate="%{theta}: %{r:.0f}th pct<extra></extra>",
    ))
    # 50th percentile reference ring
    fig.add_trace(go.Scatterpolar(
        r=[50] * (len(labels) + 1),
        theta=labels_closed,
        mode="lines",
        line=dict(color="#333333", width=1, dash="dot"),
        name="Avg (50th)",
        hoverinfo="skip",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor=BG2,
            radialaxis=dict(
                range=[0, 100],
                tickvals=[25, 50, 75],
                ticktext=["25", "50", "75"],
                tickfont=dict(size=8, color=NEUTRAL),
                gridcolor="#2a2d35",
            ),
            angularaxis=dict(
                tickfont=dict(size=10, color="#fafafa"),
                gridcolor="#2a2d35",
            ),
        ),
        paper_bgcolor=BG,
        font=dict(color="#fafafa"),
        showlegend=False,
        height=280,
        margin=dict(l=40, r=40, t=30, b=20),
    )
    return fig


def _build_sector_heatmap(sectors: list) -> go.Figure:
    """
    Horizontal bar chart showing today's % change for all 11 sector ETFs.
    Bars are green (positive) or red (negative).
    The top 2 and bottom 2 sectors are labeled HOT / COLD.
    """
    if not sectors:
        fig = go.Figure()
        fig.update_layout(
            title="No sector data available",
            paper_bgcolor=BG, plot_bgcolor=BG2,
            font=dict(color="#fafafa"), height=300,
        )
        return fig

    # Already sorted best→worst by get_sector_data(); keep that order for the chart
    sorted_s = list(sectors)
    names    = [f'{s["name"]} ({s["ticker"]})' for s in sorted_s]
    values   = [s["change_pct"] or 0            for s in sorted_s]
    colors   = [GREEN if v >= 0 else RED         for v in values]
    text     = [f"{v:+.2f}%" if v != 0 else "N/A" for v in values]

    # Determine HOT / COLD labels (top 2 / bottom 2 valid sectors)
    valid = [s for s in sorted_s if s["change_pct"] is not None]
    hot_tickers  = {s["ticker"] for s in valid[:2]}
    cold_tickers = {s["ticker"] for s in valid[-2:]}

    customdata = []
    for s in sorted_s:
        if s["ticker"] in hot_tickers:
            customdata.append("HOT")
        elif s["ticker"] in cold_tickers:
            customdata.append("COLD")
        else:
            customdata.append("")

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker_color=colors,
        text=text,
        textposition="outside",
        textfont=dict(size=10, color="#cccccc"),
        hovertemplate="%{y}: %{x:+.2f}%<extra></extra>",
        customdata=customdata,
    ))

    # Annotate HOT / COLD labels inside bars
    for i, (s, label) in enumerate(zip(sorted_s, customdata)):
        if not label:
            continue
        emoji = "🔥" if label == "HOT" else "❄️"
        fig.add_annotation(
            x=0,
            y=names[i],
            text=f" {emoji} {label}",
            showarrow=False,
            font=dict(size=11, color=YELLOW if label == "HOT" else BLUE),
            xanchor="left",
            xref="paper",
        )

    fig.update_layout(
        title=dict(text="S&P 500 Sector Performance — Today",
                   font=dict(color=NEUTRAL, size=13)),
        paper_bgcolor=BG,
        plot_bgcolor=BG2,
        font=dict(color="#fafafa"),
        height=380,
        margin=dict(l=0, r=90, t=50, b=10),
        xaxis=dict(
            zeroline=True, zerolinecolor="#555", zerolinewidth=1,
            ticksuffix="%", gridcolor="#2a2d35",
        ),
        yaxis=dict(gridcolor="#2a2d35", tickfont=dict(size=10)),
        showlegend=False,
    )
    return fig


# ── Market View tab content ────────────────────────────────────────────────────

def _render_market_tab(selected: str):
    """Render the full per-ticker market view panel."""

    with st.spinner(f"Loading {selected}..."):
        data = cached_market_data(selected)
        sent = cached_sentiment(selected)
        tech = cached_technicals(selected)

    fund     = data.get("fundamentals") or {}
    price_yf = data.get("price_yf")     or {}
    quote    = data.get("quote")         or {}

    live_price   = data.get("live_price")
    change_pct   = data.get("change_pct")
    price_source = data.get("price_source")   # "alpaca" | "polygon" | "yfinance" | None
    prev_close   = price_yf.get("prev_close") or quote.get("prev_close")
    short_name   = fund.get("short_name") or selected
    sector       = fund.get("sector") or ""
    industry     = fund.get("industry") or ""

    price_str = _fmt_price(live_price)
    chg_str   = _fmt_pct(change_pct)
    prev_str  = _fmt_price(prev_close)
    chg_c     = _chg_color(change_pct)
    sub_str   = " — ".join(filter(None, [short_name, sector, industry]))

    # Data source badge — labels match price_source values in market_data.py
    if price_source == "Alpaca Live":
        src_label = "Alpaca Live"
        src_color = GREEN
    elif price_source == "yfinance Delayed":
        src_label = "yfinance Delayed"
        src_color = YELLOW
    elif price_source == "Polygon":
        src_label = "Polygon"
        src_color = BLUE
    else:
        src_label = "No Price"
        src_color = RED

    # ── PRICE HEADER ──────────────────────────────────────────────────────────
    ph_left, ph_right = st.columns([6, 2])
    with ph_left:
        st.markdown(
            f'<div style="line-height:1.2">'
            f'<span style="font-size:2.2rem;font-weight:bold">{price_str}</span>'
            f'&nbsp;&nbsp;'
            f'<span style="font-size:1.5rem;font-weight:bold;color:{chg_c}">{chg_str}</span>'
            f'&nbsp;&nbsp;'
            f'<span style="font-size:0.72rem;color:{src_color};font-family:monospace;'
            f'border:1px solid {src_color};border-radius:3px;padding:1px 5px">'
            f'{src_label}</span>'
            f'<br>'
            f'<span style="font-size:0.85rem;color:{NEUTRAL}">'
            f'Prev close: {prev_str} &nbsp;|&nbsp; {sub_str}'
            f'</span></div>',
            unsafe_allow_html=True,
        )
    with ph_right:
        st.markdown(
            f'<div style="margin-top:8px;text-align:right">'
            f'<span style="font-size:1.5rem;font-weight:bold;color:{GREEN}">{selected}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── FUNDAMENTALS PANEL ────────────────────────────────────────────────────
    fund2        = cached_fundamentals(selected)
    fund_scored2 = score_fundamentals(fund2)
    fund_score   = fund_scored2["fundamental_score"]
    fund_signal  = fund_scored2["fundamental_signal"]
    fund_reasons = fund_scored2["fundamental_reasons"]
    squeeze_info = get_short_squeeze_score(fund2)

    # Row 1 — Valuation
    st.caption("Valuation")
    r1_cols = st.columns(6)
    r1_data = [
        ("Market Cap",   _fmt_cap(fund2.get("market_cap"))),
        ("P/E (Trail.)", _fmt_float(fund2.get("pe_ratio"),    2, "x")),
        ("P/E (Fwd)",    _fmt_float(fund2.get("forward_pe"),  2, "x")),
        ("PEG",          _fmt_float(fund2.get("peg_ratio"),   2)),
        ("P/B",          _fmt_float(fund2.get("pb_ratio"),    2, "x")),
        ("EV/EBITDA",    _fmt_float(fund2.get("ev_ebitda"),   1, "x")),
    ]
    for i, (lbl, val) in enumerate(r1_data):
        with r1_cols[i]:
            st.metric(label=lbl, value=val)

    # Row 2 — Growth & Health
    st.caption("Growth & Financial Health")
    r2_cols = st.columns(6)
    rev_g  = fund2.get("revenue_growth")
    earn_g = fund2.get("earnings_growth")
    pm     = fund2.get("profit_margins")
    r2_data = [
        ("Revenue Growth",  ("N/A" if rev_g  is None else f"{rev_g*100:+.1f}%")),
        ("Earnings Growth", ("N/A" if earn_g is None else f"{earn_g*100:+.1f}%")),
        ("Profit Margin",   ("N/A" if pm     is None else f"{pm*100:.1f}%")),
        ("ROE",             ("N/A" if fund2.get("roe")           is None else f"{fund2['roe']*100:.1f}%")),
        ("Current Ratio",   _fmt_float(fund2.get("current_ratio"), 2)),
        ("D/E",             _fmt_float(fund2.get("debt_to_equity"), 2)),
    ]
    for i, (lbl, val) in enumerate(r2_data):
        with r2_cols[i]:
            st.metric(label=lbl, value=val)

    # Row 3 — Institutional & Analyst
    st.caption("Institutional & Analyst")
    r3_cols = st.columns(6)
    inst_p     = fund2.get("inst_ownership_pct")
    spf        = fund2.get("short_pct_float")
    beat_rate  = fund2.get("earnings_beat_rate")
    tgt_price  = fund2.get("target_price")
    r3_data = [
        ("Inst. Ownership", ("N/A" if inst_p    is None else f"{inst_p*100:.1f}%")),
        ("Short % Float",   ("N/A" if spf       is None else f"{spf*100:.1f}%" if spf < 1.0 else f"{spf:.1f}%")),
        ("Short Ratio",     _fmt_float(fund2.get("short_ratio"), 1, "d")),
        ("Analyst Rec.",    (fund2.get("recommendation") or "N/A").title()),
        ("Target Price",    _fmt_price(tgt_price)),
        ("EPS Beat Rate",   ("N/A" if beat_rate is None else f"{beat_rate*100:.0f}%")),
    ]
    for i, (lbl, val) in enumerate(r3_data):
        with r3_cols[i]:
            st.metric(label=lbl, value=val)

    # Fundamental score banner
    fscore_color = GREEN if fund_signal == "Bullish" else (RED if fund_signal == "Bearish" else NEUTRAL)
    reasons_str  = " + ".join(fund_reasons) if fund_reasons else "—"
    sign_char    = "+" if fund_score > 0 else ""
    st.markdown(
        f'<div style="background:{BG2};border:1px solid {fscore_color};border-radius:6px;'
        f'padding:8px 14px;margin:6px 0">'
        f'<span style="font-size:1.1rem;font-weight:bold;color:{fscore_color}">'
        f'Fundamental Score: {sign_char}{fund_score} / {fund_signal.upper()}'
        f'</span>'
        f'<span style="color:{NEUTRAL};font-size:0.85rem;margin-left:12px">{reasons_str}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Short squeeze alert
    if squeeze_info.get("is_squeeze_candidate"):
        spf_disp  = fund2.get("short_pct_float")
        spf_pct   = (spf_disp * 100 if spf_disp is not None and spf_disp < 1.0
                     else (spf_disp or 0))
        sr_disp   = fund2.get("short_ratio") or 0
        sq_score  = squeeze_info["squeeze_score"]
        st.warning(
            f"SHORT SQUEEZE CANDIDATE — "
            f"Short Float: {spf_pct:.1f}% | "
            f"Short Ratio: {sr_disp:.1f}d | "
            f"Squeeze Score: {sq_score}"
        )

    st.divider()

    # ── SENTIMENT ─────────────────────────────────────────────────────────────
    st.subheader("Sentiment Analysis")

    overall_score = sent.get("overall_sentiment")
    overall_label = sent.get("sentiment_label", "Unknown")
    yahoo_score   = sent.get("yahoo_score")
    finviz_score  = sent.get("finviz_score")
    trend_dir     = sent.get("google_trend_direction")
    trend_val_v   = sent.get("google_trend_value")

    yahoo_str     = _fmt_float(yahoo_score, 3) if yahoo_score is not None else "N/A"
    finviz_str    = _fmt_float(finviz_score, 3) if finviz_score is not None else "N/A"
    trend_label   = (trend_dir or "N/A").capitalize()
    trend_val_display = "N/A" if trend_val_v is None else str(trend_val_v)
    overall_str   = "N/A" if overall_score is None else _fmt_float(overall_score, 4)

    sent_left, sent_right = st.columns([3, 2])
    with sent_left:
        sc_row = st.columns(3)
        with sc_row[0]:
            st.metric("Yahoo News", yahoo_str)
        with sc_row[1]:
            st.metric("Finviz", finviz_str)
        with sc_row[2]:
            st.metric("G-Trends", trend_val_display, delta=trend_label)
        st.markdown("<br>", unsafe_allow_html=True)
        lc = _label_color(overall_label)
        st.markdown(
            f'<span style="font-size:1.4rem;font-weight:bold;color:{lc}">'
            f'{overall_label}</span>'
            f'&nbsp;&nbsp;<span style="color:{NEUTRAL};font-size:0.9rem">'
            f'Combined: {overall_str}</span>',
            unsafe_allow_html=True,
        )
    with sent_right:
        st.plotly_chart(_build_gauge(overall_score, overall_label),
                        width="stretch",
                        config={"displayModeBar": False})

    st.divider()

    # ── TECHNICAL ANALYSIS ────────────────────────────────────────────────────
    st.subheader("Technical Analysis")

    rsi_val  = tech.get("rsi")
    rsi_sig  = tech.get("rsi_signal", "Neutral")
    macd_sig = tech.get("macd_signal", "Neutral")
    bb_sig   = tech.get("bb_signal", "Inside Bands")
    vwap_sig = tech.get("vwap_signal") or "N/A"
    vol_sig  = tech.get("volume_signal", "Normal")
    vol_rat  = tech.get("volume_ratio")
    vwap_val = tech.get("vwap")

    rsi_display  = f"{_fmt_float(rsi_val, 1)} · {rsi_sig}"
    vwap_display = f"{_fmt_price(vwap_val)} · {vwap_sig}"
    vol_display  = f"{_fmt_float(vol_rat, 2)}x · {vol_sig}"

    badges_html = (
        '<div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:12px">'
        + _signal_badge("RSI (14)",   rsi_display,  _signal_color(rsi_sig))
        + _signal_badge("MACD",       macd_sig,      _signal_color(macd_sig))
        + _signal_badge("Bol. Bands", bb_sig,        _signal_color(bb_sig))
        + _signal_badge("VWAP",       vwap_display,  _signal_color(vwap_sig))
        + _signal_badge("Volume",     vol_display,   _signal_color(vol_sig))
        + _signal_badge("SMA 20",     tech.get("price_vs_sma20", "N/A"),
                        _signal_color(tech.get("price_vs_sma20", "")))
        + _signal_badge("SMA 50",     tech.get("price_vs_sma50", "N/A"),
                        _signal_color(tech.get("price_vs_sma50", "")))
        + '</div>'
    )
    st.markdown(badges_html, unsafe_allow_html=True)

    an_data = cached_anomaly(selected)

    # Earnings proximity warning
    if an_data.get("earnings_proximity"):
        earn_date = an_data.get("earnings_date", "")
        st.warning(
            f"⚠️ Earnings within 5 days"
            + (f" ({earn_date})" if earn_date else "")
            + " — consider avoiding new positions.",
            icon=None,
        )

    if an_data.get("is_watch"):
        direction  = an_data.get("direction", "")
        dir_color  = GREEN if direction == "Long" else (RED if direction == "Short" else NEUTRAL)
        dir_badge  = (
            f'<span style="background:{dir_color}22;border:1px solid {dir_color};'
            f'border-radius:3px;padding:1px 7px;font-size:0.78rem;'
            f'color:{dir_color};font-weight:bold">{direction}</span>&nbsp;'
            if direction else ""
        )
        st.markdown(
            f'<div style="background:#ffd70018;border:1px solid {YELLOW};'
            f'border-radius:5px;padding:8px 14px;margin-bottom:6px;'
            f'font-size:0.85rem;color:{YELLOW}">'
            f'🔥 <b>Watch:</b> {dir_badge}{an_data.get("reason", "")}'
            f'</div>',
            unsafe_allow_html=True,
        )
        q_score = an_data.get("quality_score", 0)
        st.progress(
            q_score / 100,
            text=f"Signal Quality: {q_score}/100",
        )

    # ── FACTOR MODEL (Stage 11) ───────────────────────────────────────────────
    fm_data = an_data.get("factor_model")
    if fm_data and not fm_data.get("error"):
        composite_fs  = an_data.get("composite_factor_score")
        fm_signal     = fm_data.get("composite_signal", "Neutral")
        fm_completeness = fm_data.get("data_completeness", 0.0)

        fm_color = GREEN if fm_signal == "Long" else (RED if fm_signal == "Short" else NEUTRAL)
        sign_c   = "+" if (composite_fs or 50) >= 50 else ""
        st.markdown(
            f'<div style="background:{BG2};border:1px solid {fm_color};border-radius:6px;'
            f'padding:7px 14px;margin:6px 0;display:flex;justify-content:space-between;align-items:center">'
            f'<span style="font-size:1.0rem;font-weight:bold;color:{fm_color}">'
            f'Factor Model: {composite_fs:.0f}/100 · {fm_signal.upper()}'
            f'</span>'
            f'<span style="color:{NEUTRAL};font-size:0.78rem">'
            f'Data completeness: {fm_completeness*100:.0f}%</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # PEAD banner
        if an_data.get("pead_candidate"):
            st.info("POST-EARNINGS DRIFT (PEAD) — Active drift opportunity detected. "
                    "Positive earnings surprise within the drift window.")

        # Radar chart + factor table side by side
        fm_left, fm_right = st.columns([2, 3])
        with fm_left:
            st.plotly_chart(_build_factor_radar(fm_data), width="stretch",
                            config={"displayModeBar": False})

        with fm_right:
            st.caption("Factor Breakdown")
            factors       = fm_data.get("factors", {})
            weights_used  = fm_data.get("weights_used", {})
            factor_labels = [
                ("momentum",          "Price Momentum"),
                ("earnings_momentum", "Earnings Momentum"),
                ("value",             "Value"),
                ("quality",           "Quality"),
                ("short_interest",    "Short Interest"),
                ("institutional",     "Institutional Flow"),
                ("dcf",               "DCF Intrinsic Value"),
            ]
            tbl_rows = []
            for fkey, fname in factor_labels:
                f       = factors.get(fkey, {})
                pct     = f.get("percentile", 50.0) if f.get("available") else None
                wt      = weights_used.get(fkey, 0.0)
                notes   = f.get("notes", "—")
                avail   = f.get("available", False)
                if pct is not None:
                    bar_filled = int(round(pct / 5))   # 0-20 blocks
                    bar_empty  = 20 - bar_filled
                    bar_str    = "▓" * bar_filled + "░" * bar_empty
                    pct_color  = GREEN if pct >= 65 else (RED if pct <= 35 else NEUTRAL)
                    pct_str    = f"{pct:.0f}th"
                else:
                    bar_str   = "░" * 20
                    pct_color = NEUTRAL
                    pct_str   = "N/A"
                tbl_rows.append((fname, pct_str, pct_color, f"{wt*100:.0f}%", bar_str, notes))

            for fname, pct_str, pct_color, wt_str, bar_str, notes in tbl_rows:
                st.markdown(
                    f'<div style="font-family:monospace;font-size:0.76rem;'
                    f'border-bottom:1px solid #1e2128;padding:3px 0">'
                    f'<span style="color:#fafafa;width:150px;display:inline-block">{fname}</span>'
                    f'<span style="color:{pct_color};width:52px;display:inline-block">{pct_str}</span>'
                    f'<span style="color:{NEUTRAL};width:32px;display:inline-block">{wt_str}</span>'
                    f'<span style="color:#555;font-size:0.65rem;margin-left:8px">{bar_str}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── DCF VALUATION SUMMARY ────────────────────────────────────────────────
    if fm_data and not fm_data.get("error"):
        dcf_factor = fm_data.get("factors", {}).get("dcf", {})
        dcf_details = dcf_factor.get("details", {})
        dcf_iv = dcf_details.get("intrinsic_value")
        dcf_price = dcf_details.get("current_price")
        dcf_mos = dcf_details.get("margin_of_safety_pct")
        dcf_sig = dcf_details.get("dcf_signal")
        dcf_scenarios = dcf_details.get("scenarios") or {}

        if dcf_iv is not None and dcf_price is not None:
            dcf_color = GREEN if dcf_sig == "Undervalued" else (RED if dcf_sig == "Overvalued" else NEUTRAL)
            mos_sign = "+" if (dcf_mos or 0) >= 0 else ""

            scenario_str = ""
            if dcf_scenarios:
                bear_v = dcf_scenarios.get("bear", {}).get("intrinsic", "?")
                bull_v = dcf_scenarios.get("bull", {}).get("intrinsic", "?")
                scenario_str = f"  |  Bear: ${bear_v}  —  Bull: ${bull_v}"

            st.markdown(
                f'<div style="background:{BG2};border:1px solid {dcf_color};border-radius:6px;'
                f'padding:7px 14px;margin:6px 0;display:flex;justify-content:space-between;align-items:center">'
                f'<span style="font-size:1.0rem;font-weight:bold;color:{dcf_color}">'
                f'DCF: ${dcf_iv:.2f} vs ${dcf_price:.2f} · {dcf_sig.upper()}'
                f'</span>'
                f'<span style="color:{NEUTRAL};font-size:0.78rem">'
                f'Margin: {mos_sign}{dcf_mos:.1f}%{scenario_str}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── CATALYST DETECTOR (Stage 12) ────────────────────────────────────────────
    cat_data = an_data.get("catalyst_result")
    cat_boost = an_data.get("catalyst_boost", 0)
    cat_why   = an_data.get("catalyst_why", [])
    if cat_data and cat_data.get("catalysts"):
        cat_dir   = cat_data.get("direction", "Neutral")
        cat_color = GREEN if cat_dir == "Bullish" else (RED if cat_dir == "Bearish" else NEUTRAL)
        sign_str  = f"+{cat_boost}" if cat_boost > 0 else str(cat_boost)
        st.markdown(
            f'<div style="background:{BG2};border:1px solid {cat_color};border-radius:6px;'
            f'padding:7px 14px;margin:6px 0;display:flex;justify-content:space-between;align-items:center">'
            f'<span style="font-size:1.0rem;font-weight:bold;color:{cat_color}">'
            f'Catalysts: {cat_dir.upper()} ({sign_str})'
            f'</span>'
            f'<span style="color:{NEUTRAL};font-size:0.78rem">'
            f'{len(cat_data["catalysts"])} active</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        for reason in cat_why:
            bullet_c = GREEN if cat_boost > 0 else (RED if cat_boost < 0 else NEUTRAL)
            st.markdown(
                f'<div style="font-family:monospace;font-size:0.76rem;color:{bullet_c};'
                f'padding:2px 0 2px 12px">* {reason}</div>',
                unsafe_allow_html=True,
            )

    # Tabbed charts
    tab_intra, tab_daily, tab_weekly = st.tabs(
        ["Intraday (1-Min)", "Daily (3 Month + Indicators)", "Weekly (1 Year)"]
    )

    with tab_intra:
        intraday = data.get("intraday_bars") or []
        if len(intraday) >= 10:
            bars      = intraday
            chart_ttl = f"{selected} — 1-Min Intraday (Polygon)"
        else:
            bars      = cached_daily_bars(selected)
            chart_ttl = f"{selected} — Daily 1-Month (yfinance, intraday unavailable)"
        st.plotly_chart(_build_intraday_chart(bars, chart_ttl),
                        width="stretch",
                        config={"displayModeBar": False})

    with tab_daily:
        st.plotly_chart(_build_daily_ta_chart(tech, selected),
                        width="stretch",
                        config={"displayModeBar": False})

        sr    = tech.get("support_resistance", [])
        cur_p = tech.get("current_price") or live_price
        if sr and cur_p:
            st.markdown(
                f'<span style="color:{NEUTRAL};font-size:0.75rem;font-weight:bold">'
                f'SUPPORT / RESISTANCE LEVELS</span>',
                unsafe_allow_html=True,
            )
            sr_html = '<div style="display:flex;gap:10px;margin-top:6px">'
            for lv in sr:
                dist   = round((cur_p - lv) / cur_p * 100, 1)
                sign   = "+" if dist >= 0 else ""
                dist_c = GREEN if dist >= 0 else RED
                sr_html += (
                    f'<div style="background:{BG2};border:1px solid #333;'
                    f'border-radius:4px;padding:6px 14px;text-align:center">'
                    f'<div style="font-size:0.9rem;font-weight:bold">${lv:.2f}</div>'
                    f'<div style="font-size:0.72rem;color:{dist_c}">{sign}{dist}%</div>'
                    f'</div>'
                )
            sr_html += '</div>'
            st.markdown(sr_html, unsafe_allow_html=True)

    with tab_weekly:
        weekly_bars = cached_weekly_bars(selected)
        st.plotly_chart(_build_weekly_chart(weekly_bars, selected),
                        width="stretch",
                        config={"displayModeBar": False})

    st.divider()

    # ── HEADLINES ─────────────────────────────────────────────────────────────
    st.subheader("Latest Headlines")
    hl_left, hl_right = st.columns(2)

    yahoo_hl  = (sent.get("yahoo_headlines")  or [])[:5]
    finviz_hl = (sent.get("finviz_headlines") or [])[:5]

    with hl_left:
        st.markdown(
            f'<span style="color:{NEUTRAL};font-size:0.75rem;font-weight:bold">YAHOO FINANCE</span>',
            unsafe_allow_html=True,
        )
        if yahoo_hl:
            rows_html = ""
            for hl in yahoo_hl:
                rows_html += _headline_row(
                    hl.get("title", ""), hl.get("url", "#") or "#",
                    hl.get("score"), hl.get("publisher", ""),
                )
            st.markdown(rows_html, unsafe_allow_html=True)
        else:
            st.caption("No headlines available.")

    with hl_right:
        st.markdown(
            f'<span style="color:{NEUTRAL};font-size:0.75rem;font-weight:bold">FINVIZ</span>',
            unsafe_allow_html=True,
        )
        if finviz_hl:
            rows_html = ""
            for hl in finviz_hl:
                rows_html += _headline_row(
                    hl.get("title", ""), hl.get("url", "#") or "#",
                    hl.get("score"), hl.get("timestamp", ""),
                )
            st.markdown(rows_html, unsafe_allow_html=True)
        else:
            st.caption("No headlines available.")

    st.divider()

    # ── EARNINGS ──────────────────────────────────────────────────────────────
    earnings = (data.get("earnings") or [])[:4]
    if earnings:
        st.subheader("Earnings")
        earn_cols = st.columns(len(earnings))
        for i, e in enumerate(earnings):
            with earn_cols[i]:
                date_str   = e.get("date", "N/A")
                est        = e.get("eps_estimate")
                actual     = e.get("reported_eps")
                surp       = e.get("surprise_pct")
                est_str    = "N/A" if est    is None else f"${est:.2f}"
                actual_str = "TBD" if actual is None else f"${actual:.2f}"
                surp_str   = "N/A" if surp   is None else f"{surp:+.1f}%"
                card_label = "Upcoming" if actual is None else "Reported"
                surp_c     = NEUTRAL if surp is None else (GREEN if surp > 0 else RED)
                st.markdown(
                    _card(
                        f'<div style="color:{NEUTRAL};font-size:0.7rem">{card_label}</div>'
                        f'<div style="font-size:0.95rem;font-weight:bold;margin:2px 0">{date_str}</div>'
                        f'<div style="font-size:0.8rem">Est EPS: <b>{est_str}</b></div>'
                        f'<div style="font-size:0.8rem">Actual: <b>{actual_str}</b></div>'
                        f'<div style="font-size:0.8rem;color:{surp_c}">Surprise: <b>{surp_str}</b></div>',
                        border_color=surp_c if surp else "#2a2d35",
                    ),
                    unsafe_allow_html=True,
                )


# ── Screener tab content ──────────────────────────────────────────────────────

def _render_screener_tab():
    """Render the Market Screener panel."""

    # ── Header ────────────────────────────────────────────────────────────────
    hdr_col, btn_col = st.columns([6, 1])
    with hdr_col:
        st.markdown(
            f'<span style="font-size:1.1rem;font-weight:bold;color:{GREEN}">'
            f'MARKET SCREENER</span>'
            f'&nbsp;&nbsp;<span style="color:{NEUTRAL};font-size:0.78rem">'
            f'Data cached · refreshes every 5 min'
            f'&nbsp;|&nbsp;{datetime.now().strftime("%H:%M:%S")}</span>',
            unsafe_allow_html=True,
        )
    with btn_col:
        if st.button("Force Refresh", key="screener_force_refresh", width="stretch"):
            cached_screener_movers.clear()
            cached_unusual_volume.clear()
            cached_sector_data.clear()
            cached_market_scan.clear()
            st.rerun()

    # ── TOP MOVERS ────────────────────────────────────────────────────────────
    st.markdown(
        f'<span style="color:{NEUTRAL};font-size:0.75rem;font-weight:bold">'
        f'TODAY\'S TOP MOVERS</span>',
        unsafe_allow_html=True,
    )

    with st.spinner("Loading movers..."):
        movers = cached_screener_movers()

    if movers.get("error"):
        st.caption(f"Note: {movers['error']}")

    mv_left, mv_right = st.columns(2)
    with mv_left:
        st.markdown(
            f'<span style="color:{GREEN};font-size:0.78rem;font-weight:bold">'
            f'TOP GAINERS</span>',
            unsafe_allow_html=True,
        )
        gainers = movers.get("gainers", [])
        if gainers:
            st.markdown(_movers_table_html(gainers, GREEN, st.session_state.get("_scan_score_map", {})), unsafe_allow_html=True)
        else:
            st.caption("No data available.")

    with mv_right:
        st.markdown(
            f'<span style="color:{RED};font-size:0.78rem;font-weight:bold">'
            f'TOP LOSERS</span>',
            unsafe_allow_html=True,
        )
        losers = movers.get("losers", [])
        if losers:
            st.markdown(_movers_table_html(losers, RED, st.session_state.get("_scan_score_map", {})), unsafe_allow_html=True)
        else:
            st.caption("No data available.")

    st.divider()

    # ── SECTOR HEATMAP ────────────────────────────────────────────────────────
    st.markdown(
        f'<span style="color:{NEUTRAL};font-size:0.75rem;font-weight:bold">'
        f'SECTOR PERFORMANCE</span>',
        unsafe_allow_html=True,
    )
    with st.spinner("Loading sectors..."):
        sectors = cached_sector_data()
    st.plotly_chart(_build_sector_heatmap(sectors), width="stretch",
                    config={"displayModeBar": False})

    st.divider()

    # ── ANOMALY SCAN ──────────────────────────────────────────────────────────
    st.markdown(
        f'<span style="color:{YELLOW};font-size:0.75rem;font-weight:bold">'
        f'ANOMALY SCAN — TOP SETUPS (technical signals only)</span>',
        unsafe_allow_html=True,
    )

    _flt_left, _flt_right = st.columns([2, 3])
    with _flt_left:
        scan_dir_filter = st.radio(
            "Direction", ["Both", "Long Only", "Short Only"],
            horizontal=True, label_visibility="collapsed",
        )
    with _flt_right:
        scan_min_score = st.slider("Min Quality Score", 0, 100, 40, step=5)

    with st.spinner(
        "Running anomaly scan across 75 stocks "
        "(first run may take 1–2 min while technicals load)..."
    ):
        scan_results = cached_market_scan()

    # Populate score map for movers table (available on next render)
    st.session_state._scan_score_map = {
        r["ticker"]: r.get("quality_score", 0) for r in scan_results
    } if scan_results else {}

    # Apply client-side filters
    if scan_results:
        if scan_dir_filter == "Long Only":
            scan_results = [r for r in scan_results if r.get("direction") == "Long"]
        elif scan_dir_filter == "Short Only":
            scan_results = [r for r in scan_results if r.get("direction") == "Short"]
        scan_results = [r for r in scan_results if r.get("quality_score", 0) >= scan_min_score]

    if not scan_results:
        st.caption(
            "No Watch-flagged tickers detected in the current scan universe."
        )
    else:
        watchlist = load_watchlist()

        # Auto-log each Watch signal once per ticker per day (dedup via session state)
        today_str = datetime.now().strftime("%Y-%m-%d")
        for _sr in scan_results:
            _key = f"{_sr['ticker']}_{today_str}"
            if _key not in st.session_state.logged_signals_today:
                try:
                    log_signal(
                        ticker            = _sr["ticker"],
                        anomaly_score     = _sr.get("score", 0),
                        signals_triggered = _sr.get("signals", []),
                        reason            = _sr.get("reason", ""),
                        price_at_signal   = _sr.get("price") or 0.0,
                    )
                    st.session_state.logged_signals_today.add(_key)
                except Exception:
                    pass

        # Column header strip (HTML — no buttons)
        st.markdown(
            '<div style="display:grid;'
            'grid-template-columns:85px 75px 70px 55px 55px 60px 60px 1fr 88px;'
            'gap:4px;padding:5px 0 3px 0;'
            'font-size:0.7rem;color:#555;font-weight:bold;'
            'border-bottom:1px solid #2a2d35;font-family:monospace">'
            '<span>TICKER</span>'
            '<span>PRICE</span>'
            '<span>CHG%</span>'
            '<span>SCORE</span>'
            '<span>DIR</span>'
            '<span>FACTOR</span>'
            '<span>CAT</span>'
            '<span>SIGNALS / REASON</span>'
            '<span></span>'
            '</div>',
            unsafe_allow_html=True,
        )

        for row in scan_results:
            ticker    = row["ticker"]
            score     = row.get("score", 0)
            reason    = row.get("reason", "—")
            price     = row.get("price")
            chg       = row.get("change_pct")
            direction = row.get("direction", "")
            in_wl     = ticker in watchlist
            fm_score  = row.get("composite_factor_score")
            fm_sig    = row.get("factor_signal", "")
            pead      = row.get("pead_candidate", False)
            cat_boost = row.get("catalyst_boost", 0)
            cat_dir_r = row.get("catalyst_direction", "Neutral")

            dir_color = GREEN if direction == "Long" else (RED if direction == "Short" else NEUTRAL)
            accent    = dir_color
            score_c   = GREEN if score >= 5 else (YELLOW if score >= 4 else NEUTRAL)
            chg_c     = GREEN if (chg or 0) >= 0 else RED
            price_str = f"${price:.2f}" if price else "N/A"
            chg_str   = f"{chg:+.2f}%" if chg is not None else "N/A"
            fm_color  = (GREEN if fm_sig == "Long" else (RED if fm_sig == "Short" else NEUTRAL))
            fm_str    = f"{fm_score:.0f}" if fm_score is not None else "—"
            cat_color = GREEN if cat_boost > 0 else (RED if cat_boost < 0 else NEUTRAL)
            cat_str   = f"{cat_boost:+d}" if cat_boost != 0 else "—"

            ticker_display = ticker + (" 🔥" if row.get("is_watch") else "")
            if pead:
                ticker_display += " 📈"

            # One st.columns() strip per row — required for st.button in last column
            c1, c2, c3, c4, c4b, c4c, c4d, c5, c6 = st.columns([1.1, 1, 0.9, 0.7, 0.7, 0.75, 0.75, 3.5, 1.1])

            with c1:
                st.markdown(
                    f'<span style="font-weight:bold;color:{accent};'
                    f'font-family:monospace;font-size:0.9rem">{ticker_display}</span>',
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f'<span style="font-size:0.82rem">{price_str}</span>',
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f'<span style="color:{chg_c};font-size:0.82rem">{chg_str}</span>',
                    unsafe_allow_html=True,
                )
            with c4:
                st.markdown(
                    f'<span style="color:{score_c};font-weight:bold;'
                    f'font-size:0.9rem">{score}</span>',
                    unsafe_allow_html=True,
                )
            with c4b:
                st.markdown(
                    f'<span style="color:{dir_color};font-weight:bold;'
                    f'font-size:0.82rem">{direction}</span>',
                    unsafe_allow_html=True,
                )
            with c4c:
                st.markdown(
                    f'<span style="color:{fm_color};font-size:0.82rem">{fm_str}</span>',
                    unsafe_allow_html=True,
                )
            with c4d:
                st.markdown(
                    f'<span style="color:{cat_color};font-size:0.82rem;font-weight:bold">{cat_str}</span>',
                    unsafe_allow_html=True,
                )
            with c5:
                with st.expander(f"{reason[:70]}{'...' if len(reason) > 70 else ''}"):
                    # Individual signals
                    for sig in row.get("signals", []):
                        sig_color = GREEN if direction == "Long" else (RED if direction == "Short" else NEUTRAL)
                        st.markdown(
                            f'<span style="color:{sig_color};font-size:0.76rem">• {sig}</span>',
                            unsafe_allow_html=True,
                        )
                    # Factor + Catalyst details
                    detail_parts = []
                    if fm_score is not None:
                        detail_parts.append(f"Factor: {fm_score:.0f}/100")
                    if cat_boost != 0:
                        detail_parts.append(f"Catalyst: {cat_boost:+d}")
                        cat_why = row.get("catalyst_why", [])
                        if cat_why:
                            detail_parts.append(" | ".join(cat_why[:2]))
                    if detail_parts:
                        st.markdown(
                            f'<span style="color:{NEUTRAL};font-size:0.72rem">'
                            f'{"  ·  ".join(detail_parts)}</span>',
                            unsafe_allow_html=True,
                        )
                    # Coach verdict (derived from existing data)
                    q = row.get("quality_score", 0)
                    s = row.get("score", 0)
                    if q >= 75 and s >= 5:
                        verdict, grade = "Trade", "A"
                    elif q >= 65 and s >= 4:
                        verdict, grade = "Trade", "B"
                    elif q >= 55 and s >= 3:
                        verdict, grade = "Caution", "C"
                    else:
                        verdict, grade = "Skip", "D"
                    v_color = GREEN if verdict == "Trade" else (YELLOW if verdict == "Caution" else NEUTRAL)
                    st.markdown(
                        f'<span style="color:{v_color};font-weight:bold;font-size:0.78rem">'
                        f'{verdict} — Grade {grade} ({q})</span>',
                        unsafe_allow_html=True,
                    )
            with c6:
                if in_wl:
                    st.markdown(
                        f'<span style="color:{NEUTRAL};font-size:0.72rem">In WL ✓</span>',
                        unsafe_allow_html=True,
                    )
                else:
                    if st.button("+ Watch", key=f"scan_add_{ticker}"):
                        add_ticker(ticker)
                        st.session_state.selected_ticker = ticker
                        st.rerun()

        # PEAD section — show any PEAD candidates from the scan
        pead_rows = [r for r in scan_results if r.get("pead_candidate")]
        if pead_rows:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f'<span style="color:{GREEN};font-size:0.75rem;font-weight:bold">'
                f'📈 PEAD CANDIDATES — Post-Earnings Drift Active</span>',
                unsafe_allow_html=True,
            )
            for pr in pead_rows:
                st.markdown(
                    f'<div style="background:#00ff8812;border:1px solid {GREEN};'
                    f'border-radius:4px;padding:5px 12px;margin:2px 0;font-size:0.78rem">'
                    f'<b style="color:{GREEN}">{pr["ticker"]}</b>'
                    f'&nbsp;— Factor: {pr.get("composite_factor_score", "—"):.0f}/100'
                    f'&nbsp;|&nbsp;{pr.get("reason", "")}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Catalyst section — show tickers with active catalysts from the scan
        cat_rows = [r for r in scan_results if r.get("catalyst_boost", 0) != 0]
        if cat_rows:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f'<span style="color:{YELLOW};font-size:0.75rem;font-weight:bold">'
                f'ACTIVE CATALYSTS DETECTED</span>',
                unsafe_allow_html=True,
            )
            for cr in cat_rows:
                cb = cr.get("catalyst_boost", 0)
                cd = cr.get("catalyst_direction", "Neutral")
                cc = GREEN if cb > 0 else (RED if cb < 0 else NEUTRAL)
                why_str = " | ".join(cr.get("catalyst_why", [])[:2]) or "—"
                st.markdown(
                    f'<div style="background:{cc}12;border:1px solid {cc};'
                    f'border-radius:4px;padding:5px 12px;margin:2px 0;font-size:0.78rem">'
                    f'<b style="color:{cc}">{cr["ticker"]}</b>'
                    f'&nbsp;— {cd} ({cb:+d})'
                    f'&nbsp;|&nbsp;{why_str}'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ── Whale Flow + Trade Coach Tab ──────────────────────────────────────────────

def _render_whale_tab(selected: str):
    """Render the Whale Flow & Trade Coach panel."""

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        f'<span style="font-size:1.1rem;font-weight:bold;color:{GREEN}">'
        f'Whale Flow & Trade Coach</span>',
        unsafe_allow_html=True,
    )
    st.caption("Track unusual options activity, big money flow, and get coached on setups.")

    # ── Settings row ──────────────────────────────────────────────────────────
    # Sync whale ticker input with globally selected ticker
    selected = st.session_state.selected_ticker
    if st.session_state.get("_last_synced_ticker") != selected:
        st.session_state["whale_ticker_input"] = selected
        st.session_state["_last_synced_ticker"] = selected

    set_c1, set_c2, set_c3, set_c4 = st.columns([2, 1.5, 1.5, 1])
    with set_c1:
        whale_ticker = st.text_input("Ticker", value=selected, key="whale_ticker_input")
        whale_ticker = whale_ticker.upper().strip() if whale_ticker else selected
    with set_c2:
        account_val = st.number_input("Account Value ($)", value=100_000, step=5000, key="whale_acct")
    with set_c3:
        risk_pct = st.number_input("Risk per Trade (%)", value=1.0, step=0.25, min_value=0.25,
                                    max_value=5.0, key="whale_risk")
    with set_c4:
        st.markdown("<br>", unsafe_allow_html=True)
        scan_clicked = st.button("Analyze", key="whale_analyze_btn", type="primary")

    if not scan_clicked and whale_ticker == selected:
        # Auto-analyze selected ticker
        pass

    # ── Whale Detection ───────────────────────────────────────────────────────
    with st.spinner(f"Scanning {whale_ticker} for whale activity..."):
        whale_data = cached_whales(whale_ticker)

    alert_level = whale_data.get("alert_level", "None")
    whale_score = whale_data.get("whale_score", 0)
    whale_dir = whale_data.get("whale_direction", "Neutral")

    # Alert banner
    if alert_level == "Whale Alert":
        alert_color = GREEN if whale_dir == "Bullish" else RED
        st.markdown(
            f'<div style="background:{alert_color}18;border:2px solid {alert_color};'
            f'border-radius:8px;padding:12px 18px;margin:8px 0;text-align:center">'
            f'<span style="font-size:1.3rem;font-weight:bold;color:{alert_color}">'
            f'WHALE ALERT — {whale_dir.upper()} (Score: {whale_score:+d})'
            f'</span></div>',
            unsafe_allow_html=True,
        )
        # Auto-log whale alert
        _wkey = f"whale_{whale_ticker}_{datetime.now().strftime('%Y-%m-%d')}"
        if _wkey not in st.session_state.logged_alerts_today:
            log_alert(whale_ticker, "whale",
                      f"WHALE ALERT {whale_ticker} — {whale_dir} (Score: {whale_score:+d})",
                      score=abs(whale_score))
            st.session_state.logged_alerts_today.add(_wkey)
    elif alert_level == "Alert":
        alert_color = YELLOW
        st.markdown(
            f'<div style="background:{alert_color}18;border:1px solid {alert_color};'
            f'border-radius:6px;padding:10px 16px;margin:8px 0">'
            f'<span style="font-size:1.1rem;font-weight:bold;color:{alert_color}">'
            f'Alert: {whale_dir} unusual activity (Score: {whale_score:+d})'
            f'</span></div>',
            unsafe_allow_html=True,
        )

    # Squeeze alerts
    squeeze = whale_data.get("squeeze", {})
    gamma_risk = squeeze.get("gamma_squeeze_risk", "None")
    short_risk = squeeze.get("short_squeeze_risk", "None")

    if gamma_risk in ("High", "Moderate") or short_risk in ("High", "Moderate"):
        # Auto-log squeeze alerts
        _sqkey = f"squeeze_{whale_ticker}_{datetime.now().strftime('%Y-%m-%d')}"
        if _sqkey not in st.session_state.logged_alerts_today:
            parts = []
            if gamma_risk in ("High", "Moderate"):
                parts.append(f"Gamma {gamma_risk} ({squeeze.get('gamma_score', 0)})")
            if short_risk in ("High", "Moderate"):
                parts.append(f"Short {short_risk} ({squeeze.get('short_score', 0)})")
            log_alert(whale_ticker, "squeeze",
                      f"{whale_ticker} squeeze detected: {', '.join(parts)}",
                      score=max(squeeze.get("gamma_score", 0), squeeze.get("short_score", 0)))
            st.session_state.logged_alerts_today.add(_sqkey)

        sq_cols = st.columns(2)
        if gamma_risk in ("High", "Moderate"):
            with sq_cols[0]:
                gc = RED if gamma_risk == "High" else YELLOW
                st.markdown(
                    f'<div style="background:{gc}18;border:1px solid {gc};'
                    f'border-radius:6px;padding:8px 14px;margin:4px 0">'
                    f'<span style="font-size:0.95rem;font-weight:bold;color:{gc}">'
                    f'GAMMA SQUEEZE — {gamma_risk.upper()} RISK (Score: {squeeze.get("gamma_score", 0)})</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        if short_risk in ("High", "Moderate"):
            col_idx = 1 if gamma_risk in ("High", "Moderate") else 0
            with sq_cols[col_idx]:
                sc = RED if short_risk == "High" else YELLOW
                st.markdown(
                    f'<div style="background:{sc}18;border:1px solid {sc};'
                    f'border-radius:6px;padding:8px 14px;margin:4px 0">'
                    f'<span style="font-size:0.95rem;font-weight:bold;color:{sc}">'
                    f'SHORT SQUEEZE — {short_risk.upper()} RISK (Score: {squeeze.get("short_score", 0)})</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # Two columns: Options Flow | Volume & Institutional
    wc1, wc2 = st.columns(2)

    with wc1:
        st.markdown(
            f'<span style="font-size:0.9rem;font-weight:bold;color:{BLUE}">OPTIONS FLOW</span>',
            unsafe_allow_html=True,
        )
        opt = whale_data.get("options_flow", {})
        call_vol = opt.get("total_call_volume", 0)
        put_vol = opt.get("total_put_volume", 0)
        pc_ratio = opt.get("put_call_ratio")
        pc_sig = opt.get("pc_signal", "Neutral")
        call_prem = opt.get("total_call_premium", 0)
        put_prem = opt.get("total_put_premium", 0)

        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.metric("Call Volume", f"{call_vol:,}")
        with mc2:
            st.metric("Put Volume", f"{put_vol:,}")
        with mc3:
            pc_str = f"{pc_ratio:.2f}" if pc_ratio is not None else "N/A"
            st.metric("P/C Ratio", pc_str)

        # Premium flow
        mc4, mc5 = st.columns(2)
        with mc4:
            st.metric("Call Premium", f"${call_prem:,.0f}")
        with mc5:
            st.metric("Put Premium", f"${put_prem:,.0f}")

        # Unusual contracts table
        unusual = opt.get("unusual_contracts", [])
        if unusual:
            st.markdown(
                f'<span style="font-size:0.78rem;font-weight:bold;color:{YELLOW}">'
                f'UNUSUAL CONTRACTS ({len(unusual)})</span>',
                unsafe_allow_html=True,
            )
            for uc in unusual[:8]:
                side_c = GREEN if uc["side"] == "CALL" else RED
                sweep_tag = " [SWEEP]" if uc.get("is_sweep") else ""
                vol_oi = f"Vol/OI: {uc['vol_oi_ratio']}x" if uc.get("vol_oi_ratio") else "New OI"
                st.markdown(
                    f'<div style="background:{BG2};border-left:3px solid {side_c};'
                    f'padding:4px 10px;margin:2px 0;font-family:monospace;font-size:0.75rem">'
                    f'<b style="color:{side_c}">{uc["side"]}</b> '
                    f'${uc["strike"]:.0f} exp {uc["expiration"]} '
                    f'— Vol: {uc["volume"]:,} | {vol_oi} '
                    f'| ${uc["dollar_value"]:,} '
                    f'| IV: {uc["iv"]}%'
                    f'<span style="color:{YELLOW}">{sweep_tag}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No unusual options activity detected.")

    with wc2:
        st.markdown(
            f'<span style="font-size:0.9rem;font-weight:bold;color:{BLUE}">VOLUME & FLOW</span>',
            unsafe_allow_html=True,
        )
        vol = whale_data.get("volume", {})
        inst = whale_data.get("institutional", {})

        vc1, vc2, vc3 = st.columns(3)
        with vc1:
            vr = vol.get("volume_ratio", 1.0)
            vr_color = GREEN if vr >= 2 else (YELLOW if vr >= 1.5 else NEUTRAL)
            st.metric("Vol Ratio", f"{vr:.1f}x")
        with vc2:
            st.metric("Signal", vol.get("volume_signal", "Normal"))
        with vc3:
            st.metric("Pattern", vol.get("accumulation", "Neutral"))

        ic1, ic2 = st.columns(2)
        with ic1:
            inst_pct = inst.get("institutional_pct")
            st.metric("Inst. Own %", f"{inst_pct:.0f}%" if inst_pct else "N/A")
        with ic2:
            st.metric("Inst. Activity", inst.get("recent_institutional_change", "Unknown"))

        if vol.get("block_detected"):
            st.markdown(
                f'<div style="background:{YELLOW}18;border:1px solid {YELLOW};'
                f'border-radius:4px;padding:6px 12px;margin:4px 0;font-size:0.8rem;'
                f'color:{YELLOW}">Block trade detected in recent sessions</div>',
                unsafe_allow_html=True,
            )

        # Signal summary
        whale_sigs = whale_data.get("whale_signals", [])
        if whale_sigs:
            st.markdown(
                f'<span style="font-size:0.78rem;font-weight:bold;color:{YELLOW}">SIGNALS</span>',
                unsafe_allow_html=True,
            )
            for ws in whale_sigs:
                sig_c = GREEN if "bullish" in ws.lower() or "accumulation" in ws.lower() \
                    else (RED if "bearish" in ws.lower() or "distribution" in ws.lower() else NEUTRAL)
                st.markdown(
                    f'<div style="font-family:monospace;font-size:0.74rem;color:{sig_c};'
                    f'padding:1px 0">* {ws}</div>',
                    unsafe_allow_html=True,
                )

    # ── Trade Coach ───────────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        f'<span style="font-size:1.0rem;font-weight:bold;color:{GREEN}">TRADE COACH</span>',
        unsafe_allow_html=True,
    )

    with st.spinner(f"Analyzing trade setup for {whale_ticker}..."):
        coach = cached_trade_coach(whale_ticker, account_value=account_val, risk_pct=risk_pct)

    verdict = coach.get("verdict", "NO TRADE")
    direction = coach.get("direction", "Neutral")
    grade = coach.get("grade", "F")
    confidence = coach.get("confidence", 0)

    # Verdict banner
    if "TAKE THE TRADE" in verdict:
        v_color = GREEN
    elif "CAUTION" in verdict:
        v_color = YELLOW
    elif "RISKY" in verdict:
        v_color = RED
    else:
        v_color = NEUTRAL

    dir_word = "LONG" if direction == "Long" else ("SHORT" if direction == "Short" else "NO TRADE")
    st.markdown(
        f'<div style="background:{v_color}18;border:2px solid {v_color};'
        f'border-radius:8px;padding:14px 20px;margin:8px 0;'
        f'display:flex;justify-content:space-between;align-items:center">'
        f'<span style="font-size:1.2rem;font-weight:bold;color:{v_color}">'
        f'{verdict}</span>'
        f'<span style="font-size:1.0rem;color:{v_color}">'
        f'{dir_word} | Grade: {grade} | Confidence: {confidence}/100</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Trade levels — only show if there's a trade
    if coach.get("entry") is not None:
        tc1, tc2, tc3, tc4, tc5 = st.columns(5)
        with tc1:
            st.metric("Entry", f"${coach['entry']:.2f}")
        with tc2:
            st.metric("Stop Loss", f"${coach['stop_loss']:.2f}",
                       delta=f"-${coach['risk_per_share']:.2f}")
        with tc3:
            st.metric("Take Profit", f"${coach['take_profit']:.2f}",
                       delta=f"+${coach['reward_per_share']:.2f}")
        with tc4:
            rr = coach.get("risk_reward", 0)
            rr_color = "normal" if rr >= 2 else ("off" if rr >= 1.5 else "off")
            st.metric("Risk/Reward", f"1:{rr:.1f}")
        with tc5:
            st.metric("Shares", f"{coach['position_size']}")

        tc6, tc7 = st.columns(2)
        with tc6:
            st.metric("If Wrong (Loss)", f"-${coach['dollar_risk']:,.0f}")
        with tc7:
            potential_gain = coach["position_size"] * coach["reward_per_share"]
            st.metric("If Right (Gain)", f"+${potential_gain:,.0f}")

        # ── Execute Trade Button ──────────────────────────────────────────────
        side = "buy" if direction == "Long" else "sell"
        btn_label = f"Execute {side.upper()} {whale_ticker} — {coach['position_size']} shares @ ${coach['entry']:.2f}"
        exec_key = f"exec_coach_{whale_ticker}"
        if st.button(btn_label, key=exec_key, type="primary"):
            with st.spinner("Placing bracket order..."):
                result = place_bracket_order(
                    ticker=whale_ticker,
                    qty=coach["position_size"],
                    side=side,
                    limit_price=coach["entry"],
                    stop_loss=coach["stop_loss"],
                    take_profit=coach["take_profit"],
                )
            if result.get("error"):
                st.error(f"Order failed: {result['error']}")
            else:
                st.success(
                    f"Bracket order placed! {result['status'].upper()} — "
                    f"Entry ${coach['entry']:.2f} | SL ${coach['stop_loss']:.2f} | "
                    f"TP ${coach['take_profit']:.2f} | {coach['position_size']} shares"
                )
                log_trade(whale_ticker, coach["entry"], coach["position_size"], side,
                          strategy_notes=f"Trade coach bracket. SL=${coach['stop_loss']:.2f} TP=${coach['take_profit']:.2f}")
                if result.get("legs"):
                    for leg in result["legs"]:
                        st.caption(f"  Leg {leg['id']}: {leg['type']} {leg['side']} — {leg['status']}")

    # Coaching text
    coaching_lines = coach.get("coaching", [])
    if coaching_lines:
        coach_text = "\n".join(coaching_lines)
        st.code(coach_text, language=None)

    # Signals breakdown
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown(
            f'<span style="font-size:0.82rem;font-weight:bold;color:{GREEN}">SIGNALS FOR</span>',
            unsafe_allow_html=True,
        )
        for sf in coach.get("signals_for", []):
            st.markdown(
                f'<div style="font-size:0.74rem;color:{GREEN};padding:1px 0;'
                f'font-family:monospace">+ {sf}</div>',
                unsafe_allow_html=True,
            )
    with sc2:
        st.markdown(
            f'<span style="font-size:0.82rem;font-weight:bold;color:{RED}">SIGNALS AGAINST</span>',
            unsafe_allow_html=True,
        )
        for sa in coach.get("signals_against", []):
            st.markdown(
                f'<div style="font-size:0.74rem;color:{RED};padding:1px 0;'
                f'font-family:monospace">- {sa}</div>',
                unsafe_allow_html=True,
            )

    # ── News & Catalysts ─────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        f'<span style="font-size:1.0rem;font-weight:bold;color:{YELLOW}">NEWS & CATALYSTS — {whale_ticker}</span>',
        unsafe_allow_html=True,
    )
    news_c1, news_c2 = st.columns(2)
    with news_c1:
        try:
            _wt_sent = cached_sentiment(whale_ticker)
            _wt_yhl = (_wt_sent.get("yahoo_headlines") or [])[:5]
            if _wt_yhl:
                for hl in _wt_yhl:
                    sc_val = hl.get("score")
                    sc_c = GREEN if sc_val and sc_val > 0.1 else (RED if sc_val and sc_val < -0.1 else NEUTRAL)
                    sc_str = f" ({sc_val:+.2f})" if sc_val is not None else ""
                    title = hl.get("title", "")[:70]
                    url = hl.get("url", "#") or "#"
                    st.markdown(
                        f'<div style="font-size:0.72rem;padding:2px 0">'
                        f'<a href="{url}" target="_blank" style="color:#ccc;text-decoration:none">'
                        f'{title}</a>'
                        f'<span style="color:{sc_c};font-size:0.65rem">{sc_str}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No Yahoo headlines.")
        except Exception:
            st.caption("Headlines unavailable.")
    with news_c2:
        try:
            _wt_cats = cached_catalysts(whale_ticker)
            if _wt_cats and _wt_cats.get("catalysts"):
                for cat in _wt_cats["catalysts"][:5]:
                    cat_type = cat.get("type", "")
                    cat_desc = cat.get("description", "")[:60]
                    cat_date = cat.get("date", "")
                    cat_c = YELLOW if "earnings" in cat_type.lower() else GREEN
                    st.markdown(
                        f'<div style="font-size:0.72rem;padding:2px 0">'
                        f'<span style="color:{cat_c};font-weight:bold">[{cat_type}]</span> '
                        f'<span style="color:#ccc">{cat_desc}</span>'
                        f'{" — " + cat_date if cat_date else ""}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No upcoming catalysts detected.")
        except Exception:
            st.caption("Catalyst data unavailable.")

    # ── Best Trades Now ─────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        f'<span style="font-size:1.0rem;font-weight:bold;color:{GREEN}">BEST TRADES RIGHT NOW</span>',
        unsafe_allow_html=True,
    )
    st.caption("Scanning watchlist + top movers for actionable setups...")

    watchlist = load_watchlist()
    # Add top movers to scan pool
    try:
        movers = cached_screener_movers()
        mover_tickers = [r["ticker"] for r in (movers.get("gainers", [])[:5] + movers.get("losers", [])[:5])]
    except Exception:
        mover_tickers = []
    scan_pool = list(dict.fromkeys(watchlist + mover_tickers))  # dedupe, preserve order

    best_trades = []
    squeeze_alerts = []
    for st_ticker in scan_pool:
        try:
            coach_result = cached_trade_coach(st_ticker, account_value=account_val, risk_pct=risk_pct)
            whale_result = cached_whales(st_ticker)
            coach_result["_whale"] = whale_result

            # Collect squeeze alerts
            sq = whale_result.get("squeeze", {})
            if sq.get("gamma_squeeze_risk") in ("High", "Moderate"):
                squeeze_alerts.append({
                    "ticker": st_ticker, "type": "GAMMA",
                    "risk": sq["gamma_squeeze_risk"], "score": sq["gamma_score"],
                    "signals": sq.get("squeeze_signals", []),
                })
            if sq.get("short_squeeze_risk") in ("High", "Moderate"):
                squeeze_alerts.append({
                    "ticker": st_ticker, "type": "SHORT",
                    "risk": sq["short_squeeze_risk"], "score": sq["short_score"],
                    "signals": sq.get("squeeze_signals", []),
                })

            if coach_result.get("verdict") != "NO TRADE" and coach_result.get("entry") and coach_result.get("direction") == "Long":
                best_trades.append(coach_result)
        except Exception:
            continue

    # Sort by grade then confidence
    grade_order = {"A": 0, "B": 1, "C": 2, "D": 3, "F": 4}
    best_trades.sort(key=lambda x: (grade_order.get(x["grade"], 4), -x["confidence"]))

    # Squeeze alerts section
    if squeeze_alerts:
        st.markdown(
            f'<span style="font-size:0.85rem;font-weight:bold;color:{RED}">SQUEEZE ALERTS</span>',
            unsafe_allow_html=True,
        )
        for sa in squeeze_alerts:
            sa_c = RED if sa["risk"] == "High" else YELLOW
            top_reason = sa["signals"][0] if sa["signals"] else ""
            st.markdown(
                f'<div style="background:{sa_c}15;border-left:3px solid {sa_c};'
                f'padding:5px 12px;margin:2px 0;font-family:monospace;font-size:0.78rem">'
                f'<b style="color:{sa_c}">{sa["ticker"]}</b> — '
                f'{sa["type"]} SQUEEZE {sa["risk"].upper()} (Score: {sa["score"]})'
                f'&nbsp;| {top_reason}'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown("<br>", unsafe_allow_html=True)

    if best_trades:
        for bt in best_trades[:5]:
            dir_word = "LONG" if bt["direction"] == "Long" else "SHORT"
            bt_color = GREEN if bt["grade"] in ("A", "B") else (YELLOW if bt["grade"] == "C" else RED)
            whale_info = bt.get("_whale", {})
            whale_badge = ""
            if whale_info.get("alert_level") in ("Whale Alert", "Alert"):
                whale_badge = f' | Whale: {whale_info["whale_direction"]} ({whale_info["whale_score"]:+d})'

            st.markdown(
                f'<div style="background:{bt_color}12;border:1px solid {bt_color};'
                f'border-radius:6px;padding:10px 16px;margin:4px 0">'
                f'<div style="display:flex;justify-content:space-between;align-items:center">'
                f'<span style="font-size:1.0rem;font-weight:bold;color:{bt_color}">'
                f'{bt["ticker"]} — {dir_word} | Grade {bt["grade"]}</span>'
                f'<span style="color:{NEUTRAL};font-size:0.82rem">'
                f'Confidence: {bt["confidence"]}/100{whale_badge}</span>'
                f'</div>'
                f'<div style="font-family:monospace;font-size:0.78rem;color:#ccc;margin-top:4px">'
                f'Entry: ${bt["entry"]:.2f} | Stop: ${bt["stop_loss"]:.2f} | '
                f'Target: ${bt["take_profit"]:.2f} | R:R 1:{bt["risk_reward"]:.1f} | '
                f'{bt["position_size"]} shares (${bt["dollar_risk"]:,.0f} risk)'
                f'</div>'
                f'<div style="font-size:0.72rem;color:{NEUTRAL};margin-top:2px">'
                f'{bt["signals_for"][0] if bt["signals_for"] else ""}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("No actionable setups found right now. Check back during market hours.")

    # ── Watchlist Whale Scan ─────────────────────────────────────────────────
    st.divider()
    st.markdown(
        f'<span style="font-size:0.9rem;font-weight:bold;color:{BLUE}">WATCHLIST WHALE SCAN</span>',
        unsafe_allow_html=True,
    )
    st.caption("Scanning your watchlist for unusual activity...")

    watchlist = load_watchlist()
    if watchlist:
        whale_results = []
        for wt in watchlist:
            try:
                wd = cached_whales(wt)
                if wd.get("whale_signals"):
                    whale_results.append(wd)
            except Exception:
                continue

        whale_results.sort(key=lambda r: abs(r["whale_score"]), reverse=True)

        if whale_results:
            for wr in whale_results:
                ws = wr["whale_score"]
                wd = wr["whale_direction"]
                wc = GREEN if wd == "Bullish" else (RED if wd == "Bearish" else NEUTRAL)
                alert = wr.get("alert_level", "None")
                alert_badge = ""
                if alert == "Whale Alert":
                    alert_badge = " [WHALE]"
                elif alert == "Alert":
                    alert_badge = " [ALERT]"
                top_sig = wr["whale_signals"][0] if wr["whale_signals"] else ""

                st.markdown(
                    f'<div style="background:{BG2};border-left:3px solid {wc};'
                    f'padding:6px 12px;margin:3px 0;font-family:monospace;font-size:0.78rem">'
                    f'<b style="color:{wc}">{wr["ticker"]}</b>'
                    f'<span style="color:{wc}"> {wd} ({ws:+d}){alert_badge}</span>'
                    f'&nbsp;— {top_sig}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No unusual activity detected in your watchlist.")
    else:
        st.caption("No tickers in watchlist. Add some from the Market View or Screener tab.")


# ── Squeeze Scanner Tab ───────────────────────────────────────────────────────

def _render_squeeze_tab(selected: str):
    """Dedicated squeeze scanner — gamma and short squeeze detection across the market."""

    st.markdown(
        f'<span style="font-size:1.1rem;font-weight:bold;color:{RED}">'
        f'Squeeze Scanner</span>',
        unsafe_allow_html=True,
    )
    st.caption("Scanning for gamma squeeze and short squeeze setups across watchlist + top movers.")

    # Build scan universe: watchlist + top movers + unusual volume
    watchlist = load_watchlist()
    try:
        movers = cached_screener_movers()
        mover_tickers = [r["ticker"] for r in (movers.get("gainers", [])[:10] + movers.get("losers", [])[:5])]
    except Exception:
        mover_tickers = []
    try:
        unusual = cached_unusual_volume()
        unusual_tickers = [r["ticker"] for r in unusual[:10]]
    except Exception:
        unusual_tickers = []

    scan_universe = list(dict.fromkeys(watchlist + mover_tickers + unusual_tickers))

    col_refresh, col_count = st.columns([1, 5])
    with col_refresh:
        if st.button("Rescan", key="squeeze_rescan", type="primary"):
            cached_whales.clear()
            st.rerun()
    with col_count:
        st.caption(f"Scanning {len(scan_universe)} tickers...")

    # Scan all tickers
    gamma_candidates = []
    short_candidates = []
    all_squeeze_data = []

    progress = st.progress(0, text="Scanning...")
    for i, stk in enumerate(scan_universe):
        progress.progress((i + 1) / len(scan_universe), text=f"Scanning {stk}... ({i+1}/{len(scan_universe)})")
        try:
            wd = cached_whales(stk)
            sq = wd.get("squeeze", {})
            whale_score = wd.get("whale_score", 0)
            whale_dir = wd.get("whale_direction", "Neutral")

            entry = {
                "ticker": stk,
                "whale_score": whale_score,
                "whale_direction": whale_dir,
                "alert_level": wd.get("alert_level", "None"),
                "gamma_risk": sq.get("gamma_squeeze_risk", "None"),
                "gamma_score": sq.get("gamma_score", 0),
                "short_risk": sq.get("short_squeeze_risk", "None"),
                "short_score": sq.get("short_score", 0),
                "squeeze_signals": sq.get("squeeze_signals", []),
                "options_flow": wd.get("options_flow", {}),
                "volume": wd.get("volume", {}),
            }
            all_squeeze_data.append(entry)

            if sq.get("gamma_squeeze_risk") in ("High", "Moderate", "Low"):
                gamma_candidates.append(entry)
            if sq.get("short_squeeze_risk") in ("High", "Moderate", "Low"):
                short_candidates.append(entry)
        except Exception:
            continue

    progress.empty()

    gamma_candidates.sort(key=lambda x: x["gamma_score"], reverse=True)
    short_candidates.sort(key=lambda x: x["short_score"], reverse=True)

    # ── Summary metrics ──────────────────────────────────────────────────────
    high_gamma = len([g for g in gamma_candidates if g["gamma_risk"] == "High"])
    mod_gamma = len([g for g in gamma_candidates if g["gamma_risk"] == "Moderate"])
    high_short = len([s for s in short_candidates if s["short_risk"] == "High"])
    mod_short = len([s for s in short_candidates if s["short_risk"] == "Moderate"])

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Gamma - High Risk", high_gamma)
    with m2:
        st.metric("Gamma - Moderate", mod_gamma)
    with m3:
        st.metric("Short - High Risk", high_short)
    with m4:
        st.metric("Short - Moderate", mod_short)

    # ── Best Squeeze Setups (Ranked) ─────────────────────────────────────────
    # Combine all squeeze candidates (High + Moderate only), run trade coach, rank by R:R
    top_squeeze = []
    seen_tickers = set()
    for c in gamma_candidates + short_candidates:
        if c["ticker"] in seen_tickers:
            continue
        risk = c.get("gamma_risk", "None") if c in gamma_candidates else c.get("short_risk", "None")
        if risk not in ("High", "Moderate"):
            continue
        seen_tickers.add(c["ticker"])
        top_squeeze.append(c)

    if top_squeeze:
        st.markdown(
            f'<div style="margin-top:8px"><span style="font-size:1.0rem;font-weight:bold;'
            f'color:{GREEN}">BEST SQUEEZE SETUPS — RANKED BY RISK/REWARD</span></div>',
            unsafe_allow_html=True,
        )
        st.caption("Trade coach analysis on every High/Moderate squeeze candidate.")

        setup_rows = []
        for sq_c in top_squeeze:
            try:
                # Get account settings from session or defaults
                acct = st.session_state.get("whale_acct", 100_000) or 100_000
                rpct = st.session_state.get("whale_risk", 1.0) or 1.0
                coach = cached_trade_coach(sq_c["ticker"], account_value=float(acct), risk_pct=float(rpct))
                squeeze_type = []
                if sq_c.get("gamma_risk") in ("High", "Moderate"):
                    squeeze_type.append(f"Gamma {sq_c['gamma_risk']}")
                if sq_c.get("short_risk") in ("High", "Moderate"):
                    squeeze_type.append(f"Short {sq_c['short_risk']}")
                coach["_squeeze_type"] = " + ".join(squeeze_type)
                coach["_whale_score"] = sq_c.get("whale_score", 0)
                coach["_whale_dir"] = sq_c.get("whale_direction", "Neutral")
                coach["_gamma_score"] = sq_c.get("gamma_score", 0)
                coach["_short_score"] = sq_c.get("short_score", 0)
                setup_rows.append(coach)
            except Exception:
                continue

        # Long only — filter out shorts and no-trade
        setup_rows = [s for s in setup_rows if s.get("direction") == "Long"]

        # Rank: grade first, then R:R, then confidence
        grade_rank = {"A": 0, "B": 1, "C": 2, "D": 3, "F": 4}
        setup_rows.sort(key=lambda x: (
            grade_rank.get(x.get("grade", "F"), 4),
            -(x.get("risk_reward", 0)),
            -(x.get("confidence", 0)),
        ))

        for rank, sr in enumerate(setup_rows, 1):
            ticker = sr["ticker"]
            grade = sr.get("grade", "F")
            verdict = sr.get("verdict", "NO TRADE")
            direction = sr.get("direction", "Neutral")
            rr = sr.get("risk_reward", 0)
            conf = sr.get("confidence", 0)
            entry = sr.get("entry")
            stop = sr.get("stop_loss")
            tp = sr.get("take_profit")
            shares = sr.get("position_size", 0)
            d_risk = sr.get("dollar_risk", 0)
            d_reward = shares * sr.get("reward_per_share", 0) if shares else 0
            sq_type = sr.get("_squeeze_type", "")
            w_score = sr.get("_whale_score", 0)
            w_dir = sr.get("_whale_dir", "Neutral")

            # Colors
            if grade in ("A", "B"):
                card_color = GREEN
            elif grade == "C":
                card_color = YELLOW
            else:
                card_color = RED

            dir_word = "LONG" if direction == "Long" else ("SHORT" if direction == "Short" else "—")
            whale_c = GREEN if w_dir == "Bullish" else (RED if w_dir == "Bearish" else NEUTRAL)

            # Entry/stop/target line
            if entry is not None:
                levels_str = (
                    f'Entry: ${entry:.2f} | Stop: ${stop:.2f} | Target: ${tp:.2f} | '
                    f'R:R 1:{rr:.1f} | {shares} shares | '
                    f'Risk: ${d_risk:,.0f} | Reward: ${d_reward:,.0f}'
                )
            else:
                levels_str = "No clear entry — wait for setup"

            # Top reason
            top_for = sr.get("signals_for", [""])[0] if sr.get("signals_for") else ""

            # Card + Trade button side by side
            card_col, btn_col = st.columns([6, 1])
            with card_col:
                st.markdown(
                    f'<div style="background:{card_color}10;border:1px solid {card_color};'
                    f'border-radius:8px;padding:12px 18px;margin:6px 0">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center">'
                    f'<span style="font-size:1.1rem;font-weight:bold;color:{card_color}">'
                    f'#{rank} {ticker} — {dir_word} | Grade {grade}'
                    f'</span>'
                    f'<span style="font-size:0.82rem">'
                    f'<span style="color:{card_color}">{verdict}</span>'
                    f' | Conf: {conf}/100'
                    f' | <span style="color:{whale_c}">Whale {w_dir} ({w_score:+d})</span>'
                    f'</span>'
                    f'</div>'
                    f'<div style="font-size:0.72rem;color:{YELLOW};margin:3px 0">{sq_type}</div>'
                    f'<div style="font-family:monospace;font-size:0.8rem;color:#ddd;margin:4px 0">'
                    f'{levels_str}'
                    f'</div>'
                    f'<div style="font-size:0.72rem;color:{NEUTRAL};margin-top:2px">{top_for}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with btn_col:
                if entry is not None and shares > 0:
                    st.markdown("<br>", unsafe_allow_html=True)
                    side = "buy" if direction == "Long" else "sell"
                    confirm_key = f"squeeze_confirm_{ticker}"
                    # Two-step: Trade → Confirm
                    if st.session_state.get(confirm_key):
                        c1, c2 = st.columns(2)
                        with c1:
                            if st.button("Confirm", key=f"squeeze_yes_{ticker}", type="primary"):
                                with st.spinner("Placing order..."):
                                    result = place_bracket_order(
                                        ticker=ticker,
                                        qty=shares,
                                        side=side,
                                        limit_price=entry,
                                        stop_loss=stop,
                                        take_profit=tp,
                                    )
                                if result.get("error"):
                                    st.error(f"Failed: {result['error']}")
                                else:
                                    st.success(f"Bracket order placed! {side.upper()} {shares} {ticker} @ ${entry:.2f}")
                                    log_trade(ticker, entry, shares, side,
                                              strategy_notes=f"Squeeze scanner bracket. SL=${stop:.2f} TP=${tp:.2f}")
                                st.session_state[confirm_key] = False
                        with c2:
                            if st.button("Cancel", key=f"squeeze_no_{ticker}"):
                                st.session_state[confirm_key] = False
                                st.rerun()
                    else:
                        if st.button("Trade", key=f"squeeze_trade_{ticker}", type="primary"):
                            st.session_state[confirm_key] = True
                            st.rerun()

    # ── Gamma Squeeze Section ────────────────────────────────────────────────
    st.markdown(
        f'<div style="margin-top:12px"><span style="font-size:1.0rem;font-weight:bold;color:{RED}">'
        f'GAMMA SQUEEZE CANDIDATES</span></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Gamma squeeze = massive call OI near current price forces market makers to buy shares "
        "to hedge, creating a self-reinforcing price surge."
    )

    if gamma_candidates:
        for gc in gamma_candidates:
            risk = gc["gamma_risk"]
            score = gc["gamma_score"]
            gc_color = RED if risk == "High" else (YELLOW if risk == "Moderate" else NEUTRAL)

            # Options flow context
            opt = gc.get("options_flow", {})
            call_vol = opt.get("total_call_volume", 0)
            put_vol = opt.get("total_put_volume", 0)
            pc = opt.get("put_call_ratio")
            call_prem = opt.get("total_call_premium", 0)
            whale_badge = ""
            if gc["alert_level"] in ("Whale Alert", "Alert"):
                whale_badge = (
                    f'<span style="color:{GREEN if gc["whale_direction"] == "Bullish" else RED};'
                    f'margin-left:8px;font-size:0.72rem">'
                    f'Whale: {gc["whale_direction"]} ({gc["whale_score"]:+d})</span>'
                )

            st.markdown(
                f'<div style="background:{gc_color}12;border:1px solid {gc_color};'
                f'border-radius:6px;padding:10px 16px;margin:6px 0">'
                f'<div style="display:flex;justify-content:space-between;align-items:center">'
                f'<span style="font-size:1.05rem;font-weight:bold;color:{gc_color}">'
                f'{gc["ticker"]} — GAMMA {risk.upper()} (Score: {score}/100)</span>'
                f'{whale_badge}'
                f'</div>'
                f'<div style="font-family:monospace;font-size:0.76rem;color:#bbb;margin-top:4px">'
                f'Calls: {call_vol:,} | Puts: {put_vol:,} | '
                f'P/C: {f"{pc:.2f}" if pc is not None else "N/A"} | '
                f'Call Premium: ${call_prem:,.0f}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Squeeze signals detail
            for sig in gc["squeeze_signals"]:
                if "gamma" in sig.lower() or "call" in sig.lower() or "sweep" in sig.lower():
                    st.markdown(
                        f'<div style="font-family:monospace;font-size:0.74rem;color:{gc_color};'
                        f'padding:1px 0 1px 20px">* {sig}</div>',
                        unsafe_allow_html=True,
                    )

            # Trade idea
            if risk in ("High", "Moderate"):
                st.markdown(
                    f'<div style="background:{BG2};border-radius:4px;padding:6px 14px;'
                    f'margin:2px 0 8px 0;font-size:0.74rem;color:{NEUTRAL}">'
                    f'<b>Trade idea:</b> Buy shares or near-ATM calls. '
                    f'As price rises through strike clusters, market maker hedging '
                    f'accelerates the move. Set stop below nearest support. '
                    f'Take partial profits at each +5% level.</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.caption("No gamma squeeze candidates detected in current scan.")

    # ── Short Squeeze Section ────────────────────────────────────────────────
    st.markdown(
        f'<div style="margin-top:16px"><span style="font-size:1.0rem;font-weight:bold;color:{RED}">'
        f'SHORT SQUEEZE CANDIDATES</span></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Short squeeze = high short interest + rising price + volume surge forces short sellers "
        "to cover (buy back shares), creating rapid price spikes."
    )

    if short_candidates:
        for sc in short_candidates:
            risk = sc["short_risk"]
            score = sc["short_score"]
            sc_color = RED if risk == "High" else (YELLOW if risk == "Moderate" else NEUTRAL)

            vol_data = sc.get("volume", {})
            vol_ratio = vol_data.get("volume_ratio", 1.0)
            accum = vol_data.get("accumulation", "Neutral")
            whale_badge = ""
            if sc["alert_level"] in ("Whale Alert", "Alert"):
                whale_badge = (
                    f'<span style="color:{GREEN if sc["whale_direction"] == "Bullish" else RED};'
                    f'margin-left:8px;font-size:0.72rem">'
                    f'Whale: {sc["whale_direction"]} ({sc["whale_score"]:+d})</span>'
                )

            st.markdown(
                f'<div style="background:{sc_color}12;border:1px solid {sc_color};'
                f'border-radius:6px;padding:10px 16px;margin:6px 0">'
                f'<div style="display:flex;justify-content:space-between;align-items:center">'
                f'<span style="font-size:1.05rem;font-weight:bold;color:{sc_color}">'
                f'{sc["ticker"]} — SHORT SQUEEZE {risk.upper()} (Score: {score}/100)</span>'
                f'{whale_badge}'
                f'</div>'
                f'<div style="font-family:monospace;font-size:0.76rem;color:#bbb;margin-top:4px">'
                f'Volume: {vol_ratio:.1f}x avg | Pattern: {accum}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            for sig in sc["squeeze_signals"]:
                if "short" in sig.lower() or "cover" in sig.lower() or "days" in sig.lower() \
                        or "interest" in sig.lower() or "price up" in sig.lower():
                    st.markdown(
                        f'<div style="font-family:monospace;font-size:0.74rem;color:{sc_color};'
                        f'padding:1px 0 1px 20px">* {sig}</div>',
                        unsafe_allow_html=True,
                    )

            if risk in ("High", "Moderate"):
                st.markdown(
                    f'<div style="background:{BG2};border-radius:4px;padding:6px 14px;'
                    f'margin:2px 0 8px 0;font-size:0.74rem;color:{NEUTRAL}">'
                    f'<b>Trade idea:</b> Buy shares on volume confirmation. '
                    f'Short squeezes are explosive but brief — set tight trailing stop. '
                    f'Take 50% off at +10%, trail rest. '
                    f'Do NOT hold overnight if momentum fades.</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.caption("No short squeeze candidates detected in current scan.")

    # ── Full Scan Table ──────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        f'<span style="font-size:0.9rem;font-weight:bold;color:{BLUE}">FULL SCAN RESULTS</span>',
        unsafe_allow_html=True,
    )

    # Header
    st.markdown(
        '<div style="display:grid;'
        'grid-template-columns:80px 75px 75px 75px 75px 1fr;'
        'gap:4px;padding:5px 0 3px 0;'
        'font-size:0.7rem;color:#555;font-weight:bold;'
        'border-bottom:1px solid #2a2d35;font-family:monospace">'
        '<span>TICKER</span>'
        '<span>GAMMA</span>'
        '<span>SHORT</span>'
        '<span>WHALE</span>'
        '<span>FLOW</span>'
        '<span>TOP SIGNAL</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Sort by combined squeeze score
    all_squeeze_data.sort(
        key=lambda x: x["gamma_score"] + x["short_score"] + abs(x["whale_score"]),
        reverse=True,
    )

    for row in all_squeeze_data:
        gr = row["gamma_risk"]
        sr = row["short_risk"]
        gs = row["gamma_score"]
        ss = row["short_score"]
        ws = row["whale_score"]
        wd = row["whale_direction"]

        # Skip tickers with no meaningful activity
        if gs == 0 and ss == 0 and abs(ws) < 30:
            continue

        g_color = RED if gr == "High" else (YELLOW if gr == "Moderate" else (NEUTRAL if gr == "Low" else "#333"))
        s_color = RED if sr == "High" else (YELLOW if sr == "Moderate" else (NEUTRAL if sr == "Low" else "#333"))
        w_color = GREEN if wd == "Bullish" else (RED if wd == "Bearish" else NEUTRAL)

        g_str = f"{gs}" if gs > 0 else "—"
        s_str = f"{ss}" if ss > 0 else "—"
        w_str = f"{ws:+d}" if ws != 0 else "—"

        top_sig = row["squeeze_signals"][0] if row["squeeze_signals"] else "—"
        top_sig = top_sig[:70] + "..." if len(top_sig) > 70 else top_sig

        flow_dir = row.get("options_flow", {}).get("flow_direction", "—")
        flow_c = GREEN if flow_dir == "Bullish" else (RED if flow_dir == "Bearish" else NEUTRAL)

        st.markdown(
            f'<div style="display:grid;'
            f'grid-template-columns:80px 75px 75px 75px 75px 1fr;'
            f'gap:4px;padding:3px 0;'
            f'font-size:0.76rem;font-family:monospace;'
            f'border-bottom:1px solid #1a1d24">'
            f'<span style="font-weight:bold;color:#ddd">{row["ticker"]}</span>'
            f'<span style="color:{g_color};font-weight:bold">{g_str}</span>'
            f'<span style="color:{s_color};font-weight:bold">{s_str}</span>'
            f'<span style="color:{w_color}">{w_str}</span>'
            f'<span style="color:{flow_c}">{flow_dir}</span>'
            f'<span style="color:{NEUTRAL};font-size:0.72rem">{top_sig}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── Portfolio Tab ─────────────────────────────────────────────────────────────

def _render_portfolio_tab():
    """Render the Alpaca paper trading portfolio panel."""

    # ── Header ────────────────────────────────────────────────────────────────
    hdr_col, btn_col = st.columns([6, 1])
    with hdr_col:
        mh = get_market_hours()
        session_color = GREEN if mh.get("is_open") else YELLOW
        st.markdown(
            f'<span style="font-size:1.1rem;font-weight:bold;color:{GREEN}">'
            f'PORTFOLIO</span>'
            f'&nbsp;&nbsp;<span style="color:{session_color};font-size:0.78rem">'
            f'{mh.get("market_session","")}</span>'
            f'&nbsp;&nbsp;<span style="color:{NEUTRAL};font-size:0.78rem">'
            f'{mh.get("current_time_et","")}</span>',
            unsafe_allow_html=True,
        )
    with btn_col:
        if st.button("Refresh", key="portfolio_refresh", width="stretch"):
            cached_account.clear()
            cached_positions.clear()
            cached_orders.clear()
            st.rerun()

    # ── ACCOUNT SUMMARY ───────────────────────────────────────────────────────
    acct = cached_account()
    if acct.get("error"):
        st.error(f"Alpaca connection error: {acct['error']}")
        st.caption("Check ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Portfolio", f"${acct.get('portfolio_value', 0):,.0f}")
        with c2:
            st.metric("Cash", f"${acct.get('cash', 0):,.0f}")
        with c3:
            st.metric("Buying Power", f"${acct.get('buying_power', 0):,.0f}")

        c4, c5 = st.columns(2)
        with c4:
            dpnl     = acct.get("daily_pnl", 0)
            dpnl_pct = acct.get("daily_pnl_pct", 0)
            sign     = "+" if dpnl >= 0 else ""
            st.metric(
                "Day P&L",
                f"{sign}${dpnl:,.2f}",
                delta=f"{sign}{dpnl_pct:.2f}%",
            )
        with c5:
            upl  = acct.get("unrealized_pl", 0)
            uplpc = acct.get("unrealized_plpc", 0)
            sign = "+" if upl >= 0 else ""
            st.metric(
                "Unrealized P&L",
                f"{sign}${upl:,.2f}",
                delta=f"{sign}{uplpc:.2f}%",
            )

    st.divider()

    # ── OPEN POSITIONS ────────────────────────────────────────────────────────
    st.markdown(
        f'<span style="color:{NEUTRAL};font-size:0.75rem;font-weight:bold">'
        f'OPEN POSITIONS</span>',
        unsafe_allow_html=True,
    )

    positions = cached_positions()
    if not positions:
        st.caption("No open positions.")
    else:
        # Header strip
        st.markdown(
            '<div style="display:grid;'
            'grid-template-columns:80px 60px 90px 90px 90px 80px 1fr;'
            'gap:4px;padding:5px 0 3px 0;'
            'font-size:0.7rem;color:#555;font-weight:bold;'
            'border-bottom:1px solid #2a2d35;font-family:monospace">'
            '<span>TICKER</span><span>QTY</span>'
            '<span>AVG ENTRY</span><span>CURRENT</span>'
            '<span>P&L</span><span>P&L %</span>'
            '<span></span>'
            '</div>',
            unsafe_allow_html=True,
        )
        for pos in positions:
            ticker = pos["ticker"]
            pl     = pos.get("unrealized_pl", 0)
            plpc   = pos.get("unrealized_plpc", 0)
            pl_c   = GREEN if pl >= 0 else RED
            sign   = "+" if pl >= 0 else ""

            pc1, pc2, pc3, pc4, pc5, pc6, pc7 = st.columns([1, 0.75, 1.1, 1.1, 1.1, 0.9, 1.1])
            with pc1:
                st.markdown(
                    f'<span style="font-weight:bold;color:{BLUE};font-family:monospace">'
                    f'{ticker}</span>',
                    unsafe_allow_html=True,
                )
            with pc2:
                st.markdown(
                    f'<span style="font-size:0.82rem">{pos["qty"]:.0f}</span>',
                    unsafe_allow_html=True,
                )
            with pc3:
                st.markdown(
                    f'<span style="font-size:0.82rem">'
                    f'${pos.get("avg_entry_price",0):.2f}</span>',
                    unsafe_allow_html=True,
                )
            with pc4:
                st.markdown(
                    f'<span style="font-size:0.82rem">'
                    f'${pos.get("current_price",0):.2f}</span>',
                    unsafe_allow_html=True,
                )
            with pc5:
                st.markdown(
                    f'<span style="color:{pl_c};font-size:0.82rem">'
                    f'{sign}${pl:,.2f}</span>',
                    unsafe_allow_html=True,
                )
            with pc6:
                st.markdown(
                    f'<span style="color:{pl_c};font-size:0.82rem">'
                    f'{sign}{plpc:.2f}%</span>',
                    unsafe_allow_html=True,
                )
            with pc7:
                if st.button("Close", key=f"close_pos_{ticker}"):
                    result = close_position(ticker)
                    if result.get("error"):
                        st.error(result["error"])
                    else:
                        cached_positions.clear()
                        cached_account.clear()
                        st.success(f"Closed {ticker}")
                        st.rerun()

    st.divider()

    # ── NEW ORDER PANEL ───────────────────────────────────────────────────────
    st.markdown(
        f'<span style="color:{NEUTRAL};font-size:0.75rem;font-weight:bold">'
        f'NEW PAPER ORDER</span>',
        unsafe_allow_html=True,
    )

    # Show pending order confirmation if one is queued
    if st.session_state.pending_order:
        po = st.session_state.pending_order
        is_bracket = po.get("order_type") == "bracket"
        lp_str  = f"  |  Limit ${po['limit_price']:.2f}"  if po.get("limit_price") else ""
        sp_str  = f"  |  Stop ${po['stop_price']:.2f}"    if po.get("stop_price")  else ""
        est_val = (po.get("limit_price") or po.get("stop_price") or 0) * po["qty"]
        est_str = f"  |  Est. value ${est_val:,.2f}" if est_val else ""

        if is_bracket:
            tp_price = po.get("_coach_tp", 0)
            rr_val = po.get("_coach_rr", 0)
            d_risk = po["qty"] * abs(po["limit_price"] - po["stop_price"])
            d_reward = po["qty"] * abs(tp_price - po["limit_price"])
            st.warning(
                f"**BRACKET ORDER:  {po['side'].upper()}  {po['qty']} shares  {po['ticker']}**\n\n"
                f"**Entry:** ${po['limit_price']:.2f} (limit)  |  "
                f"**Stop Loss:** ${po['stop_price']:.2f}  |  "
                f"**Take Profit:** ${tp_price:.2f}\n\n"
                f"**R:R** 1:{rr_val:.1f}  |  "
                f"**Risk:** ${d_risk:,.0f}  |  "
                f"**Reward:** ${d_reward:,.0f}  |  "
                f"**Position:** ${est_val:,.0f}\n\n"
                f"*All 3 orders placed at once. When entry fills, stop loss and take profit go live. "
                f"If one exit hits, the other auto-cancels.*"
            )
        else:
            coach_str = ""
            if po.get("_coach_tp"):
                coach_str = (
                    f"\n\n**Trade Coach levels:**  "
                    f"Stop: ${po['stop_price']:.2f}  |  "
                    f"Target: ${po['_coach_tp']:.2f}  |  "
                    f"R:R 1:{po['_coach_rr']:.1f}"
                )
            st.warning(
                f"Confirm:  **{po['side'].upper()}  {po['qty']} shares  {po['ticker']}**  "
                f"|  {po['order_type'].upper()}{lp_str}{sp_str}{est_str}{coach_str}"
            )

        conf_col, cancel_col = st.columns(2)
        with conf_col:
            if st.button("Confirm Order", key="order_confirm", use_container_width=True):
                if is_bracket:
                    result = place_bracket_order(
                        ticker      = po["ticker"],
                        qty         = po["qty"],
                        side        = po["side"],
                        limit_price = po["limit_price"],
                        stop_loss   = po["stop_price"],
                        take_profit = po["_coach_tp"],
                    )
                else:
                    result = place_order(
                        ticker      = po["ticker"],
                        qty         = po["qty"],
                        side        = po["side"],
                        order_type  = po["order_type"],
                        limit_price = po.get("limit_price"),
                    )
                st.session_state.pending_order = None
                if result.get("error"):
                    st.error(f"Order failed: {result['error']}")
                else:
                    notes = f"Bracket order via squeeze scanner. ID: {result.get('id','')}" if is_bracket \
                        else f"Paper order via dashboard. ID: {result.get('id','')}"
                    log_trade(
                        ticker      = po["ticker"],
                        entry_price = po.get("limit_price") or po.get("stop_price") or 0.0,
                        qty         = po["qty"],
                        side        = po["side"],
                        strategy_notes = notes,
                    )
                    cached_orders.clear()
                    cached_account.clear()
                    legs = result.get("legs", [])
                    leg_str = ""
                    if legs:
                        leg_str = " | Legs: " + ", ".join(
                            f"{l['type']} {l['side']}" for l in legs
                        )
                    st.success(
                        f"Order submitted! ID: {result.get('id','')[:8]}…  "
                        f"Status: {result.get('status','')}{leg_str}"
                    )
                    st.rerun()
        with cancel_col:
            if st.button("✕ Cancel", key="order_cancel", use_container_width=True):
                st.session_state.pending_order = None
                st.rerun()
    else:
        # ── Order form — all fields stacked vertically, no columns ────────────
        # Live price for default limit/stop value
        _live_price = 100.0
        try:
            _md = cached_market_data(st.session_state.selected_ticker)
            _live_price = float(_md.get("price_yf", {}).get("price") or 100.0)
        except Exception:
            pass

        form_ticker = st.text_input(
            "Ticker",
            value=st.session_state.selected_ticker.upper(),
            placeholder="AAPL",
            key="of_ticker",
        ).upper().strip()

        form_qty = st.number_input(
            "Shares", min_value=1, value=10, step=1, key="of_qty"
        )

        form_side = st.selectbox(
            "Side", ["buy", "sell"], key="of_side"
        )

        form_type = st.selectbox(
            "Order Type",
            ["market", "limit", "stop", "stop_limit"],
            key="of_type",
        )

        # Conditional price inputs — rendered outside any form so they
        # appear immediately when order type changes
        form_lp = None
        form_sp = None

        if form_type in ("limit", "stop_limit"):
            form_lp = st.number_input(
                "Limit Price ($)",
                min_value=0.01,
                value=round(_live_price, 2),
                step=0.01,
                format="%.2f",
                key="of_limit_price",
            )

        if form_type in ("stop", "stop_limit"):
            form_sp = st.number_input(
                "Stop Price ($)",
                min_value=0.01,
                value=round(_live_price * 0.98, 2),
                step=0.01,
                format="%.2f",
                key="of_stop_price",
            )

        # Full-width green Place Order button
        st.markdown(
            '<style>'
            'div[data-testid="stButton"]:has(button[kind="primary"]) button {'
            '  background-color: #00ff88 !important;'
            '  color: #0e1117 !important;'
            '  font-weight: bold !important;'
            '}'
            '</style>',
            unsafe_allow_html=True,
        )
        place_clicked = st.button(
            "Place Order",
            key="of_submit",
            use_container_width=True,
            type="primary",
        )

        if place_clicked and form_ticker:
            acct_data = cached_account()
            pos_data  = cached_positions() if not acct_data.get("error") else []
            v = validate_trade(
                ticker        = form_ticker,
                side          = form_side,
                account_value = acct_data.get("portfolio_value", 0),
                buying_power  = acct_data.get("buying_power", 0),
                positions     = pos_data,
            )
            for w in v.get("warnings", []):
                st.warning(w)
            for e in v.get("errors", []):
                st.error(e)
            if v["valid"]:
                st.session_state.pending_order = {
                    "ticker":      form_ticker,
                    "qty":         int(form_qty),
                    "side":        form_side,
                    "order_type":  form_type,
                    "limit_price": form_lp,
                    "stop_price":  form_sp,
                }
                st.rerun()

    st.divider()

    # ── RISK CALCULATOR ───────────────────────────────────────────────────────
    st.markdown(
        f'<span style="color:{NEUTRAL};font-size:0.75rem;font-weight:bold">'
        f'RISK CALCULATOR</span>',
        unsafe_allow_html=True,
    )
    rc_ticker  = st.text_input(
        "Ticker", value=st.session_state.selected_ticker.upper(),
        key="rc_ticker", placeholder="AAPL"
    ).upper().strip()
    rc_entry   = st.number_input("Entry Price ($)", min_value=0.01,
                                 value=150.00, step=0.01, format="%.2f", key="rc_entry")
    rc_side    = st.selectbox("Side", ["buy", "sell"], key="rc_side")
    rc_risk    = st.slider("Risk % of Account", 0.25, 5.0, 1.0, step=0.25, key="rc_risk")
    rc_rr      = st.slider("Reward:Risk Ratio", 1.0, 5.0, 2.0, step=0.5, key="rc_rr")

    if st.button("Calculate", key="rc_calc", use_container_width=True):
        with st.spinner("Fetching ATR…"):
            rc_stop   = calculate_stop_loss(rc_ticker, rc_entry, rc_side)
            rc_target = calculate_take_profit(rc_entry, rc_stop, rc_side, rc_rr)
            acct_data = cached_account()
            acct_val  = acct_data.get("portfolio_value", 100_000)
            rc_shares = calculate_position_size(acct_val, rc_risk, rc_entry, rc_stop)

        if rc_stop and rc_target:
            stop_pct  = abs(rc_entry - rc_stop) / rc_entry * 100
            tgt_pct   = abs(rc_target - rc_entry) / rc_entry * 100
            stop_c    = RED   if rc_side == "buy" else GREEN
            tgt_c     = GREEN if rc_side == "buy" else RED
            st.markdown(
                f'<div style="font-family:monospace;font-size:0.82rem;'
                f'line-height:1.8;background:#1a1d24;padding:10px;border-radius:5px">'
                f'<b style="color:{NEUTRAL}">Entry</b>&nbsp;&nbsp;&nbsp;'
                f'${rc_entry:.2f}<br>'
                f'<b style="color:{stop_c}">Stop</b>&nbsp;&nbsp;&nbsp;&nbsp;'
                f'${rc_stop:.2f}'
                f'<span style="color:{NEUTRAL}"> ({stop_pct:.1f}%)</span><br>'
                f'<b style="color:{tgt_c}">Target</b>&nbsp;&nbsp;'
                f'${rc_target:.2f}'
                f'<span style="color:{NEUTRAL}"> ({tgt_pct:.1f}%)</span><br>'
                f'<b style="color:{BLUE}">Shares</b>&nbsp;&nbsp;'
                f'{rc_shares} @ {rc_risk:.2f}% risk<br>'
                f'<b style="color:{NEUTRAL}">$ Risk</b>&nbsp;&nbsp;&nbsp;'
                f'${acct_val * rc_risk / 100:,.0f}'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("Enter valid ticker and prices to calculate.")

    st.divider()

    # ── RECENT ORDERS ─────────────────────────────────────────────────────────
    st.markdown(
        f'<span style="color:{NEUTRAL};font-size:0.75rem;font-weight:bold">'
        f'RECENT ORDERS</span>',
        unsafe_allow_html=True,
    )

    orders = cached_orders(status="all", limit=15)
    _stale_cutoff = (pd.Timestamp.now() - pd.tseries.offsets.BDay(1)).strftime("%Y-%m-%dT%H:%M:%S")
    if not orders:
        st.caption("No recent orders.")
    else:
        st.markdown(
            '<div style="display:grid;'
            'grid-template-columns:80px 55px 50px 70px 90px 90px 1fr;'
            'gap:4px;padding:5px 0 3px 0;'
            'font-size:0.7rem;color:#555;font-weight:bold;'
            'border-bottom:1px solid #2a2d35;font-family:monospace">'
            '<span>TICKER</span><span>QTY</span><span>SIDE</span>'
            '<span>TYPE</span><span>STATUS</span><span>FILL</span>'
            '<span>SUBMITTED</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        for o in orders:
            side_c  = GREEN if o.get("side") == "buy" else RED
            fill    = o.get("filled_price")
            fill_s  = f"${fill:.2f}" if fill else "—"
            status  = o.get("status", "")
            stat_c  = GREEN if status == "filled" else (RED if status in ("canceled","rejected") else NEUTRAL)
            # Cancel button only for open orders
            oc1, oc2, oc3, oc4, oc5, oc6, oc7 = st.columns([1, 0.7, 0.65, 0.85, 1.1, 1.1, 1.7])
            with oc1:
                st.markdown(
                    f'<span style="font-family:monospace;font-size:0.82rem">{o["ticker"]}</span>',
                    unsafe_allow_html=True,
                )
            with oc2:
                st.markdown(f'<span style="font-size:0.82rem">{o["qty"]:.0f}</span>',
                            unsafe_allow_html=True)
            with oc3:
                st.markdown(
                    f'<span style="color:{side_c};font-size:0.82rem">'
                    f'{o.get("side","").upper()}</span>',
                    unsafe_allow_html=True,
                )
            with oc4:
                st.markdown(f'<span style="font-size:0.8rem">{o.get("order_type","")}</span>',
                            unsafe_allow_html=True)
            with oc5:
                stale_badge = ""
                if status == "new" and o.get("submitted_at", "") < _stale_cutoff:
                    stale_badge = (
                        ' <span style="background:#ff8c00;color:#000;font-size:0.6rem;'
                        'padding:1px 4px;border-radius:3px;font-weight:bold">STALE</span>'
                    )
                st.markdown(
                    f'<span style="color:{stat_c};font-size:0.8rem">{status}{stale_badge}</span>',
                    unsafe_allow_html=True,
                )
            with oc6:
                st.markdown(f'<span style="font-size:0.82rem">{fill_s}</span>',
                            unsafe_allow_html=True)
            with oc7:
                st.markdown(
                    f'<span style="color:{NEUTRAL};font-size:0.76rem">'
                    f'{o.get("submitted_at","")[:16]}</span>',
                    unsafe_allow_html=True,
                )

    st.divider()

    # ── PERFORMANCE SUMMARY ───────────────────────────────────────────────────
    st.markdown(
        f'<span style="color:{NEUTRAL};font-size:0.75rem;font-weight:bold">'
        f'TRADE LOG PERFORMANCE</span>',
        unsafe_allow_html=True,
    )

    perf = get_performance_summary()
    if perf.get("error"):
        st.caption(f"Trade log error: {perf['error']}")
    elif perf.get("total_trades", 0) == 0:
        st.caption("No trades logged yet. Paper trades appear here after confirmation.")
    else:
        pc1, pc2, pc3, pc4, pc5 = st.columns(5)
        with pc1:
            st.metric("Closed Trades", perf.get("closed_trades", 0))
        with pc2:
            wr = perf.get("win_rate")
            st.metric("Win Rate", f"{wr:.1f}%" if wr is not None else "N/A")
        with pc3:
            tpnl = perf.get("total_pnl", 0)
            sign = "+" if tpnl >= 0 else ""
            st.metric("Total P&L", f"{sign}${tpnl:,.2f}")
        with pc4:
            best = perf.get("best_trade")
            st.metric("Best Trade", f"+${best:,.2f}" if best is not None else "N/A")
        with pc5:
            worst = perf.get("worst_trade")
            wc    = RED if (worst or 0) < 0 else NEUTRAL
            st.metric("Worst Trade", f"${worst:,.2f}" if worst is not None else "N/A")

        w_col, l_col, o_col = st.columns(3)
        with w_col:
            st.caption(f"Wins: {perf.get('wins', 0)}")
        with l_col:
            st.caption(f"Losses: {perf.get('losses', 0)}")
        with o_col:
            st.caption(f"Open: {perf.get('open_trades', 0)}")


# ── Journal Tab ──────────────────────────────────────────────────────────────

def _render_journal_tab():
    """Trade journal with full history, P&L curve, and per-trade detail."""

    st.markdown(
        f'<span style="font-size:1.1rem;font-weight:bold;color:{GREEN}">TRADE JOURNAL</span>',
        unsafe_allow_html=True,
    )
    st.caption("Full trade history, performance analytics, and alert log.")

    # ── Performance Summary ───────────────────────────────────────────────
    perf = get_performance_summary()
    if perf.get("total_trades", 0) > 0:
        pc1, pc2, pc3, pc4, pc5, pc6 = st.columns(6)
        with pc1:
            st.metric("Total Trades", perf.get("total_trades", 0))
        with pc2:
            wr = perf.get("win_rate")
            st.metric("Win Rate", f"{wr:.1f}%" if wr is not None else "N/A")
        with pc3:
            tpnl = perf.get("total_pnl", 0)
            sign = "+" if tpnl >= 0 else ""
            pnl_c = "normal" if tpnl >= 0 else "off"
            st.metric("Total P&L", f"{sign}${tpnl:,.2f}")
        with pc4:
            avg = perf.get("avg_pnl")
            st.metric("Avg P&L", f"${avg:,.2f}" if avg is not None else "N/A")
        with pc5:
            best = perf.get("best_trade")
            st.metric("Best Trade", f"+${best:,.2f}" if best is not None else "N/A")
        with pc6:
            worst = perf.get("worst_trade")
            st.metric("Worst Trade", f"${worst:,.2f}" if worst is not None else "N/A")

    # ── Trade History Table ───────────────────────────────────────────────
    st.divider()
    st.markdown(
        f'<span style="font-size:0.85rem;font-weight:bold;color:{NEUTRAL}">TRADE HISTORY</span>',
        unsafe_allow_html=True,
    )

    trades = get_all_trades(limit=100)
    if not trades:
        st.caption("No trades logged yet. Execute trades from Trade Coach or Squeeze Scanner.")
    else:
        # P&L curve for closed trades
        closed = [t for t in trades if t["outcome"] != "open" and t["pnl"] is not None]
        if closed:
            closed_sorted = sorted(closed, key=lambda x: x["id"])
            cumulative = []
            running = 0
            labels = []
            for t in closed_sorted:
                running += t["pnl"]
                cumulative.append(running)
                labels.append(f"#{t['id']} {t['ticker']}")

            fig = go.Figure()
            colors = [GREEN if v >= 0 else RED for v in cumulative]
            fig.add_trace(go.Scatter(
                x=labels, y=cumulative, mode="lines+markers",
                line=dict(color=GREEN, width=2),
                marker=dict(color=colors, size=6),
                hovertemplate="Trade: %{x}<br>Cumulative P&L: $%{y:,.2f}<extra></extra>",
            ))
            fig.update_layout(
                title="Cumulative P&L",
                height=280, margin=dict(l=40, r=20, t=40, b=40),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ccc", size=10),
                xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#2a2d35"),
            )
            fig.add_hline(y=0, line_dash="dash", line_color=NEUTRAL, opacity=0.5)
            st.plotly_chart(fig, use_container_width=True)

        # Trade table
        for t in trades:
            outcome = t["outcome"] or "open"
            if outcome == "win":
                oc = GREEN
            elif outcome == "loss":
                oc = RED
            elif outcome == "open":
                oc = YELLOW
            else:
                oc = NEUTRAL

            pnl_str = ""
            if t["pnl"] is not None:
                s = "+" if t["pnl"] >= 0 else ""
                pnl_str = f'{s}${t["pnl"]:,.2f}'
                if t["pnl_percent"] is not None:
                    pnl_str += f' ({s}{t["pnl_percent"]:.1f}%)'
            else:
                pnl_str = "Open"

            notes = t.get("strategy_notes") or ""
            notes_short = notes[:60] + "..." if len(notes) > 60 else notes

            st.markdown(
                f'<div style="background:{oc}08;border-left:3px solid {oc};'
                f'border-radius:4px;padding:8px 14px;margin:4px 0;'
                f'font-family:monospace;font-size:0.78rem">'
                f'<div style="display:flex;justify-content:space-between;align-items:center">'
                f'<span style="font-weight:bold;color:{oc}">'
                f'#{t["id"]} {t["ticker"]} — {t["side"].upper()} {t["quantity"]} shares '
                f'@ ${t["entry_price"]:.2f}</span>'
                f'<span style="color:{oc}">{pnl_str}</span>'
                f'</div>'
                f'<div style="color:{NEUTRAL};font-size:0.68rem;margin-top:2px">'
                f'Opened: {t["entry_date"][:16] if t["entry_date"] else "?"}'
                f'{" | Closed: " + t["exit_date"][:16] if t.get("exit_date") else ""}'
                f'{" | " + notes_short if notes_short else ""}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Close trade button for open trades
            if outcome == "open":
                cl1, cl2 = st.columns([4, 1])
                with cl2:
                    if st.button("Close Trade", key=f"close_trade_{t['id']}"):
                        try:
                            md = cached_market_data(t["ticker"])
                            cur_price = float(md.get("live_price") or md.get("price_yf", {}).get("price") or 0)
                            if cur_price > 0:
                                close_trade(t["id"], cur_price)
                                st.success(f"Closed #{t['id']} {t['ticker']} @ ${cur_price:.2f}")
                                st.rerun()
                            else:
                                st.error("Could not get current price")
                        except Exception as exc:
                            st.error(f"Error: {exc}")

    # ── Alert Log ─────────────────────────────────────────────────────────
    st.divider()
    al_hdr, al_btn = st.columns([5, 1])
    with al_hdr:
        st.markdown(
            f'<span style="font-size:0.85rem;font-weight:bold;color:{YELLOW}">ALERT LOG</span>',
            unsafe_allow_html=True,
        )
    with al_btn:
        if st.button("Clear", key="clear_alerts"):
            clear_alert_log()
            st.rerun()

    alerts = get_alert_log(limit=50)
    if not alerts:
        st.caption("No alerts logged yet. Alerts fire from scans, price levels, and regime changes.")
    else:
        for al in alerts:
            atype = al["alert_type"]
            if atype == "squeeze":
                ac = RED
            elif atype == "whale":
                ac = GREEN
            elif atype == "price":
                ac = YELLOW
            elif atype == "regime":
                ac = YELLOW
            else:
                ac = NEUTRAL
            st.markdown(
                f'<div style="font-size:0.72rem;font-family:monospace;padding:3px 0;'
                f'border-bottom:1px solid #1a1d25">'
                f'<span style="color:{NEUTRAL}">{al["timestamp"][5:16]}</span> '
                f'<span style="color:{ac};font-weight:bold">[{atype.upper()}]</span> '
                f'<span style="color:#ddd">{al["message"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ── Main render ───────────────────────────────────────────────────────────────

def render():
    selected = st.session_state.selected_ticker

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            f'<p style="color:{GREEN};font-size:1.1rem;font-weight:bold;margin-bottom:4px">'
            f'TRADING TERMINAL</p>',
            unsafe_allow_html=True,
        )
        st.caption(datetime.now().strftime("%a %b %d, %Y  %H:%M:%S"))
        st.divider()

        # ── Recent Alerts (above watchlist) ─────────────────────────────────
        recent_alerts = get_alert_log(limit=8)
        if recent_alerts:
            st.markdown(
                f'<span style="color:{YELLOW};font-size:0.75rem;font-weight:bold">'
                f'RECENT ALERTS</span>',
                unsafe_allow_html=True,
            )
            for ra in recent_alerts:
                atype = ra["alert_type"]
                ac = RED if atype == "squeeze" else (GREEN if atype == "whale" else YELLOW)
                ra_col1, ra_col2 = st.columns([4, 1])
                with ra_col1:
                    st.markdown(
                        f'<div style="font-size:0.62rem;font-family:monospace;padding:1px 0;'
                        f'color:#aaa">'
                        f'<span style="color:{ac}">[{atype[:3].upper()}]</span> '
                        f'{ra["message"][:42]}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with ra_col2:
                    if st.button(ra["ticker"], key=f"alert_go_{ra['id']}", help=f"View {ra['ticker']}"):
                        st.session_state.selected_ticker = ra["ticker"]
                        st.rerun()
            st.divider()

        watchlist = load_watchlist()
        st.markdown("**Watchlist** — sorted by anomaly score")

        from concurrent.futures import ThreadPoolExecutor

        def _fetch_anomaly(ticker):
            try:
                return ticker, cached_anomaly(ticker)
            except Exception:
                return ticker, {"score": 0, "is_watch": False, "reason": ""}

        anomaly_map = {}
        if watchlist:
            with ThreadPoolExecutor(max_workers=min(len(watchlist), 6)) as pool:
                for ticker, anom_result in pool.map(lambda t: _fetch_anomaly(t), watchlist):
                    anomaly_map[ticker] = anom_result

        sorted_watchlist = sorted(
            watchlist,
            key=lambda t: anomaly_map.get(t, {}).get("score", 0),
            reverse=True,
        )

        for wl_t in sorted_watchlist:
            wl_md     = cached_market_data(wl_t)
            wl_price  = wl_md.get("live_price")
            wl_chg    = wl_md.get("change_pct")
            wl_src    = wl_md.get("price_source")   # 'Alpaca Live' | 'yfinance Delayed' | 'Polygon'
            price_str = _fmt_price(wl_price)
            chg_str   = _fmt_pct(wl_chg)

            # Source dot: green = Alpaca Live, yellow = yfinance Delayed, blue = Polygon
            if wl_src == "Alpaca Live":
                src_dot   = "●"
                src_dot_c = GREEN
                src_tip   = "Alpaca Live"
            elif wl_src == "yfinance Delayed":
                src_dot   = "●"
                src_dot_c = YELLOW
                src_tip   = "yfinance Delayed"
            elif wl_src == "Polygon":
                src_dot   = "●"
                src_dot_c = BLUE
                src_tip   = "Polygon EOD"
            else:
                src_dot   = "○"
                src_dot_c = NEUTRAL
                src_tip   = "No price"

            yf_sent    = cached_yahoo_fast(wl_t)
            fast_label = _quick_sentiment_label(yf_sent.get("score"))
            label_c    = _label_color(fast_label)

            an       = anomaly_map.get(wl_t, {})
            is_watch = an.get("is_watch", False)
            a_score  = an.get("score", 0)
            reason   = an.get("reason", "")
            fire     = " 🔥" if is_watch else ""

            col_btn, col_x = st.columns([5, 1])
            with col_btn:
                prefix   = "> " if wl_t == selected else "  "
                btn_text = f"{prefix}{wl_t}{fire}  {price_str}  {chg_str}"
                if st.button(btn_text, key=f"wl_{wl_t}", width="stretch",
                             help=f"Price source: {src_tip}"):
                    st.session_state.selected_ticker = wl_t
                    st.rerun()
            with col_x:
                if st.button("×", key=f"rm_{wl_t}", help=f"Remove {wl_t}"):
                    remaining = remove_ticker(wl_t)
                    if st.session_state.selected_ticker == wl_t:
                        st.session_state.selected_ticker = remaining[0] if remaining else "AAPL"
                    st.rerun()

            q_score = an.get("quality_score", 0)
            badge_html = (
                f'<span style="color:{src_dot_c};font-size:0.6rem" title="{src_tip}">{src_dot}</span>'
                f'<span style="color:{NEUTRAL};font-size:0.62rem;margin-left:3px">{wl_src or "—"}</span>'
                f'&nbsp;&nbsp;'
                f'<span style="color:{label_c};font-size:0.72rem">{fast_label}</span>'
                f'<span style="color:{NEUTRAL};font-size:0.68rem;margin-left:6px">'
                f'[{a_score} sig] [Q:{q_score}]</span>'
            )
            if is_watch and reason:
                direction  = an.get("direction", "")
                dir_color  = GREEN if direction == "Long" else (RED if direction == "Short" else NEUTRAL)
                dir_txt    = (f'<span style="color:{dir_color};font-size:0.65rem">'
                              f'{direction}</span> ' if direction else "")
                badge_html += (
                    f'<br>{dir_txt}<span style="color:{YELLOW};font-size:0.65rem">'
                    f'{reason[:50]}{"..." if len(reason)>50 else ""}</span>'
                )
            st.markdown(badge_html, unsafe_allow_html=True)

        # ── Top Picks — auto-ranked best trades ────────────────────────────────
        st.divider()
        st.markdown(
            f'<span style="color:{YELLOW};font-size:0.75rem;font-weight:bold">'
            f'TOP PICKS — BEST TRADES NOW</span>',
            unsafe_allow_html=True,
        )
        try:
            _scan_results = cached_market_scan()
            _top_picks = [r for r in _scan_results if r.get("ticker") not in set(watchlist) and r.get("direction") == "Long"][:8]
            if _top_picks:
                for _tp in _top_picks:
                    _tp_ticker = _tp.get("ticker", "?")
                    _tp_price  = _fmt_price(_tp.get("price"))
                    _tp_chg    = _fmt_pct(_tp.get("change_pct"))
                    _tp_score  = _tp.get("score", 0)
                    _tp_dir    = _tp.get("direction", "")
                    _tp_qual   = _tp.get("quality_score", 0)
                    _tp_reason = _tp.get("reason", "")[:45]
                    _dir_c     = GREEN if _tp_dir == "Long" else (RED if _tp_dir == "Short" else NEUTRAL)

                    _tp_col1, _tp_col2 = st.columns([5, 1])
                    with _tp_col1:
                        _tp_label = f"  {_tp_ticker}  {_tp_price}  {_tp_chg}"
                        if st.button(_tp_label, key=f"tp_{_tp_ticker}", width="stretch"):
                            st.session_state.selected_ticker = _tp_ticker
                            st.rerun()
                    with _tp_col2:
                        if st.button("+", key=f"tpa_{_tp_ticker}", help=f"Add {_tp_ticker} to watchlist"):
                            add_ticker(_tp_ticker)
                            st.rerun()
                    _tp_html = (
                        f'<span style="color:{_dir_c};font-size:0.65rem">{_tp_dir}</span> '
                        f'<span style="color:{NEUTRAL};font-size:0.65rem">'
                        f'[{_tp_score} sig] [Q:{_tp_qual}]</span>'
                        f'<br><span style="color:{YELLOW};font-size:0.6rem">{_tp_reason}</span>'
                    )
                    st.markdown(_tp_html, unsafe_allow_html=True)
            else:
                st.caption("No top picks found — scan may still be loading")
        except Exception:
            st.caption("Scan loading...")

        # ── Add ticker to watchlist ───────────────────────────────────────────
        st.divider()
        st.markdown(
            f'<span style="color:{NEUTRAL};font-size:0.75rem;font-weight:bold">'
            f'ADD TICKER TO WATCHLIST</span>',
            unsafe_allow_html=True,
        )
        _add1, _add2 = st.columns([3, 1])
        with _add1:
            new_ticker = st.text_input(
                "add_ticker_sidebar", label_visibility="collapsed",
                placeholder="e.g. MSFT, AMD",
                key="ticker_input_box",
            )
        with _add2:
            if st.button("Add", key="add_ticker_btn", width="stretch"):
                t = (new_ticker or "").strip().upper()
                if t:
                    add_ticker(t)
                    st.session_state.selected_ticker = t
                    st.rerun()

        # ── Price Alerts ─────────────────────────────────────────────────────
        st.divider()
        st.markdown(
            f'<span style="color:{YELLOW};font-size:0.75rem;font-weight:bold">'
            f'PRICE ALERTS</span>',
            unsafe_allow_html=True,
        )

        # Check triggered alerts
        price_map = {}
        for wl_t in watchlist:
            try:
                wl_md = cached_market_data(wl_t)
                p = wl_md.get("live_price") or (wl_md.get("price_yf", {}) or {}).get("price")
                if p:
                    price_map[wl_t] = float(p)
            except Exception:
                pass
        triggered = check_price_alerts(price_map)
        for trig in triggered:
            st.toast(f"{trig['ticker']} hit ${trig['current_price']:.2f}! "
                     f"(Alert: {trig['direction']} ${trig['target_price']:.2f})")

        # Show active alerts
        active_alerts = get_price_alerts()
        if active_alerts:
            for pa in active_alerts:
                pa_col1, pa_col2 = st.columns([4, 1])
                with pa_col1:
                    arrow = "^" if pa["direction"] == "above" else "v"
                    pa_c = GREEN if pa["direction"] == "above" else RED
                    st.markdown(
                        f'<span style="color:{pa_c};font-size:0.7rem;font-family:monospace">'
                        f'{arrow} {pa["ticker"]} {pa["direction"]} ${pa["target_price"]:.2f}'
                        f'</span>',
                        unsafe_allow_html=True,
                    )
                with pa_col2:
                    if st.button("x", key=f"rma_{pa['id']}"):
                        remove_price_alert(pa["id"])
                        st.rerun()
        else:
            st.caption("No active price alerts.")

        # Add new alert
        with st.expander("Set Alert", expanded=False):
            al_ticker = st.text_input("Ticker", value=selected, key="alert_ticker_input",
                                       placeholder="AAPL")
            al_price = st.number_input("Price", min_value=0.01, value=100.0,
                                        step=0.50, key="alert_price_input")
            al_dir = st.selectbox("When price goes", ["above", "below"], key="alert_dir_input")
            al_note = st.text_input("Note (optional)", key="alert_note_input",
                                     placeholder="e.g. breakout level")
            if st.button("Set Alert", key="set_alert_btn", type="primary"):
                t = (al_ticker or "").strip().upper()
                if t and al_price > 0:
                    add_price_alert(t, al_price, al_dir, al_note)
                    st.success(f"Alert set: {t} {al_dir} ${al_price:.2f}")
                    st.rerun()

        st.divider()
        remaining_secs = max(0, int(st.session_state.next_refresh - time.time()))
        st.caption(f"Auto-refresh in {remaining_secs}s")
        if st.button("Force Refresh", width="stretch"):
            st.cache_data.clear()
            st.session_state.next_refresh = time.time() + 60
            st.rerun()

    # ── TOP BAR ───────────────────────────────────────────────────────────────
    st.markdown(
        f'<span style="font-size:1.3rem;font-weight:bold;color:{GREEN}">TRADING TERMINAL</span>'
        f'<br><span style="font-size:0.8rem;color:{NEUTRAL}">'
        f'{datetime.now().strftime("%A, %B %d %Y  %H:%M:%S")}</span>',
        unsafe_allow_html=True,
    )

    # ── MACRO ROW ─────────────────────────────────────────────────────────────
    macro_items = cached_macro()
    m_cols = st.columns(5)
    for i, m in enumerate(macro_items[:5]):
        with m_cols[i]:
            st.metric(label=m.get("name", ""),
                      value=m.get("display_value", "N/A"),
                      delta=m.get("change_display"))

    # ── MARKET REGIME BANNER ────────────────────────────────────────────────
    regime = cached_market_regime()
    regime_name = regime.get("regime", "Unknown")
    regime_c_map = {"green": GREEN, "red": RED, "yellow": YELLOW, "gray": NEUTRAL}
    regime_color = regime_c_map.get(regime.get("color", "gray"), NEUTRAL)
    spy_trend = regime.get("spy_trend", "?")
    qqq_trend = regime.get("qqq_trend", "?")
    vix_val = regime.get("vix_value", 0)
    vix_lvl = regime.get("vix_level", "?")
    spy_price = regime.get("spy_price", "")
    spy_str = f"${spy_price}" if spy_price else ""
    qqq_price = regime.get("qqq_price", "")
    qqq_str = f"${qqq_price}" if qqq_price else ""

    st.markdown(
        f'<div style="background:{regime_color}10;border:1px solid {regime_color};'
        f'border-radius:6px;padding:8px 16px;margin:4px 0;'
        f'display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap">'
        f'<span style="font-size:0.9rem;font-weight:bold;color:{regime_color}">'
        f'MARKET REGIME: {regime_name}</span>'
        f'<span style="font-size:0.72rem;color:{NEUTRAL}">'
        f'SPY {spy_str} {spy_trend} &nbsp;|&nbsp; QQQ {qqq_str} {qqq_trend} '
        f'&nbsp;|&nbsp; VIX {vix_val} ({vix_lvl})</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if regime.get("warning"):
        st.markdown(
            f'<div style="font-size:0.72rem;color:{regime_color};padding:2px 16px">'
            f'{regime["warning"]}</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── TOP-LEVEL TABS ────────────────────────────────────────────────────────
    tab_market, tab_screener, tab_whales, tab_squeeze, tab_portfolio, tab_backtest, tab_performance, tab_journal = st.tabs(
        ["📊 Market View", "🔍 Screener", "🐋 Whale Flow", "🔥 Squeeze Scanner",
         "💼 Portfolio", "📈 Backtest", "📊 Performance", "📒 Journal"]
    )

    with tab_market:
        _render_market_tab(selected)

    with tab_screener:
        _render_screener_tab()

    with tab_whales:
        _render_whale_tab(selected)

    with tab_squeeze:
        _render_squeeze_tab(selected)

    with tab_portfolio:
        _render_portfolio_tab()

    with tab_backtest:
        render_backtest_tab()

    with tab_performance:
        render_performance_tab()

    with tab_journal:
        _render_journal_tab()

    # ── AUTO-REFRESH (outside tabs — fires regardless of active tab) ──────────
    if time.time() >= st.session_state.next_refresh:
        st.session_state.next_refresh = time.time() + 60
        st.rerun()


render()
