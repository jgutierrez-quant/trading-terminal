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

# Apply urllib3 v2 compat patch for pytrends before importing aggregator
import sentiment.google_trends_client  # noqa: F401

from data.market_data               import get_ticker_data
from data.technicals                import get_technicals
from data.anomaly_detector          import compute_anomaly
from sentiment.sentiment_aggregator  import get_sentiment
from sentiment.yahoo_news_client    import get_news_sentiment as _yahoo_fast
from utils.watchlist                import load_watchlist, add_ticker, remove_ticker
from utils.macro_data               import get_macro_data
from data.alpaca_client             import (get_account, get_positions, get_orders,
                                            place_order, close_position, cancel_order)
from utils.risk_manager             import (calculate_position_size, calculate_stop_loss,
                                            calculate_take_profit, validate_trade,
                                            get_market_hours)
from utils.trade_logger             import log_signal, log_trade, get_performance_summary
from dashboard.backtest_tab         import render_backtest_tab
from data.fundamentals              import get_fundamentals, score_fundamentals, get_short_squeeze_score

logging.basicConfig(level=logging.WARNING)

# ── Colors ────────────────────────────────────────────────────────────────────
GREEN   = "#00ff88"
RED     = "#ff4444"
YELLOW  = "#ffd700"
BLUE    = "#4488ff"
NEUTRAL = "#888888"
BG      = "#0e1117"
BG2     = "#1a1d24"

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
                           check_factor_model=True)


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

    # Stage 10: augment top 15 with fundamentals, then apply composite ranking
    for row in results[:15]:
        try:
            fund  = cached_fundamentals(row["ticker"])
            scored = score_fundamentals(fund)
            row["fundamental_score"]  = scored["fundamental_score"]
            row["fundamental_signal"] = scored["fundamental_signal"]
        except Exception:
            row["fundamental_score"]  = 0
            row["fundamental_signal"] = "Neutral"

    # Stage 11: augment top 10 with factor model composite score
    for row in results[:10]:
        try:
            fm = cached_factor_model(row["ticker"])
            row["composite_factor_score"] = fm.get("composite_score", 50.0)
            row["factor_signal"]          = fm.get("composite_signal", "Neutral")
            row["pead_candidate"]         = fm.get("pead_candidate", False)
        except Exception:
            row["composite_factor_score"] = 50.0
            row["factor_signal"]          = "Neutral"
            row["pead_candidate"]         = False

    def _composite(r):
        anomaly_norm = min((r.get("score") or 0) / 6.0, 1.0) * 30
        fund_norm    = ((r.get("fundamental_score") or 0) + 100) / 200 * 25
        qual_norm    = (r.get("quality_score") or 0) / 100 * 20
        factor_norm  = (r.get("composite_factor_score") or 50.0) / 100 * 25
        return anomaly_norm + fund_norm + qual_norm + factor_norm

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

def _movers_table_html(rows: list, accent: str) -> str:
    """Pure-HTML table for gainers/losers — no interactive elements needed."""
    html = (
        '<table style="width:100%;border-collapse:collapse;'
        'font-size:0.8rem;font-family:monospace">'
        '<tr style="color:#666;border-bottom:1px solid #2a2d35">'
        '<th style="text-align:left;padding:4px 8px">Ticker</th>'
        '<th style="text-align:right;padding:4px 8px">Price</th>'
        '<th style="text-align:right;padding:4px 8px">Chg%</th>'
        '<th style="text-align:right;padding:4px 8px">Vol</th>'
        '</tr>'
    )
    for row in rows:
        chg       = row.get("change_pct")
        chg_c     = GREEN if (chg or 0) >= 0 else RED
        chg_str   = f"{chg:+.2f}%" if chg is not None else "N/A"
        price_str = f'${row["price"]:.2f}' if row.get("price") else "N/A"
        vol_str   = _fmt_volume(row.get("volume"))
        html += (
            f'<tr style="border-bottom:1px solid #1e2128">'
            f'<td style="padding:4px 8px;color:{accent};font-weight:bold">'
            f'{row["ticker"]}</td>'
            f'<td style="padding:4px 8px;text-align:right">{price_str}</td>'
            f'<td style="padding:4px 8px;text-align:right;color:{chg_c}">{chg_str}</td>'
            f'<td style="padding:4px 8px;text-align:right;color:{NEUTRAL}">{vol_str}</td>'
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
                      height=250, margin=dict(l=20, r=20, t=50, b=0))
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
    trend_val_str = "" if trend_val_v is None else f" ({trend_val_v}/100)"
    trend_display = trend_label + trend_val_str
    overall_str   = "N/A" if overall_score is None else _fmt_float(overall_score, 4)

    sent_left, sent_right = st.columns([3, 2])
    with sent_left:
        sc_row = st.columns(3)
        with sc_row[0]:
            st.metric("Yahoo News", yahoo_str)
        with sc_row[1]:
            st.metric("Finviz", finviz_str)
        with sc_row[2]:
            st.metric("G-Trends", trend_display)
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
            st.markdown(_movers_table_html(gainers, GREEN), unsafe_allow_html=True)
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
            st.markdown(_movers_table_html(losers, RED), unsafe_allow_html=True)
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

    with st.spinner(
        "Running anomaly scan across 75 stocks "
        "(first run may take 1–2 min while technicals load)..."
    ):
        scan_results = cached_market_scan()

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
            'grid-template-columns:85px 75px 70px 55px 55px 60px 1fr 88px;'
            'gap:4px;padding:5px 0 3px 0;'
            'font-size:0.7rem;color:#555;font-weight:bold;'
            'border-bottom:1px solid #2a2d35;font-family:monospace">'
            '<span>TICKER</span>'
            '<span>PRICE</span>'
            '<span>CHG%</span>'
            '<span>SCORE</span>'
            '<span>DIR</span>'
            '<span>FACTOR</span>'
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

            dir_color = GREEN if direction == "Long" else (RED if direction == "Short" else NEUTRAL)
            accent    = dir_color
            score_c   = GREEN if score >= 5 else (YELLOW if score >= 4 else NEUTRAL)
            chg_c     = GREEN if (chg or 0) >= 0 else RED
            price_str = f"${price:.2f}" if price else "N/A"
            chg_str   = f"{chg:+.2f}%" if chg is not None else "N/A"
            fm_color  = (GREEN if fm_sig == "Long" else (RED if fm_sig == "Short" else NEUTRAL))
            fm_str    = f"{fm_score:.0f}" if fm_score is not None else "—"

            ticker_display = ticker + (" 🔥" if row.get("is_watch") else "")
            if pead:
                ticker_display += " 📈"

            # One st.columns() strip per row — required for st.button in last column
            c1, c2, c3, c4, c4b, c4c, c5, c6 = st.columns([1.1, 1, 0.9, 0.7, 0.7, 0.75, 4.2, 1.1])

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
            with c5:
                st.markdown(
                    f'<span style="color:{NEUTRAL};font-size:0.76rem">'
                    f'{reason[:80]}{"..." if len(reason) > 80 else ""}</span>',
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
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Portfolio Value", f"${acct.get('portfolio_value', 0):,.2f}")
        with c2:
            st.metric("Cash", f"${acct.get('cash', 0):,.2f}")
        with c3:
            st.metric("Buying Power", f"${acct.get('buying_power', 0):,.2f}")
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
        lp_str  = f"  |  Limit ${po['limit_price']:.2f}"  if po.get("limit_price") else ""
        sp_str  = f"  |  Stop ${po['stop_price']:.2f}"    if po.get("stop_price")  else ""
        est_val = (po.get("limit_price") or po.get("stop_price") or 0) * po["qty"]
        est_str = f"  |  Est. value ${est_val:,.2f}" if est_val else ""
        st.warning(
            f"Confirm:  **{po['side'].upper()}  {po['qty']} shares  {po['ticker']}**  "
            f"|  {po['order_type'].upper()}{lp_str}{sp_str}{est_str}"
        )
        conf_col, cancel_col = st.columns(2)
        with conf_col:
            if st.button("✓ Confirm Order", key="order_confirm", use_container_width=True):
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
                    log_trade(
                        ticker      = po["ticker"],
                        entry_price = po.get("limit_price") or po.get("stop_price") or 0.0,
                        qty         = po["qty"],
                        side        = po["side"],
                        strategy_notes = f"Paper order via dashboard. ID: {result.get('id','')}",
                    )
                    cached_orders.clear()
                    cached_account.clear()
                    st.success(
                        f"Order submitted! ID: {result.get('id','')[:8]}…  "
                        f"Status: {result.get('status','')}"
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
                st.markdown(
                    f'<span style="color:{stat_c};font-size:0.8rem">{status}</span>',
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
            st.metric("Total Trades", perf.get("total_trades", 0))
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

        watchlist = load_watchlist()
        st.markdown("**Watchlist** — sorted by anomaly score")

        anomaly_map = {}
        for wl_t in watchlist:
            try:
                anomaly_map[wl_t] = cached_anomaly(wl_t)
            except Exception:
                anomaly_map[wl_t] = {"score": 0, "is_watch": False, "reason": ""}

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

    st.divider()

    # ── TOP-LEVEL TABS ────────────────────────────────────────────────────────
    tab_market, tab_screener, tab_portfolio, tab_backtest = st.tabs(
        ["📊 Market View", "🔍 Screener", "💼 Portfolio", "📈 Backtest"]
    )

    with tab_market:
        _render_market_tab(selected)

    with tab_screener:
        _render_screener_tab()

    with tab_portfolio:
        _render_portfolio_tab()

    with tab_backtest:
        render_backtest_tab()

    # ── AUTO-REFRESH (outside tabs — fires regardless of active tab) ──────────
    if time.time() >= st.session_state.next_refresh:
        st.session_state.next_refresh = time.time() + 60
        st.rerun()

    time.sleep(10)
    st.rerun()


render()
