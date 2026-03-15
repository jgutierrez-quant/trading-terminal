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

from data.market_data              import get_ticker_data
from data.technicals               import get_technicals
from data.anomaly_detector         import compute_anomaly
from sentiment.sentiment_aggregator import get_sentiment
from sentiment.yahoo_news_client   import get_news_sentiment as _yahoo_fast
from utils.watchlist               import load_watchlist, add_ticker, remove_ticker
from utils.macro_data              import get_macro_data

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


# ── Cached fetchers ───────────────────────────────────────────────────────────

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
    """Calls cached_technicals + cached_sentiment — avoids duplicate API calls."""
    tech = cached_technicals(ticker)
    sent = cached_sentiment(ticker)
    return compute_anomaly(ticker, tech, sent)


@st.cache_data(ttl=300, show_spinner=False)
def cached_yahoo_fast(ticker: str) -> dict:
    return _yahoo_fast(ticker)


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


# ── Chart builders ────────────────────────────────────────────────────────────

def _build_intraday_chart(bars: list, title: str) -> go.Figure:
    """Intraday 1-min candlestick + volume (existing chart, unchanged)."""
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
    bars     = tech.get("daily_bars", [])
    sr_levels = tech.get("support_resistance", [])

    if not bars:
        fig = go.Figure()
        fig.update_layout(title="No daily data available",
                          paper_bgcolor=BG, plot_bgcolor=BG2,
                          font=dict(color="#fafafa"), height=520)
        return fig

    # Extract series from bar list
    dates  = [b["date"]         for b in bars]
    opens  = [b.get("open")     for b in bars]
    highs  = [b.get("high")     for b in bars]
    lows   = [b.get("low")      for b in bars]
    closes = [b.get("close")    for b in bars]
    vols   = [b.get("volume", 0) or 0 for b in bars]
    sma20s = [b.get("sma20")    for b in bars]
    sma50s = [b.get("sma50")    for b in bars]
    bb_ups = [b.get("bb_upper") for b in bars]
    bb_mds = [b.get("bb_middle") for b in bars]
    bb_lws = [b.get("bb_lower") for b in bars]
    macds  = [b.get("macd")       for b in bars]
    msigs  = [b.get("macd_signal") for b in bars]
    mhists = [b.get("macd_hist")   for b in bars]

    vol_colors  = [GREEN if c is not None and o is not None and c >= o else RED
                   for c, o in zip(closes, opens)]
    hist_colors = [GREEN if (h or 0) >= 0 else RED for h in mhists]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.60, 0.15, 0.25],
        vertical_spacing=0.02,
    )

    # ── Row 1: Candlestick ────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=dates, open=opens, high=highs, low=lows, close=closes,
        name="Price",
        increasing_line_color=GREEN, decreasing_line_color=RED,
        showlegend=False,
    ), row=1, col=1)

    # Bollinger Bands — upper first so fill='tonexty' fills down to lower
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

    # Support / Resistance — horizontal dashed lines with price labels
    for sr in sr_levels:
        fig.add_hline(
            y=sr, row=1, col=1,
            line_dash="dot", line_color="#aaaaaa", line_width=0.8,
            annotation_text=f"  ${sr:.2f}",
            annotation_position="right",
            annotation_font=dict(size=9, color="#aaaaaa"),
        )

    # ── Row 2: Volume ─────────────────────────────────────────────────────────
    fig.add_trace(go.Bar(
        x=dates, y=vols, name="Volume",
        marker_color=vol_colors, opacity=0.5, showlegend=False,
    ), row=2, col=1)

    # ── Row 3: MACD ───────────────────────────────────────────────────────────
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
    # MACD panel y-axis label
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

        # Gather anomaly scores for all tickers to enable sort
        anomaly_map = {}
        for wl_t in watchlist:
            try:
                anomaly_map[wl_t] = cached_anomaly(wl_t)
            except Exception:
                anomaly_map[wl_t] = {"score": 0, "is_watch": False, "reason": ""}

        # Sort descending by score (most interesting at top)
        sorted_watchlist = sorted(
            watchlist,
            key=lambda t: anomaly_map.get(t, {}).get("score", 0),
            reverse=True,
        )

        for wl_t in sorted_watchlist:
            wl_md  = cached_market_data(wl_t)
            wl_price = wl_md.get("live_price")
            wl_chg   = wl_md.get("change_pct")
            price_str = _fmt_price(wl_price)
            chg_str   = _fmt_pct(wl_chg)

            yf_sent    = cached_yahoo_fast(wl_t)
            fast_label = _quick_sentiment_label(yf_sent.get("score"))
            label_c    = _label_color(fast_label)

            an       = anomaly_map.get(wl_t, {})
            is_watch = an.get("is_watch", False)
            a_score  = an.get("score", 0)
            reason   = an.get("reason", "")

            # Watch flag icon
            fire = " 🔥" if is_watch else ""

            col_btn, col_x = st.columns([5, 1])
            with col_btn:
                prefix   = "> " if wl_t == selected else "  "
                btn_text = f"{prefix}{wl_t}{fire}  {price_str}  {chg_str}"
                if st.button(btn_text, key=f"wl_{wl_t}", width="stretch"):
                    st.session_state.selected_ticker = wl_t
                    st.rerun()
            with col_x:
                if st.button("×", key=f"rm_{wl_t}", help=f"Remove {wl_t}"):
                    remaining = remove_ticker(wl_t)
                    if st.session_state.selected_ticker == wl_t:
                        st.session_state.selected_ticker = remaining[0] if remaining else "AAPL"
                    st.rerun()

            # Sentiment badge + anomaly score
            badge_html = (
                f'<span style="color:{label_c};font-size:0.72rem">{fast_label}</span>'
                f'<span style="color:{NEUTRAL};font-size:0.68rem;margin-left:6px">'
                f'[{a_score} sig]</span>'
            )
            if is_watch and reason:
                badge_html += (
                    f'<br><span style="color:{YELLOW};font-size:0.65rem">'
                    f'{reason[:50]}{"..." if len(reason)>50 else ""}</span>'
                )
            st.markdown(badge_html, unsafe_allow_html=True)

        st.divider()
        remaining_secs = max(0, int(st.session_state.next_refresh - time.time()))
        st.caption(f"Auto-refresh in {remaining_secs}s")
        if st.button("Force Refresh", width="stretch"):
            st.cache_data.clear()
            st.session_state.next_refresh = time.time() + 60
            st.rerun()

    # ── TOP BAR ───────────────────────────────────────────────────────────────
    col_title, col_input = st.columns([3, 5])
    with col_title:
        st.markdown(
            f'<span style="font-size:1.3rem;font-weight:bold;color:{GREEN}">TRADING TERMINAL</span>'
            f'<br><span style="font-size:0.8rem;color:{NEUTRAL}">'
            f'{datetime.now().strftime("%A, %B %d %Y  %H:%M:%S")}</span>',
            unsafe_allow_html=True,
        )
    with col_input:
        col_box, col_add = st.columns([4, 1])
        with col_box:
            new_ticker = st.text_input(
                "ticker_add", label_visibility="collapsed",
                placeholder="Add ticker... (e.g. MSFT, AMD, GLD)",
                key="ticker_input_box",
            )
        with col_add:
            if st.button("+ Add", width="stretch"):
                t = (new_ticker or "").strip().upper()
                if t:
                    add_ticker(t)
                    st.session_state.selected_ticker = t
                    st.rerun()

    # ── MACRO ROW ─────────────────────────────────────────────────────────────
    macro_items = cached_macro()
    m_cols = st.columns(5)
    for i, m in enumerate(macro_items[:5]):
        with m_cols[i]:
            st.metric(label=m.get("name", ""),
                      value=m.get("display_value", "N/A"),
                      delta=m.get("change_display"))

    st.divider()

    # ── LOAD ALL DATA ─────────────────────────────────────────────────────────
    with st.spinner(f"Loading {selected}..."):
        data = cached_market_data(selected)
        sent = cached_sentiment(selected)
        tech = cached_technicals(selected)

    fund     = data.get("fundamentals") or {}
    price_yf = data.get("price_yf")     or {}
    quote    = data.get("quote")         or {}

    live_price = data.get("live_price")
    change_pct = data.get("change_pct")
    prev_close = price_yf.get("prev_close") or quote.get("prev_close")
    short_name = fund.get("short_name") or selected
    sector     = fund.get("sector") or ""
    industry   = fund.get("industry") or ""

    price_str = _fmt_price(live_price)
    chg_str   = _fmt_pct(change_pct)
    prev_str  = _fmt_price(prev_close)
    chg_c     = _chg_color(change_pct)
    sub_str   = " — ".join(filter(None, [short_name, sector, industry]))

    # ── PRICE HEADER ──────────────────────────────────────────────────────────
    ph_left, ph_right = st.columns([6, 2])
    with ph_left:
        st.markdown(
            f'<div style="line-height:1.2">'
            f'<span style="font-size:2.2rem;font-weight:bold">{price_str}</span>'
            f'&nbsp;&nbsp;'
            f'<span style="font-size:1.5rem;font-weight:bold;color:{chg_c}">{chg_str}</span>'
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

    # ── FUNDAMENTALS ROW ──────────────────────────────────────────────────────
    f_cols = st.columns(6)
    fund_rows = [
        ("Market Cap",   _fmt_cap(fund.get("market_cap"))),
        ("P/E (Trail.)", _fmt_float(fund.get("pe_ratio"),   2, "x")),
        ("P/E (Fwd)",    _fmt_float(fund.get("forward_pe"), 2, "x")),
        ("52w High",     _fmt_price(fund.get("52w_high"))),
        ("52w Low",      _fmt_price(fund.get("52w_low"))),
        ("Beta",         _fmt_float(fund.get("beta"), 2)),
    ]
    for i, (lbl, val) in enumerate(fund_rows):
        with f_cols[i]:
            st.metric(label=lbl, value=val)

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

    # Signal badges row
    rsi_val  = tech.get("rsi")
    rsi_sig  = tech.get("rsi_signal", "Neutral")
    macd_sig = tech.get("macd_signal", "Neutral")
    bb_sig   = tech.get("bb_signal", "Inside Bands")
    vwap_sig = tech.get("vwap_signal") or "N/A"
    vol_sig  = tech.get("volume_signal", "Normal")
    vol_rat  = tech.get("volume_ratio")
    vwap_val = tech.get("vwap")

    # Pre-compute all display strings
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

    # Anomaly reason
    an_data = cached_anomaly(selected)
    if an_data.get("is_watch"):
        st.markdown(
            f'<div style="background:#ffd70018;border:1px solid {YELLOW};'
            f'border-radius:5px;padding:8px 14px;margin-bottom:12px;'
            f'font-size:0.85rem;color:{YELLOW}">'
            f'🔥 <b>Watch:</b> {an_data.get("reason", "")}'
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
            bars       = intraday
            chart_ttl  = f"{selected} — 1-Min Intraday (Polygon)"
        else:
            bars       = cached_daily_bars(selected)
            chart_ttl  = f"{selected} — Daily 1-Month (yfinance, intraday unavailable)"
        st.plotly_chart(_build_intraday_chart(bars, chart_ttl),
                        width="stretch",
                        config={"displayModeBar": False})

    with tab_daily:
        st.plotly_chart(_build_daily_ta_chart(tech, selected),
                        width="stretch",
                        config={"displayModeBar": False})

        # S/R level table below chart
        sr = tech.get("support_resistance", [])
        cur_p = tech.get("current_price") or live_price
        if sr and cur_p:
            st.markdown(
                f'<span style="color:{NEUTRAL};font-size:0.75rem;font-weight:bold">'
                f'SUPPORT / RESISTANCE LEVELS</span>',
                unsafe_allow_html=True,
            )
            sr_html = '<div style="display:flex;gap:10px;margin-top:6px">'
            for lv in sr:
                dist = round((cur_p - lv) / cur_p * 100, 1)
                sign = "+" if dist >= 0 else ""
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

    # ── AUTO-REFRESH ──────────────────────────────────────────────────────────
    if time.time() >= st.session_state.next_refresh:
        st.session_state.next_refresh = time.time() + 60
        st.rerun()

    time.sleep(10)
    st.rerun()


render()
