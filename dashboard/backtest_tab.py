"""
Backtest tab — Streamlit UI component for the backtesting engine.

Single public entry point: render_backtest_tab()

Results are stored in st.session_state (no @st.cache_data — user triggers runs).
Session state keys:
    bt_results  : dict[ticker -> result_dict]
    bt_params   : dict with last run parameters and timestamp
    bt_running  : bool (True while a run is in progress)

Color constants are duplicated here from app.py to avoid circular imports.
"""

import os
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Color constants (duplicated from app.py — no circular import) ────────────────
GREEN   = "#00ff88"
RED     = "#ff4444"
YELLOW  = "#ffd700"
BLUE    = "#4488ff"
NEUTRAL = "#888888"
BG      = "#0e1117"
BG2     = "#1a1d24"


# ── Public entry point ───────────────────────────────────────────────────────────

def render_backtest_tab() -> None:
    """Render the full backtesting tab UI."""

    # ── Session state initialisation ──────────────────────────────────────────
    if 'bt_results' not in st.session_state:
        st.session_state.bt_results = {}
    if 'bt_params' not in st.session_state:
        st.session_state.bt_params = {}
    if 'bt_running' not in st.session_state:
        st.session_state.bt_running = False

    st.markdown("### BACKTESTING ENGINE")
    st.caption(
        "Replays live signal logic on 2 years of daily data  |  "
        "ATR-based stop-loss  |  No look-ahead bias  |  "
        "VWAP = daily typical-price proxy (not intraday)"
    )

    # ── Controls row ──────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6, c7 = st.columns([3, 1.5, 1, 1, 1, 1, 1])
    with c1:
        tickers_raw = st.text_input(
            "Ticker(s) — comma-separated", value="AAPL", key="bt_tickers"
        )
    with c2:
        capital = st.number_input(
            "Initial Capital ($)", min_value=100, value=1000, step=100, key="bt_capital"
        )
    with c3:
        risk_pct = st.number_input(
            "Risk %", min_value=0.1, max_value=10.0, value=2.0, step=0.5, key="bt_risk"
        )
    with c4:
        rr = st.number_input(
            "R:R", min_value=0.5, max_value=10.0, value=2.0, step=0.5, key="bt_rr"
        )
    with c5:
        hold = st.number_input(
            "Hold Days", min_value=1, max_value=30, value=5, step=1, key="bt_hold"
        )
    with c6:
        st.markdown("<br>", unsafe_allow_html=True)
        run_clicked = st.button("▶ Run", type="primary", key="bt_run")
    with c7:
        st.markdown("<br>", unsafe_allow_html=True)
        clear_clicked = st.button("✕ Clear", key="bt_clear")

    # ── Clear ─────────────────────────────────────────────────────────────────
    if clear_clicked:
        st.session_state.bt_results = {}
        st.session_state.bt_params  = {}
        st.rerun()

    # ── Run ───────────────────────────────────────────────────────────────────
    if run_clicked:
        tickers = [t.strip().upper() for t in tickers_raw.split(',') if t.strip()]
        if not tickers:
            st.error("Enter at least one ticker symbol.")
        else:
            from data.backtester import run_backtest  # lazy import — keeps tab fast
            results  = {}
            progress = st.progress(0, text="Starting backtest...")
            for k, t in enumerate(tickers):
                progress.progress(k / len(tickers), text=f"Backtesting {t}...")
                results[t] = run_backtest(
                    t,
                    float(capital),
                    float(risk_pct),
                    float(rr),
                    int(hold),
                )
            progress.progress(1.0, text="Done")
            progress.empty()
            st.session_state.bt_results = results
            st.session_state.bt_params  = {
                'tickers':  tickers,
                'capital':  capital,
                'risk_pct': risk_pct,
                'rr':       rr,
                'hold':     hold,
                'run_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }

    results = st.session_state.bt_results
    params  = st.session_state.bt_params

    # ── No results yet ────────────────────────────────────────────────────────
    if not results:
        st.info(
            "Configure parameters above and click **▶ Run** to start a backtest. "
            "Multiple tickers can be entered comma-separated (e.g. AAPL, NVDA, TSLA)."
        )
        return

    # ── Status bar ────────────────────────────────────────────────────────────
    st.caption(
        f"Last run: {params.get('run_time', '?')}  |  "
        f"Capital: ${params.get('capital', 0):,}  |  "
        f"Risk: {params.get('risk_pct', 0)}%  |  "
        f"R:R {params.get('rr', 0)}  |  "
        f"Hold: {params.get('hold', 0)}d"
    )

    # ── Results per ticker ────────────────────────────────────────────────────
    for ticker, result in results.items():
        st.markdown("---")
        st.markdown(f"#### {ticker}")

        if result.get('error') and not result.get('trades'):
            st.error(f"Error: {result['error']}")
            continue

        _render_summary_cards(result)
        st.markdown("<br>", unsafe_allow_html=True)

        col_eq, col_sig = st.columns([2, 1])
        with col_eq:
            fig = _build_equity_chart(result, ticker)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        with col_sig:
            fig = _build_signal_chart(result, ticker)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        _render_trades_table(result, ticker)

    # ── Multi-ticker comparison ───────────────────────────────────────────────
    if len(results) > 1:
        st.markdown("---")
        st.markdown("#### Multi-Ticker Comparison")
        fig = _build_comparison_chart(results)
        if fig:
            st.plotly_chart(fig, use_container_width=True)


# ── Private helpers ──────────────────────────────────────────────────────────────

def _card(label: str, value: str, color: str) -> str:
    return (
        f"<div style='background:{BG2};border:1px solid #2a2d35;"
        f"border-radius:5px;padding:8px 12px;'>"
        f"<div style='color:{NEUTRAL};font-size:0.72rem;'>{label}</div>"
        f"<div style='color:{color};font-size:1.15rem;font-weight:bold;'>{value}</div>"
        f"</div>"
    )


def _render_summary_cards(result: dict) -> None:
    cols = st.columns(6)

    tr = result.get('total_return_pct') or 0
    cols[0].markdown(
        _card("Total Return", f"{tr:+.1f}%", GREEN if tr >= 0 else RED),
        unsafe_allow_html=True,
    )

    wr = result.get('win_rate') or 0
    cols[1].markdown(
        _card("Win Rate", f"{wr:.1f}%", GREEN if wr >= 50 else RED),
        unsafe_allow_html=True,
    )

    pf     = result.get('profit_factor') or 0
    pf_str = "inf" if pf >= 9999 else f"{pf:.2f}"
    cols[2].markdown(
        _card("Profit Factor", pf_str, GREEN if pf >= 1 else RED),
        unsafe_allow_html=True,
    )

    dd     = result.get('max_drawdown_pct') or 0
    dd_col = RED if dd < -15 else YELLOW if dd < -5 else GREEN
    cols[3].markdown(
        _card("Max Drawdown", f"{dd:.1f}%", dd_col),
        unsafe_allow_html=True,
    )

    sr     = result.get('sharpe_ratio') or 0
    sr_col = GREEN if sr >= 1.0 else YELLOW if sr >= 0 else RED
    cols[4].markdown(
        _card("Sharpe", f"{sr:.2f}", sr_col),
        unsafe_allow_html=True,
    )

    nt = result.get('total_trades') or 0
    cols[5].markdown(
        _card("Trades", str(nt), BLUE),
        unsafe_allow_html=True,
    )


def _build_equity_chart(result: dict, ticker: str):
    ec = result.get('equity_curve', [])
    if not ec:
        return None

    dates  = [x[0] for x in ec]
    values = [x[1] for x in ec]
    peak   = pd.Series(values).cummax().tolist()

    fig = go.Figure()

    # Peak trace (invisible — used as fill reference)
    fig.add_trace(go.Scatter(
        x=dates, y=peak,
        mode='lines', line=dict(width=0),
        showlegend=False, hoverinfo='skip',
    ))

    # Drawdown fill (between equity line and peak)
    fig.add_trace(go.Scatter(
        x=dates, y=values,
        mode='lines', line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(255,68,68,0.18)',
        showlegend=False, hoverinfo='skip',
    ))

    # Equity line
    fig.add_trace(go.Scatter(
        x=dates, y=values,
        mode='lines',
        line=dict(color=GREEN, width=2),
        name='Portfolio Value',
        hovertemplate='%{x}<br>$%{y:,.2f}<extra></extra>',
    ))

    # Initial capital reference line
    init = result.get('initial_capital') or (values[0] if values else 1000)
    fig.add_hline(
        y=init,
        line_dash='dash', line_color=NEUTRAL,
        annotation_text='Start',
        annotation_position='top left',
        annotation_font_color=NEUTRAL,
    )

    fig.update_layout(
        title=dict(text=f"{ticker} — Equity Curve", font=dict(size=12, color='#fafafa')),
        height=280,
        paper_bgcolor=BG, plot_bgcolor=BG2,
        font=dict(color='#fafafa', size=10),
        xaxis=dict(gridcolor='#2a2d35', showgrid=True, tickangle=-30),
        yaxis=dict(gridcolor='#2a2d35', showgrid=True, tickprefix='$'),
        margin=dict(l=10, r=10, t=40, b=30),
        showlegend=False,
    )
    return fig


def _build_signal_chart(result: dict, ticker: str):
    sh = result.get('signal_history', [])
    if not sh:
        return None

    dates  = [s['date']         for s in sh]
    counts = [s['signal_count'] for s in sh]
    colors = [GREEN if s['watch_flag'] else NEUTRAL for s in sh]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dates, y=counts,
        marker_color=colors,
        name='Signal Count',
        hovertemplate='%{x}<br>Signals: %{y}<extra></extra>',
    ))

    # Watch threshold reference
    fig.add_hline(
        y=WATCH_THRESHOLD,
        line_dash='dot', line_color=YELLOW,
        annotation_text='Watch (>=3)',
        annotation_position='top left',
        annotation_font_color=YELLOW,
    )

    fig.update_layout(
        title=dict(text=f"{ticker} — Daily Signals", font=dict(size=12, color='#fafafa')),
        height=280,
        paper_bgcolor=BG, plot_bgcolor=BG2,
        font=dict(color='#fafafa', size=10),
        xaxis=dict(gridcolor='#2a2d35', showgrid=False, tickangle=-30),
        yaxis=dict(gridcolor='#2a2d35', showgrid=True, title='Signal Count'),
        margin=dict(l=10, r=10, t=40, b=30),
        showlegend=False,
        bargap=0.1,
    )
    return fig


def _render_trades_table(result: dict, ticker: str) -> None:
    trades = result.get('trades', [])
    if not trades:
        st.caption("No trades generated for this ticker in the backtest period.")
        return

    st.markdown("**Trade Log**")
    df = pd.DataFrame(trades)
    col_order = [
        'entry_date', 'exit_date', 'side', 'entry_price',
        'exit_price', 'shares', 'pnl', 'pnl_pct',
        'outcome', 'signal_count', 'signals',
    ]
    df = df[[c for c in col_order if c in df.columns]]
    st.dataframe(df, use_container_width=True, height=260)

    if st.button(f"Export {ticker} CSV", key=f"bt_export_{ticker}"):
        os.makedirs("backtest_results", exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"backtest_results/{ticker}_{ts}.csv"
        df.to_csv(path, index=False)
        st.success(f"Saved to {path}")


def _build_comparison_chart(results: dict):
    rows = [
        {'ticker': t, 'return': r.get('total_return_pct') or 0}
        for t, r in results.items()
        if not (r.get('error') and not r.get('trades'))
    ]
    if not rows:
        return None

    rows.sort(key=lambda x: x['return'])   # ascending → best at top in h-bar
    tickers = [r['ticker'] for r in rows]
    returns = [r['return']  for r in rows]
    colors  = [GREEN if v >= 0 else RED for v in returns]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=returns, y=tickers,
        orientation='h',
        marker_color=colors,
        text=[f"{v:+.1f}%" for v in returns],
        textposition='outside',
        hovertemplate='%{y}: %{x:.2f}%<extra></extra>',
    ))
    fig.add_vline(x=0, line_color=NEUTRAL, line_width=1)

    fig.update_layout(
        title=dict(text="Total Return Comparison", font=dict(size=12, color='#fafafa')),
        height=max(220, 50 * len(rows) + 80),
        paper_bgcolor=BG, plot_bgcolor=BG2,
        font=dict(color='#fafafa', size=10),
        xaxis=dict(
            gridcolor='#2a2d35', showgrid=True,
            ticksuffix='%', title='Total Return %',
        ),
        yaxis=dict(gridcolor='#2a2d35', showgrid=False),
        margin=dict(l=10, r=60, t=40, b=30),
        showlegend=False,
    )
    return fig


# Module-level constant used in _build_signal_chart
WATCH_THRESHOLD = 3
