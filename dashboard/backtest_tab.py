"""
Backtest tab — Streamlit UI component for the backtesting engine.

Single public entry point: render_backtest_tab()

Stage 9 additions:
    - Quality score threshold slider (0-100, default 60)
    - Hold days input (default 10)
    - Exit method breakdown pie chart (Target / Stop / Timeout / Trend %)
    - Avg hold days added to summary cards (7 cards)
    - Automatic threshold 3 vs 4 side-by-side comparison

Session state keys:
    bt_results      : dict[ticker -> result_dict]  (main run)
    bt_comparison   : dict[ticker -> {thr3, thr4}] (comparison run)
    bt_params       : dict with last run parameters
    bt_running      : bool

Color constants duplicated from app.py — no circular imports.
"""

import os
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Color constants + thresholds ─────────────────────────────────────────────
from utils.config import GREEN, RED, YELLOW, BLUE, NEUTRAL, BG, BG2, WATCH_THRESHOLD


# ── Public entry point ────────────────────────────────────────────────────────

def render_backtest_tab() -> None:
    """Render the full backtesting tab UI."""

    # Session state init
    for key, default in [
        ('bt_results', {}), ('bt_comparison', {}),
        ('bt_params', {}),  ('bt_running', False),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    st.markdown("### BACKTESTING ENGINE")
    st.caption(
        "Replays live signal logic on 2y daily data  |  ATR stop  |  "
        "Direction filter (SMA50)  |  MA20 trend-invalidation exit  |  "
        "Earnings avoidance  |  VWAP = daily proxy"
    )

    # ── Controls ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns([3, 1.5, 1, 1, 1])
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
            "Hold Days", min_value=1, max_value=30, value=10, step=1, key="bt_hold"
        )

    c6, c7, c8 = st.columns([3, 1, 1])
    with c6:
        quality_thr = st.slider(
            "Min Quality Score", min_value=0, max_value=100,
            value=60, step=5, key="bt_quality",
            help="Only backtest trades where the signal quality score >= this threshold (0 = no filter)",
        )
    with c7:
        run_clicked = st.button("▶ Run", type="primary", key="bt_run")
    with c8:
        clear_clicked = st.button("✕ Clear", key="bt_clear")

    if clear_clicked:
        st.session_state.bt_results    = {}
        st.session_state.bt_comparison = {}
        st.session_state.bt_params     = {}
        st.rerun()

    if run_clicked:
        tickers = [t.strip().upper() for t in tickers_raw.split(',') if t.strip()]
        if not tickers:
            st.error("Enter at least one ticker symbol.")
        else:
            from data.backtester import run_backtest
            results    = {}
            comparison = {}
            n_runs     = len(tickers) * 3   # main + thr3 + thr4 comparison
            progress   = st.progress(0, text="Starting backtest...")
            run_idx    = 0

            for t in tickers:
                progress.progress(run_idx / n_runs, text=f"Backtesting {t} (main run)...")
                results[t] = run_backtest(
                    t, float(capital), float(risk_pct), float(rr),
                    int(hold), int(quality_thr),
                    watch_threshold=WATCH_THRESHOLD, check_earnings=True,
                )
                run_idx += 1

                progress.progress(run_idx / n_runs, text=f"{t}: threshold 3 comparison...")
                thr3 = run_backtest(
                    t, float(capital), float(risk_pct), float(rr),
                    int(hold), quality_threshold=0,
                    watch_threshold=3, check_earnings=False,
                )
                run_idx += 1

                progress.progress(run_idx / n_runs, text=f"{t}: threshold 4 comparison...")
                thr4 = run_backtest(
                    t, float(capital), float(risk_pct), float(rr),
                    int(hold), quality_threshold=0,
                    watch_threshold=4, check_earnings=False,
                )
                run_idx += 1
                comparison[t] = {'thr3': thr3, 'thr4': thr4}

            progress.progress(1.0, text="Done")
            progress.empty()
            st.session_state.bt_results    = results
            st.session_state.bt_comparison = comparison
            st.session_state.bt_params     = {
                'tickers':     tickers,
                'capital':     capital,
                'risk_pct':    risk_pct,
                'rr':          rr,
                'hold':        hold,
                'quality_thr': quality_thr,
                'run_time':    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }

    results    = st.session_state.bt_results
    comparison = st.session_state.bt_comparison
    params     = st.session_state.bt_params

    if not results:
        st.info(
            "Configure parameters above and click **▶ Run** to start a backtest.\n\n"
            "Multiple tickers can be entered comma-separated (e.g. AAPL, NVDA, TSLA)."
        )
        return

    # ── Status bar ────────────────────────────────────────────────────────────
    st.caption(
        f"Last run: {params.get('run_time', '?')}  |  "
        f"Capital: ${params.get('capital', 0):,}  |  "
        f"Risk: {params.get('risk_pct', 0)}%  |  "
        f"R:R {params.get('rr', 0)}  |  "
        f"Hold: {params.get('hold', 0)}d  |  "
        f"Min Quality: {params.get('quality_thr', 60)}"
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

        # Exit method pie chart
        exit_col, _ = st.columns([1, 1])
        with exit_col:
            fig = _build_exit_pie(result, ticker)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        _render_trades_table(result, ticker)

    # ── Threshold 3 vs 4 comparison ───────────────────────────────────────────
    if comparison:
        st.markdown("---")
        st.markdown("#### Threshold Comparison: 3 Signals vs 4 Signals")
        st.caption(
            "Both runs use quality_threshold=0 and no earnings avoidance "
            "to isolate the effect of raising the signal threshold."
        )
        _render_threshold_comparison(comparison)

    # ── Multi-ticker total return comparison ──────────────────────────────────
    if len(results) > 1:
        st.markdown("---")
        st.markdown("#### Multi-Ticker Comparison")
        fig = _build_comparison_chart(results)
        if fig:
            st.plotly_chart(fig, use_container_width=True)


# ── Private helpers ───────────────────────────────────────────────────────────

def _card(label: str, value: str, color: str) -> str:
    return (
        f"<div style='background:{BG2};border:1px solid #2a2d35;"
        f"border-radius:5px;padding:8px 12px;'>"
        f"<div style='color:{NEUTRAL};font-size:0.72rem;'>{label}</div>"
        f"<div style='color:{color};font-size:1.1rem;font-weight:bold;'>{value}</div>"
        f"</div>"
    )


def _render_summary_cards(result: dict) -> None:
    cols = st.columns(7)

    tr = result.get('total_return_pct') or 0
    cols[0].markdown(_card("Total Return", f"{tr:+.1f}%", GREEN if tr >= 0 else RED),
                     unsafe_allow_html=True)

    wr = result.get('win_rate') or 0
    cols[1].markdown(_card("Win Rate", f"{wr:.1f}%", GREEN if wr >= 50 else RED),
                     unsafe_allow_html=True)

    pf     = result.get('profit_factor') or 0
    pf_str = "inf" if pf >= 9999 else f"{pf:.2f}"
    cols[2].markdown(_card("Profit Factor", pf_str, GREEN if pf >= 1 else RED),
                     unsafe_allow_html=True)

    dd     = result.get('max_drawdown_pct') or 0
    dd_col = RED if dd < -15 else YELLOW if dd < -5 else GREEN
    cols[3].markdown(_card("Max Drawdown", f"{dd:.1f}%", dd_col),
                     unsafe_allow_html=True)

    sr     = result.get('sharpe_ratio') or 0
    sr_col = GREEN if sr >= 1.0 else YELLOW if sr >= 0 else RED
    cols[4].markdown(_card("Sharpe", f"{sr:.2f}", sr_col),
                     unsafe_allow_html=True)

    ahd = result.get('avg_hold_days') or 0
    cols[5].markdown(_card("Avg Hold Days", f"{ahd:.1f}d", BLUE),
                     unsafe_allow_html=True)

    nt = result.get('total_trades') or 0
    cols[6].markdown(_card("Trades", str(nt), BLUE),
                     unsafe_allow_html=True)


def _build_equity_chart(result: dict, ticker: str):
    ec = result.get('equity_curve', [])
    if not ec:
        return None

    dates  = [x[0] for x in ec]
    values = [x[1] for x in ec]
    peak   = pd.Series(values).cummax().tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=peak,
        mode='lines', line=dict(width=0),
        showlegend=False, hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=values,
        mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(255,68,68,0.18)',
        showlegend=False, hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=values,
        mode='lines', line=dict(color=GREEN, width=2),
        name='Portfolio Value',
        hovertemplate='%{x}<br>$%{y:,.2f}<extra></extra>',
    ))
    init = result.get('initial_capital') or (values[0] if values else 1000)
    fig.add_hline(y=init, line_dash='dash', line_color=NEUTRAL,
                  annotation_text='Start', annotation_position='top left',
                  annotation_font_color=NEUTRAL)
    fig.update_layout(
        title=dict(text=f"{ticker} — Equity Curve", font=dict(size=12, color='#fafafa')),
        height=260, paper_bgcolor=BG, plot_bgcolor=BG2,
        font=dict(color='#fafafa', size=10),
        xaxis=dict(gridcolor='#2a2d35', showgrid=True, tickangle=-30),
        yaxis=dict(gridcolor='#2a2d35', showgrid=True, tickprefix='$'),
        margin=dict(l=10, r=10, t=40, b=30), showlegend=False,
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
        x=dates, y=counts, marker_color=colors,
        name='Signal Count',
        hovertemplate='%{x}<br>Signals: %{y}<extra></extra>',
    ))
    fig.add_hline(y=WATCH_THRESHOLD, line_dash='dot', line_color=YELLOW,
                  annotation_text=f'Watch (>={WATCH_THRESHOLD})',
                  annotation_position='top left',
                  annotation_font_color=YELLOW)
    fig.update_layout(
        title=dict(text=f"{ticker} — Daily Signals", font=dict(size=12, color='#fafafa')),
        height=260, paper_bgcolor=BG, plot_bgcolor=BG2,
        font=dict(color='#fafafa', size=10),
        xaxis=dict(gridcolor='#2a2d35', showgrid=False, tickangle=-30),
        yaxis=dict(gridcolor='#2a2d35', showgrid=True, title='Signal Count'),
        margin=dict(l=10, r=10, t=40, b=30), showlegend=False, bargap=0.1,
    )
    return fig


def _build_exit_pie(result: dict, ticker: str):
    eb = result.get('exit_breakdown', {})
    if not eb or result.get('total_trades', 0) == 0:
        return None

    labels = list(eb.keys())
    values = list(eb.values())
    colors_pie = [GREEN, RED, NEUTRAL, YELLOW]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=colors_pie[:len(labels)]),
        textinfo='label+percent',
        hovertemplate='%{label}: %{value:.1f}%<extra></extra>',
        hole=0.4,
    ))
    fig.update_layout(
        title=dict(text=f"{ticker} — Exit Methods", font=dict(size=12, color='#fafafa')),
        height=240, paper_bgcolor=BG,
        font=dict(color='#fafafa', size=10),
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=True,
        legend=dict(font=dict(size=9), bgcolor=BG2),
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
        'outcome', 'quality_score', 'signal_count', 'signals',
    ]
    df = df[[c for c in col_order if c in df.columns]]
    st.dataframe(df, use_container_width=True, height=260)

    if st.button(f"Export {ticker} CSV", key=f"bt_export_{ticker}"):
        os.makedirs("backtest_results", exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"backtest_results/{ticker}_{ts}.csv"
        df.to_csv(path, index=False)
        st.success(f"Saved to {path}")


def _render_threshold_comparison(comparison: dict) -> None:
    """Show side-by-side win rate and profit factor for threshold 3 vs 4."""
    rows = []
    for ticker, cmp in comparison.items():
        r3 = cmp.get('thr3', {})
        r4 = cmp.get('thr4', {})
        rows.append({
            'Ticker':          ticker,
            'Trades (thr=3)':  r3.get('total_trades', 0),
            'WinRate% (thr=3)': r3.get('win_rate', 0),
            'PF (thr=3)':      r3.get('profit_factor', 0),
            'Return% (thr=3)': r3.get('total_return_pct', 0),
            'Trades (thr=4)':  r4.get('total_trades', 0),
            'WinRate% (thr=4)': r4.get('win_rate', 0),
            'PF (thr=4)':      r4.get('profit_factor', 0),
            'Return% (thr=4)': r4.get('total_return_pct', 0),
        })

    if not rows:
        return

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    # Bar chart: win rate and profit factor side by side
    tickers = [r['Ticker'] for r in rows]
    wr3 = [r['WinRate% (thr=3)'] for r in rows]
    wr4 = [r['WinRate% (thr=4)'] for r in rows]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Win Rate% — Thr 3', x=tickers, y=wr3,
        marker_color=NEUTRAL,
        hovertemplate='%{x} (thr=3): %{y:.1f}%<extra></extra>',
    ))
    fig.add_trace(go.Bar(
        name='Win Rate% — Thr 4', x=tickers, y=wr4,
        marker_color=GREEN,
        hovertemplate='%{x} (thr=4): %{y:.1f}%<extra></extra>',
    ))
    fig.add_hline(y=50, line_dash='dot', line_color=YELLOW,
                  annotation_text='50% break-even', annotation_font_color=YELLOW)
    fig.update_layout(
        title=dict(text="Win Rate: Threshold 3 vs Threshold 4", font=dict(size=12, color='#fafafa')),
        height=260, barmode='group',
        paper_bgcolor=BG, plot_bgcolor=BG2,
        font=dict(color='#fafafa', size=10),
        xaxis=dict(gridcolor='#2a2d35'),
        yaxis=dict(gridcolor='#2a2d35', ticksuffix='%', title='Win Rate %'),
        margin=dict(l=10, r=10, t=40, b=30),
        legend=dict(font=dict(size=9), bgcolor=BG2),
    )
    st.plotly_chart(fig, use_container_width=True)


def _build_comparison_chart(results: dict):
    rows = [
        {'ticker': t, 'return': r.get('total_return_pct') or 0}
        for t, r in results.items()
        if not (r.get('error') and not r.get('trades'))
    ]
    if not rows:
        return None

    rows.sort(key=lambda x: x['return'])
    tickers = [r['ticker'] for r in rows]
    returns = [r['return']  for r in rows]
    colors  = [GREEN if v >= 0 else RED for v in returns]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=returns, y=tickers, orientation='h',
        marker_color=colors,
        text=[f"{v:+.1f}%" for v in returns], textposition='outside',
        hovertemplate='%{y}: %{x:.2f}%<extra></extra>',
    ))
    fig.add_vline(x=0, line_color=NEUTRAL, line_width=1)
    fig.update_layout(
        title=dict(text="Total Return Comparison", font=dict(size=12, color='#fafafa')),
        height=max(220, 50 * len(rows) + 80),
        paper_bgcolor=BG, plot_bgcolor=BG2,
        font=dict(color='#fafafa', size=10),
        xaxis=dict(gridcolor='#2a2d35', showgrid=True, ticksuffix='%', title='Total Return %'),
        yaxis=dict(gridcolor='#2a2d35', showgrid=False),
        margin=dict(l=10, r=60, t=40, b=30), showlegend=False,
    )
    return fig
