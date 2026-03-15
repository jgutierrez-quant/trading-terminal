"""
Performance Analytics tab — detailed trade performance analysis.

Single public entry point: render_performance_tab()

Session state keys:
    perf_date_range  : tuple (start, end) date filter
    perf_side_filter : str ('All' | 'buy' | 'sell')
"""

from datetime import datetime

import plotly.graph_objects as go
import streamlit as st

from utils.config import GREEN, RED, NEUTRAL, BG, BG2
from utils.trade_logger import get_performance_summary, get_all_trades


# ── Card helper (matches backtest_tab.py style) ─────────────────────────────

def _card(label: str, value: str, color: str) -> str:
    return (
        f"<div style='background:{BG2};border:1px solid #2a2d35;"
        f"border-radius:5px;padding:8px 12px;'>"
        f"<div style='color:{NEUTRAL};font-size:0.72rem;'>{label}</div>"
        f"<div style='color:{color};font-size:1.1rem;font-weight:bold;'>{value}</div>"
        f"</div>"
    )


# ── Public entry point ───────────────────────────────────────────────────────

def render_performance_tab() -> None:
    """Render the full performance analytics tab UI."""

    st.markdown(
        f'<span style="font-size:1.1rem;font-weight:bold;color:{GREEN}">PERFORMANCE ANALYTICS</span>',
        unsafe_allow_html=True,
    )
    st.caption("Detailed trade performance metrics, charts, and history.")

    # ── Load data ────────────────────────────────────────────────────────
    perf = get_performance_summary()
    trades = get_all_trades(limit=500)

    if perf.get("error") and perf.get("total_trades") is None:
        st.error(f"Database error: {perf['error']}")
        return

    if perf.get("total_trades", 0) == 0:
        st.info("No trades logged yet. Execute trades from Trade Coach or the Portfolio tab to see analytics here.")
        return

    closed = [t for t in trades if t["outcome"] != "open" and t["pnl"] is not None]
    winners = [t for t in closed if t["pnl"] > 0]
    losers = [t for t in closed if t["pnl"] < 0]

    # ── Computed metrics ─────────────────────────────────────────────────
    avg_winner = round(sum(t["pnl"] for t in winners) / len(winners), 2) if winners else 0.0
    avg_loser = round(sum(t["pnl"] for t in losers) / len(losers), 2) if losers else 0.0
    gross_profit = sum(t["pnl"] for t in winners)
    gross_loss = abs(sum(t["pnl"] for t in losers))
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

    # ── Summary cards (8 cards) ──────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c5, c6, c7, c8 = st.columns(4)

    total_trades = perf.get("total_trades", 0)
    win_rate = perf.get("win_rate")
    total_pnl = perf.get("total_pnl", 0)
    best = perf.get("best_trade")
    worst = perf.get("worst_trade")

    c1.markdown(_card("Total Trades", str(total_trades), "#fafafa"), unsafe_allow_html=True)
    c2.markdown(_card("Win Rate", f"{win_rate:.1f}%" if win_rate is not None else "N/A",
                       GREEN if (win_rate or 0) >= 50 else RED), unsafe_allow_html=True)
    c3.markdown(_card("Total P&L", f"{'+'if total_pnl>=0 else ''}${total_pnl:,.2f}",
                       GREEN if total_pnl >= 0 else RED), unsafe_allow_html=True)
    c4.markdown(_card("Avg Winner", f"+${avg_winner:,.2f}" if avg_winner else "N/A",
                       GREEN), unsafe_allow_html=True)
    c5.markdown(_card("Avg Loser", f"${avg_loser:,.2f}" if avg_loser else "N/A",
                       RED), unsafe_allow_html=True)
    c6.markdown(_card("Profit Factor", f"{profit_factor:.2f}" if profit_factor != float("inf") else "INF",
                       GREEN if profit_factor >= 1.0 else RED), unsafe_allow_html=True)
    c7.markdown(_card("Best Trade", f"+${best:,.2f}" if best is not None else "N/A",
                       GREEN), unsafe_allow_html=True)
    c8.markdown(_card("Worst Trade", f"${worst:,.2f}" if worst is not None else "N/A",
                       RED), unsafe_allow_html=True)

    st.divider()

    # ── Cumulative P&L Curve ─────────────────────────────────────────────
    if closed:
        closed_sorted = sorted(closed, key=lambda x: x.get("exit_date") or t.get("entry_date") or "")
        cumulative = []
        running = 0.0
        dates = []
        for t in closed_sorted:
            running += t["pnl"]
            cumulative.append(round(running, 2))
            dates.append(t.get("exit_date") or t.get("entry_date") or f"#{t['id']}")

        fig = go.Figure()

        # Green fill above zero
        cum_above = [max(v, 0) for v in cumulative]
        cum_below = [min(v, 0) for v in cumulative]

        fig.add_trace(go.Scatter(
            x=dates, y=cum_above, fill="tozeroy",
            fillcolor="rgba(0,255,136,0.12)", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=cum_below, fill="tozeroy",
            fillcolor="rgba(255,68,68,0.12)", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))

        # Main line
        fig.add_trace(go.Scatter(
            x=dates, y=cumulative, mode="lines+markers",
            line=dict(color=GREEN if cumulative[-1] >= 0 else RED, width=2),
            marker=dict(
                color=[GREEN if v >= 0 else RED for v in cumulative],
                size=5,
            ),
            hovertemplate="Date: %{x}<br>Cumulative P&L: $%{y:,.2f}<extra></extra>",
            name="Cumulative P&L",
        ))

        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color=NEUTRAL, line_width=1)

        fig.update_layout(
            title="Cumulative P&L",
            height=320,
            margin=dict(l=40, r=20, t=40, b=40),
            paper_bgcolor=BG,
            plot_bgcolor=BG,
            font=dict(color="#fafafa", family="monospace"),
            xaxis=dict(gridcolor="#2a2d35", showgrid=True),
            yaxis=dict(gridcolor="#2a2d35", showgrid=True, title="P&L ($)"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.divider()

    # ── Two-column charts ────────────────────────────────────────────────
    if closed:
        chart_left, chart_right = st.columns(2)

        # Left: Win/Loss Distribution Pie
        with chart_left:
            wins_n = len(winners)
            losses_n = len(losers)
            be_n = len([t for t in closed if t["pnl"] == 0])

            pie_labels = []
            pie_values = []
            pie_colors = []
            if wins_n:
                pie_labels.append("Wins")
                pie_values.append(wins_n)
                pie_colors.append(GREEN)
            if losses_n:
                pie_labels.append("Losses")
                pie_values.append(losses_n)
                pie_colors.append(RED)
            if be_n:
                pie_labels.append("Breakeven")
                pie_values.append(be_n)
                pie_colors.append(NEUTRAL)

            fig_pie = go.Figure(data=[go.Pie(
                labels=pie_labels, values=pie_values,
                marker=dict(colors=pie_colors),
                hole=0.4,
                textinfo="label+value",
                textfont=dict(color="#fafafa", size=12),
            )])
            fig_pie.update_layout(
                title="Win / Loss Distribution",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor=BG,
                plot_bgcolor=BG,
                font=dict(color="#fafafa", family="monospace"),
                showlegend=False,
            )
            st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

        # Right: P&L by Ticker (horizontal bar)
        with chart_right:
            ticker_pnl = {}
            for t in closed:
                ticker_pnl[t["ticker"]] = ticker_pnl.get(t["ticker"], 0) + t["pnl"]

            sorted_tickers = sorted(ticker_pnl.items(), key=lambda x: x[1])
            tickers_list = [x[0] for x in sorted_tickers]
            pnl_list = [round(x[1], 2) for x in sorted_tickers]
            bar_colors = [GREEN if v >= 0 else RED for v in pnl_list]

            fig_bar = go.Figure(data=[go.Bar(
                x=pnl_list, y=tickers_list, orientation="h",
                marker=dict(color=bar_colors),
                hovertemplate="%{y}: $%{x:,.2f}<extra></extra>",
            )])
            fig_bar.update_layout(
                title="P&L by Ticker",
                height=300,
                margin=dict(l=60, r=20, t=40, b=20),
                paper_bgcolor=BG,
                plot_bgcolor=BG,
                font=dict(color="#fafafa", family="monospace"),
                xaxis=dict(gridcolor="#2a2d35", title="P&L ($)"),
                yaxis=dict(gridcolor="#2a2d35"),
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    st.divider()

    # ── Closed Trades Table ──────────────────────────────────────────────
    st.markdown(
        f'<span style="font-size:0.85rem;font-weight:bold;color:{NEUTRAL}">CLOSED TRADES</span>',
        unsafe_allow_html=True,
    )

    if not closed:
        st.caption("No closed trades yet.")
    else:
        import pandas as pd

        rows = []
        for t in sorted(closed, key=lambda x: x.get("exit_date") or "", reverse=True):
            rows.append({
                "Date": t.get("exit_date", "")[:10] if t.get("exit_date") else "",
                "Ticker": t["ticker"],
                "Side": t["side"].upper(),
                "Entry": f"${t['entry_price']:,.2f}" if t.get("entry_price") else "",
                "Exit": f"${t['exit_price']:,.2f}" if t.get("exit_price") else "",
                "Qty": t.get("quantity", 0),
                "P&L ($)": round(t["pnl"], 2),
                "P&L (%)": round(t.get("pnl_percent", 0), 2),
                "Outcome": (t.get("outcome") or "").upper(),
            })

        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "P&L ($)": st.column_config.NumberColumn(format="$%.2f"),
                "P&L (%)": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )
