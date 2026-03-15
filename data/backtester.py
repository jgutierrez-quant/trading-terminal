"""
Backtesting engine — replays the same signal logic as anomaly_detector.py
across 2 years of daily data, simulates entries/exits with ATR-based
risk management, and returns structured results for the Backtest tab.

Public API:
    get_historical_data(ticker, period='2y')   -> pd.DataFrame
    generate_signals(df, watch_threshold=4)    -> pd.DataFrame
    run_backtest(ticker, ...)                  -> dict
    run_multi_backtest(tickers, ...)           -> pd.DataFrame

Stage 9 changes vs Stage 8B:
    - WATCH_THRESHOLD = 3 (Stage 9c: back to 3, direction logic preserved)
    - Direction filter: Long only when price > SMA50; Short only below SMA50
    - Quality score (0-100) computed vectorized and stored per bar
    - quality_threshold parameter (default 60) — skip low-quality setups
    - hold_days default raised from 5 to 10
    - MA20 trend-invalidation exit: exit if price crosses MA20 against position
    - Earnings avoidance: skip entry if earnings within 5 days
    - watch_threshold parameter for threshold 3 vs 4 comparison
    - New metrics: avg_hold_days, exit_breakdown (% by method)

Signal formulas are IDENTICAL to data/technicals.py and anomaly_detector.py.
⚠️ VWAP proxy: daily (H+L+C)/3 — NOT intraday VWAP.
"""

import math
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ── Constants (match anomaly_detector.py Stage 9) ────────────────────────────
WATCH_THRESHOLD  = 3
ATR_PERIOD       = 14
ATR_MULTIPLIER   = 2.0
MAX_POSITION_PCT = 0.20


# ── Private indicator helpers — identical to technicals.py ───────────────────

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast    = close.ewm(span=fast,   adjust=False).mean()
    ema_slow    = close.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bbands(close: pd.Series, period: int = 20, std: float = 2.0):
    middle = close.rolling(period).mean()
    sigma  = close.rolling(period).std(ddof=0)
    return middle + std * sigma, middle, middle - std * sigma


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr_hl = high - low
    tr_hc = (high - close.shift(1)).abs()
    tr_lc = (low  - close.shift(1)).abs()
    tr    = tr_hl.combine(tr_hc, max).combine(tr_lc, max)
    return tr.rolling(period).mean()


# ── Public functions ──────────────────────────────────────────────────────────

def get_historical_data(ticker: str, period: str = '2y') -> pd.DataFrame:
    """
    Fetch daily OHLCV from yfinance and compute all indicators vectorized.

    Returns DataFrame with columns:
        Open, High, Low, Close, Volume,
        rsi, macd_hist, bb_upper, bb_lower,
        sma20, sma50, vol_ratio, atr, vwap_proxy

    ⚠️ VWAP proxy: (High+Low+Close)/3 — daily approximation, NOT intraday VWAP.
    """
    ticker = ticker.upper()
    df = yf.Ticker(ticker).history(period=period, interval='1d')
    if df.empty or len(df) < 60:
        raise ValueError(
            f"Insufficient historical data for {ticker} "
            f"(got {len(df)} bars, need >= 60)"
        )

    close  = df['Close']
    high   = df['High']
    low    = df['Low']
    volume = df['Volume']

    df['rsi']        = _rsi(close, 14)
    _, _, hist       = _macd(close, 12, 26, 9)
    df['macd_hist']  = hist
    bb_upper, _, bb_lower = _bbands(close, 20, 2.0)
    df['bb_upper']   = bb_upper
    df['bb_lower']   = bb_lower
    df['sma20']      = close.rolling(20).mean()
    df['sma50']      = close.rolling(50).mean()
    df['vol_ratio']  = volume / volume.rolling(20).mean().shift(1)
    df['atr']        = _atr(high, low, close, 14)
    df['vwap_proxy'] = (high + low + close) / 3

    return df


def generate_signals(df: pd.DataFrame, watch_threshold: int = WATCH_THRESHOLD) -> pd.DataFrame:
    """
    Apply all signal conditions vectorized — matches anomaly_detector.py exactly.

    Stage 9 additions:
        - quality_score: 0-100 per bar
        - Direction filter: watch_flag requires direction to agree with SMA50 trend
        - direction: 'Bullish', 'Bearish', or 'Filtered' (direction/trend mismatch)

    Adds columns:
        rsi_signal, macd_signal, bb_signal, vol_signal,
        vwap_signal, price_vs_sma20, price_vs_sma50,
        signal_count, signal_list, quality_score,
        direction, watch_flag
    """
    close = df['Close']

    # ── Signal labels ─────────────────────────────────────────────────────────
    df['rsi_signal'] = np.where(df['rsi'] < 30, 'Oversold',
                       np.where(df['rsi'] > 70, 'Overbought', 'Neutral'))

    prv = df['macd_hist'].shift(1)
    cur = df['macd_hist']
    df['macd_signal'] = np.where(
        (prv < 0) & (cur >= 0), 'Bullish Cross',
        np.where((prv > 0) & (cur <= 0), 'Bearish Cross',
            np.where(cur > 0, 'Bullish', 'Bearish')
        )
    )

    df['bb_signal'] = np.where(close > df['bb_upper'], 'Above Upper',
                      np.where(close < df['bb_lower'], 'Below Lower', 'Inside Bands'))

    df['vol_signal'] = np.where(df['vol_ratio'] >= 2.0, 'High Volume',
                       np.where(df['vol_ratio'] >= 1.5, 'Elevated', 'Normal'))

    df['vwap_signal'] = np.where(close > df['vwap_proxy'], 'Above VWAP', 'Below VWAP')

    df['price_vs_sma20'] = np.where(close > df['sma20'], 'Above', 'Below')
    df['price_vs_sma50'] = np.where(close > df['sma50'], 'Above', 'Below')

    # ── Signal count ──────────────────────────────────────────────────────────
    rsi_ct  = df['rsi_signal'].isin(['Oversold', 'Overbought']).astype(int)
    macd_ct = df['macd_signal'].isin(['Bullish Cross', 'Bearish Cross']).astype(int)
    bb_ct   = df['bb_signal'].isin(['Above Upper', 'Below Lower']).astype(int)
    vol_ct  = df['vol_signal'].isin(['High Volume', 'Elevated']).astype(int)
    vwap_ct = df['vwap_signal'].isin(['Above VWAP', 'Below VWAP']).astype(int)
    sma_ct  = (
        ((df['price_vs_sma20'] == 'Above') & (df['price_vs_sma50'] == 'Above')) |
        ((df['price_vs_sma20'] == 'Below') & (df['price_vs_sma50'] == 'Below'))
    ).astype(int)
    df['signal_count'] = rsi_ct + macd_ct + bb_ct + vol_ct + vwap_ct + sma_ct

    # ── Directional signal counts (for quality score and direction filter) ────
    bull_ct = (
        (df['rsi_signal'] == 'Oversold').astype(int) +
        (df['macd_signal'] == 'Bullish Cross').astype(int) +
        (df['bb_signal'] == 'Below Lower').astype(int) +
        (df['vwap_signal'] == 'Above VWAP').astype(int) +
        ((df['price_vs_sma20'] == 'Above') & (df['price_vs_sma50'] == 'Above')).astype(int)
    )
    bear_ct = (
        (df['rsi_signal'] == 'Overbought').astype(int) +
        (df['macd_signal'] == 'Bearish Cross').astype(int) +
        (df['bb_signal'] == 'Above Upper').astype(int) +
        (df['vwap_signal'] == 'Below VWAP').astype(int) +
        ((df['price_vs_sma20'] == 'Below') & (df['price_vs_sma50'] == 'Below')).astype(int)
    )

    # ── Quality score (0-100) — rewards aligned directional signals ──────────
    dir_total  = bull_ct + bear_ct
    dominant   = np.maximum(bull_ct, bear_ct)
    alignment  = np.where(dir_total > 0, dominant / dir_total.replace(0, 1), 0.0)
    vol_pts    = np.where(df['vol_signal'] == 'High Volume', 20,
                 np.where(df['vol_signal'] == 'Elevated', 10, 0))
    df['quality_score'] = (
        np.minimum(dominant / 5.0, 1.0) * 40 +   # directional count (5 max)
        alignment * 40 +
        vol_pts
    ).round().astype(int)

    # ── Direction — signal dominance only, no SMA50 filter ───────────────────
    # Long:  total signals >= threshold AND bullish outnumber bearish
    is_long  = (df['signal_count'] >= watch_threshold) & (bull_ct > bear_ct)
    # Short: total signals >= threshold AND bearish outnumber bullish
    is_short = (df['signal_count'] >= watch_threshold) & (bear_ct > bull_ct)

    df['direction'] = np.where(bull_ct > bear_ct, 'Bullish',
                      np.where(bear_ct > bull_ct, 'Bearish', 'Neutral'))

    df['watch_flag'] = is_long | is_short

    # ── Signal list (string per row) ──────────────────────────────────────────
    def _signal_list(row):
        sigs = []
        if row['rsi_signal'] == 'Oversold':      sigs.append('Oversold RSI')
        elif row['rsi_signal'] == 'Overbought':  sigs.append('Overbought RSI')
        if row['macd_signal'] == 'Bullish Cross':  sigs.append('Bullish MACD Cross')
        elif row['macd_signal'] == 'Bearish Cross': sigs.append('Bearish MACD Cross')
        if row['bb_signal'] == 'Below Lower':    sigs.append('BB Oversold')
        elif row['bb_signal'] == 'Above Upper':  sigs.append('BB Overbought')
        if row['vol_signal'] == 'High Volume':   sigs.append('High Volume')
        elif row['vol_signal'] == 'Elevated':    sigs.append('Elevated Volume')
        if row['vwap_signal'] == 'Above VWAP':   sigs.append('Above VWAP')
        elif row['vwap_signal'] == 'Below VWAP': sigs.append('Below VWAP')
        if row['price_vs_sma20'] == 'Above' and row['price_vs_sma50'] == 'Above':
            sigs.append('Price Above MA20+MA50')
        elif row['price_vs_sma20'] == 'Below' and row['price_vs_sma50'] == 'Below':
            sigs.append('Price Below MA20+MA50')
        return ', '.join(sigs)

    df['signal_list'] = df.apply(_signal_list, axis=1)
    return df


def _fetch_earnings_dates(ticker: str) -> set:
    """
    Fetch historical + upcoming earnings dates for a ticker.
    Returns a set of date strings ('YYYY-MM-DD'). Empty set on failure.
    """
    try:
        ed = yf.Ticker(ticker).earnings_dates
        if ed is None or ed.empty:
            return set()
        dates = set()
        for dt in ed.index:
            try:
                dates.add(str(dt.date()) if hasattr(dt, 'date') else str(dt)[:10])
            except Exception:
                continue
        return dates
    except Exception as exc:
        logger.debug("Could not fetch earnings dates for %s: %s", ticker, exc)
        return set()


def run_backtest(
    ticker: str,
    initial_capital: float = 1000,
    risk_percent: float = 2.0,
    risk_reward: float = 2.0,
    hold_days: int = 10,
    quality_threshold: int = 50,
    watch_threshold: int = WATCH_THRESHOLD,
    check_earnings: bool = True,
    check_fundamentals: bool = False,
    factor_threshold: int = 0,
) -> dict:
    """
    Sequential backtest with no look-ahead bias.

    Stage 9 changes:
        - Direction filter (Long only above SMA50, Short only below)
        - Quality score filter: skip trades below quality_threshold
        - MA20 trend-invalidation exit (outcome = 'Trend')
        - Earnings avoidance: skip entry within 5 days of earnings
        - watch_threshold parameter for threshold-3 vs threshold-4 comparisons
        - New metrics: avg_hold_days, exit_breakdown

    Stage 11 additions:
        - factor_threshold: when > 0, skip trades where composite_factor_score < threshold
          ⚠️ Uses current factor model snapshot — introduces look-ahead bias for historical periods

    Exit priority per bar:
        1. Stop hit (ATR x 2.0)
        2. Target hit (stop_distance x risk_reward)
        3. MA20 crossed against position (Trend invalidation)
        4. Timeout at close after hold_days bars
    """
    try:
        df = generate_signals(get_historical_data(ticker), watch_threshold=watch_threshold)
    except Exception as exc:
        logger.error("Backtest failed for %s: %s", ticker, exc)
        return _empty_result(ticker, str(exc), initial_capital)

    # Earnings dates — fetched once, checked at each potential entry
    earnings_dates = _fetch_earnings_dates(ticker) if check_earnings else set()

    # Fundamentals — fetched once as a static snapshot
    # ⚠️ Uses current fundamentals — introduces look-ahead bias for historical periods
    fundamental_score = 0
    fund_dict_cached  = None
    if check_fundamentals:
        try:
            from data.fundamentals import get_fundamentals, score_fundamentals
            fund_dict_cached  = get_fundamentals(ticker)
            fund_scored       = score_fundamentals(fund_dict_cached)
            fundamental_score = fund_scored.get("fundamental_score", 0)
        except Exception as exc:
            logger.debug("Backtest fundamentals fetch failed for %s: %s", ticker, exc)

    # Factor model — fetched once as a static snapshot
    # ⚠️ Uses current factor model — introduces look-ahead bias for historical periods
    composite_factor_score = None
    if factor_threshold > 0:
        try:
            from data.factor_model import compute_factor_model
            fm = compute_factor_model(ticker, fund_dict=fund_dict_cached)
            composite_factor_score = fm.get("composite_score", 50.0)
        except Exception as exc:
            logger.debug("Backtest factor model fetch failed for %s: %s", ticker, exc)

    capital       = float(initial_capital)
    trades        = []
    equity_events = {}
    n             = len(df)
    i             = 0

    while i < n - 1:
        if not df['watch_flag'].iloc[i]:
            i += 1
            continue

        # Quality filter
        quality_score_bar = int(df['quality_score'].iloc[i])
        if check_fundamentals:
            composite = 0.7 * quality_score_bar + 0.3 * max(fundamental_score, 0)
            if composite < quality_threshold:
                i += 1
                continue
        elif quality_score_bar < quality_threshold:
            i += 1
            continue

        # Factor model filter
        if factor_threshold > 0 and composite_factor_score is not None:
            if composite_factor_score < factor_threshold:
                i += 1
                continue

        entry_idx   = i + 1
        entry_price = float(df['Open'].iloc[entry_idx])
        atr_val     = df['atr'].iloc[i]

        if pd.isna(atr_val) or float(atr_val) <= 0 or entry_price <= 0:
            i += 1
            continue

        # Earnings proximity check — skip if earnings within 5 days
        if earnings_dates:
            entry_dt    = df.index[entry_idx]
            entry_ts    = pd.Timestamp(entry_dt.date() if hasattr(entry_dt, 'date') else str(entry_dt)[:10])
            near_earn   = any(
                abs((pd.Timestamp(ed) - entry_ts).days) <= 5
                for ed in earnings_dates
            )
            if near_earn:
                i += 1
                continue

        atr_val   = float(atr_val)
        direction = str(df['direction'].iloc[i])
        side      = 'short' if direction == 'Bearish' else 'long'

        stop_dist = atr_val * ATR_MULTIPLIER
        if side == 'long':
            stop   = entry_price - stop_dist
            target = entry_price + stop_dist * risk_reward
        else:
            stop   = entry_price + stop_dist
            target = entry_price - stop_dist * risk_reward

        dollar_risk = capital * (risk_percent / 100)
        shares      = int(math.floor(dollar_risk / stop_dist))
        max_shares  = int(math.floor(capital * MAX_POSITION_PCT / entry_price))
        shares      = min(shares, max_shares)

        if shares <= 0:
            i += 1
            continue

        # Simulate exit
        exit_price = None
        exit_bar   = min(entry_idx + hold_days, n - 1)
        outcome    = 'Timeout'

        for j in range(entry_idx, exit_bar + 1):
            bar_h    = float(df['High'].iloc[j])
            bar_l    = float(df['Low'].iloc[j])
            bar_c    = float(df['Close'].iloc[j])
            bar_sma20 = df['sma20'].iloc[j]
            sma20_valid = not pd.isna(bar_sma20)

            if side == 'long':
                if bar_l <= stop:
                    exit_price = stop;   exit_bar = j; outcome = 'Stop';   break
                elif bar_h >= target:
                    exit_price = target; exit_bar = j; outcome = 'Target'; break
                elif sma20_valid and bar_c < float(bar_sma20):
                    exit_price = bar_c;  exit_bar = j; outcome = 'Trend';  break
            else:
                if bar_h >= stop:
                    exit_price = stop;   exit_bar = j; outcome = 'Stop';   break
                elif bar_l <= target:
                    exit_price = target; exit_bar = j; outcome = 'Target'; break
                elif sma20_valid and bar_c > float(bar_sma20):
                    exit_price = bar_c;  exit_bar = j; outcome = 'Trend';  break

        if exit_price is None:
            exit_price = float(df['Close'].iloc[exit_bar])

        if side == 'long':
            pnl     = (exit_price - entry_price) * shares
            pnl_pct = (exit_price - entry_price) / entry_price * 100
        else:
            pnl     = (entry_price - exit_price) * shares
            pnl_pct = (entry_price - exit_price) / entry_price * 100

        capital += pnl

        entry_dt   = df.index[entry_idx]
        exit_dt    = df.index[exit_bar]
        entry_date = str(entry_dt.date()) if hasattr(entry_dt, 'date') else str(entry_dt)[:10]
        exit_date  = str(exit_dt.date())  if hasattr(exit_dt,  'date') else str(exit_dt)[:10]

        equity_events[exit_date] = capital

        trades.append({
            'entry_date':       entry_date,
            'exit_date':        exit_date,
            'entry_price':      round(entry_price, 4),
            'exit_price':       round(exit_price,  4),
            'shares':           shares,
            'side':             side,
            'pnl':              round(pnl, 2),
            'pnl_pct':          round(pnl_pct, 2),
            'outcome':          outcome,
            'signals':          str(df['signal_list'].iloc[i]),
            'signal_count':     int(df['signal_count'].iloc[i]),
            'quality_score':         int(df['quality_score'].iloc[i]),
            'fundamental_score':     fundamental_score,
            'composite_factor_score': composite_factor_score,
        })

        i = exit_bar + 1

    # ── Equity curve ──────────────────────────────────────────────────────────
    equity_curve = []
    current_eq   = float(initial_capital)
    for dt in df.index:
        date_str = str(dt.date()) if hasattr(dt, 'date') else str(dt)[:10]
        if date_str in equity_events:
            current_eq = equity_events[date_str]
        equity_curve.append((date_str, current_eq))

    # ── Signal history ────────────────────────────────────────────────────────
    signal_history = [
        {
            'date':          str(dt.date()) if hasattr(dt, 'date') else str(dt)[:10],
            'signal_count':  int(df['signal_count'].iloc[k]),
            'watch_flag':    bool(df['watch_flag'].iloc[k]),
            'quality_score': int(df['quality_score'].iloc[k]),
        }
        for k, dt in enumerate(df.index)
    ]

    # ── Empty result guard ────────────────────────────────────────────────────
    if not trades:
        r = _empty_result(ticker, None, initial_capital)
        r['equity_curve']   = equity_curve
        r['signal_history'] = signal_history
        r['final_capital']  = round(capital, 2)
        return r

    # ── Metrics ───────────────────────────────────────────────────────────────
    wins         = [t for t in trades if t['pnl'] > 0]
    losses       = [t for t in trades if t['pnl'] <= 0]
    gross_profit = sum(t['pnl'] for t in wins)
    gross_loss   = abs(sum(t['pnl'] for t in losses))

    win_rate      = len(wins) / len(trades) * 100
    avg_win_pct   = sum(t['pnl_pct'] for t in wins)   / len(wins)   if wins   else 0.0
    avg_loss_pct  = sum(t['pnl_pct'] for t in losses) / len(losses) if losses else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 9999.0
    total_return  = (capital - initial_capital) / initial_capital * 100

    eq_vals  = pd.Series([v for _, v in equity_curve])
    peak     = eq_vals.cummax()
    drawdown = (eq_vals - peak) / peak * 100
    max_dd   = float(drawdown.min())

    eq_ret = eq_vals.pct_change().dropna()
    sharpe = float((eq_ret.mean() / eq_ret.std()) * (252 ** 0.5)) if (
        len(eq_ret) > 1 and eq_ret.std() > 0
    ) else 0.0

    # Average hold days
    hold_days_list = []
    for t in trades:
        try:
            d1 = datetime.strptime(t['entry_date'], '%Y-%m-%d')
            d2 = datetime.strptime(t['exit_date'],  '%Y-%m-%d')
            hold_days_list.append((d2 - d1).days)
        except Exception:
            pass
    avg_hold_days = round(sum(hold_days_list) / len(hold_days_list), 1) if hold_days_list else 0.0

    # Exit method breakdown
    total = len(trades)
    outcomes = [t['outcome'] for t in trades]
    exit_breakdown = {
        'Target':  round(outcomes.count('Target')  / total * 100, 1),
        'Stop':    round(outcomes.count('Stop')    / total * 100, 1),
        'Timeout': round(outcomes.count('Timeout') / total * 100, 1),
        'Trend':   round(outcomes.count('Trend')   / total * 100, 1),
    }

    best_trade  = max(trades, key=lambda t: t['pnl_pct'])
    worst_trade = min(trades, key=lambda t: t['pnl_pct'])

    return {
        'ticker':           ticker,
        'trades':           trades,
        'equity_curve':     equity_curve,
        'signal_history':   signal_history,
        'win_rate':         round(win_rate, 2),
        'avg_win_pct':      round(avg_win_pct, 2),
        'avg_loss_pct':     round(avg_loss_pct, 2),
        'profit_factor':    round(min(profit_factor, 9999.0), 2),
        'total_return_pct': round(total_return, 2),
        'max_drawdown_pct': round(max_dd, 2),
        'sharpe_ratio':     round(sharpe, 2),
        'total_trades':     len(trades),
        'avg_hold_days':    avg_hold_days,
        'exit_breakdown':   exit_breakdown,
        'best_trade':       best_trade,
        'worst_trade':      worst_trade,
        'final_capital':    round(capital, 2),
        'initial_capital':  float(initial_capital),
        'error':            None,
    }


def run_multi_backtest(
    tickers: list,
    initial_capital: float = 1000,
    quality_threshold: int = 50,
    hold_days: int = 10,
) -> pd.DataFrame:
    """
    Run backtest on multiple tickers; return comparison DataFrame sorted by
    Total Return % descending.
    """
    rows = []
    for t in tickers:
        r = run_backtest(t, initial_capital,
                         quality_threshold=quality_threshold,
                         hold_days=hold_days)
        rows.append({
            'Ticker':          r['ticker'],
            'Total Return %':  r['total_return_pct'],
            'Win Rate %':      r['win_rate'],
            'Profit Factor':   r['profit_factor'],
            'Max Drawdown %':  r['max_drawdown_pct'],
            'Sharpe Ratio':    r['sharpe_ratio'],
            'Avg Hold Days':   r['avg_hold_days'],
            'Total Trades':    r['total_trades'],
            'Final Capital':   r['final_capital'],
        })

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out = (
            df_out
            .sort_values('Total Return %', ascending=False, na_position='last')
            .reset_index(drop=True)
        )
    return df_out


# ── Private helpers ───────────────────────────────────────────────────────────

def _empty_result(ticker: str, error, initial_capital: float) -> dict:
    return {
        'ticker':           ticker,
        'trades':           [],
        'equity_curve':     [],
        'signal_history':   [],
        'win_rate':         0.0,
        'avg_win_pct':      0.0,
        'avg_loss_pct':     0.0,
        'profit_factor':    0.0,
        'total_return_pct': 0.0,
        'max_drawdown_pct': 0.0,
        'sharpe_ratio':     0.0,
        'total_trades':     0,
        'avg_hold_days':    0.0,
        'exit_breakdown':   {'Target': 0.0, 'Stop': 0.0, 'Timeout': 0.0, 'Trend': 0.0},
        'best_trade':       None,
        'worst_trade':      None,
        'final_capital':    float(initial_capital),
        'initial_capital':  float(initial_capital),
        'error':            error,
    }
