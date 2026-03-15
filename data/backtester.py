"""
Backtesting engine — replays the same signal logic as anomaly_detector.py
across 2 years of daily data, simulates entries/exits with ATR-based
risk management, and returns structured results for the Backtest tab.

Public API:
    get_historical_data(ticker, period='2y')  -> pd.DataFrame
    generate_signals(df)                       -> pd.DataFrame
    run_backtest(ticker, ...)                  -> dict
    run_multi_backtest(tickers, ...)           -> pd.DataFrame

CRITICAL: Signal logic mirrors data/technicals.py and data/anomaly_detector.py EXACTLY.
    RSI:    gain.ewm(com=13, min_periods=14).mean()  (Wilder's smoothing)
    MACD:   EWM span=12/26/9, adjust=False
    BB:     rolling(20).mean() +/- 2.0 * rolling(20).std(ddof=0)
    Volume: volume / volume.rolling(20).mean().shift(1)   <- shift avoids look-ahead
    SMA:    rolling(20/50).mean()
    VWAP:   (High + Low + Close) / 3  (daily PROXY — see note below)
    Watch threshold: score >= 3  (same as WATCH_THRESHOLD in anomaly_detector.py)

NOTE ON VWAP PROXY:
    Live code computes VWAP from intraday 1-minute bars (cumulative TP*Vol / cumVol).
    That data is not available in historical mode. This module uses the daily typical
    price (H+L+C)/3 as an approximation. Signals derived from vwap_proxy will differ
    from live VWAP signals, especially intra-day.
"""

import math
import logging

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ── Constants — must match anomaly_detector.py and utils/risk_manager.py ────────
WATCH_THRESHOLD  = 3
ATR_PERIOD       = 14
ATR_MULTIPLIER   = 2.0
MAX_POSITION_PCT = 0.20


# ── Private indicator helpers — identical formulas to technicals.py ─────────────

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


# ── Public functions ─────────────────────────────────────────────────────────────

def get_historical_data(ticker: str, period: str = '2y') -> pd.DataFrame:
    """
    Fetch daily OHLCV from yfinance and compute all indicators vectorized.

    Returns DataFrame with columns:
        Open, High, Low, Close, Volume,
        rsi, macd_hist, bb_upper, bb_lower,
        sma20, sma50, vol_ratio, atr, vwap_proxy

    Raises ValueError if data is insufficient (< 60 bars).
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

    # RSI — identical EWM (Wilder's smoothing) to technicals.py
    df['rsi'] = _rsi(close, 14)

    # MACD — EWM span=12/26/9, adjust=False (identical to technicals.py)
    _, _, hist = _macd(close, 12, 26, 9)
    df['macd_hist'] = hist

    # Bollinger Bands — rolling(20), ddof=0 (identical to technicals.py)
    bb_upper, _, bb_lower = _bbands(close, 20, 2.0)
    df['bb_upper'] = bb_upper
    df['bb_lower'] = bb_lower

    # SMA 20 / 50
    df['sma20'] = close.rolling(20).mean()
    df['sma50'] = close.rolling(50).mean()

    # Volume ratio — shift(1) avoids look-ahead bias (vectorized equivalent of
    # vol.iloc[-21:-1].mean() used in technicals.py for the single-bar case)
    df['vol_ratio'] = volume / volume.rolling(20).mean().shift(1)

    # ATR 14-period for stop-loss sizing (same formula as risk_manager.py)
    df['atr'] = _atr(high, low, close, 14)

    # VWAP proxy — daily typical price (NOT intraday VWAP)
    # ⚠️  Live code: cumulative (TP * Vol) / cumVol from 1-min bars.
    #     This is a daily approximation only — signals will differ from live.
    df['vwap_proxy'] = (high + low + close) / 3

    return df


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all signal conditions vectorized, matching anomaly_detector.py exactly.

    Adds columns:
        rsi_signal, macd_signal, bb_signal, vol_signal,
        vwap_signal, price_vs_sma20, price_vs_sma50,
        signal_count, signal_list, watch_flag, direction
    """
    close = df['Close']

    # ── RSI ───────────────────────────────────────────────────────────────────
    df['rsi_signal'] = np.where(df['rsi'] < 30, 'Oversold',
                       np.where(df['rsi'] > 70, 'Overbought', 'Neutral'))

    # ── MACD — sign-change detection (matches prv_hist/cur_hist logic) ────────
    prv = df['macd_hist'].shift(1)
    cur = df['macd_hist']
    df['macd_signal'] = np.where(
        (prv < 0) & (cur >= 0), 'Bullish Cross',
        np.where(
            (prv > 0) & (cur <= 0), 'Bearish Cross',
            np.where(cur > 0, 'Bullish', 'Bearish')
        )
    )

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    df['bb_signal'] = np.where(close > df['bb_upper'], 'Above Upper',
                      np.where(close < df['bb_lower'], 'Below Lower', 'Inside Bands'))

    # ── Volume ────────────────────────────────────────────────────────────────
    df['vol_signal'] = np.where(df['vol_ratio'] >= 2.0, 'High Volume',
                       np.where(df['vol_ratio'] >= 1.5, 'Elevated', 'Normal'))

    # ── VWAP (proxy) ──────────────────────────────────────────────────────────
    df['vwap_signal'] = np.where(close > df['vwap_proxy'], 'Above VWAP', 'Below VWAP')

    # ── SMA ───────────────────────────────────────────────────────────────────
    df['price_vs_sma20'] = np.where(close > df['sma20'], 'Above', 'Below')
    df['price_vs_sma50'] = np.where(close > df['sma50'], 'Above', 'Below')

    # ── Signal count — identical scoring to anomaly_detector.py ──────────────
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
    df['watch_flag']   = df['signal_count'] >= WATCH_THRESHOLD

    # ── Signal list + direction (per-row apply — acceptable for ~500 rows) ────
    def _signals_and_dir(row):
        sigs = []
        bull = 0
        bear = 0

        if row['rsi_signal'] == 'Oversold':
            sigs.append('Oversold RSI');      bull += 1
        elif row['rsi_signal'] == 'Overbought':
            sigs.append('Overbought RSI');    bear += 1

        if row['macd_signal'] == 'Bullish Cross':
            sigs.append('Bullish MACD Cross'); bull += 1
        elif row['macd_signal'] == 'Bearish Cross':
            sigs.append('Bearish MACD Cross'); bear += 1

        if row['bb_signal'] == 'Below Lower':
            sigs.append('BB Oversold');       bull += 1
        elif row['bb_signal'] == 'Above Upper':
            sigs.append('BB Overbought');     bear += 1

        if row['vol_signal'] == 'High Volume':
            sigs.append('High Volume')
        elif row['vol_signal'] == 'Elevated':
            sigs.append('Elevated Volume')

        if row['vwap_signal'] == 'Above VWAP':
            sigs.append('Above VWAP');        bull += 1
        elif row['vwap_signal'] == 'Below VWAP':
            sigs.append('Below VWAP');        bear += 1

        if row['price_vs_sma20'] == 'Above' and row['price_vs_sma50'] == 'Above':
            sigs.append('Price Above MA20+MA50'); bull += 1
        elif row['price_vs_sma20'] == 'Below' and row['price_vs_sma50'] == 'Below':
            sigs.append('Price Below MA20+MA50'); bear += 1

        direction = (
            'Bullish' if bull > bear else
            'Bearish' if bear > bull else
            'Mixed'   if sigs else
            'Neutral'
        )
        return pd.Series({'signal_list': ', '.join(sigs), 'direction': direction})

    _res = df.apply(_signals_and_dir, axis=1)
    df['signal_list'] = _res['signal_list']
    df['direction']   = _res['direction']

    return df


def run_backtest(
    ticker: str,
    initial_capital: float = 1000,
    risk_percent: float = 2.0,
    risk_reward: float = 2.0,
    hold_days: int = 5,
) -> dict:
    """
    Sequential backtest with no look-ahead bias.

    Algorithm:
        Entry:  next-day Open after a watch_flag bar (entry bar = signal_bar + 1)
        Stop:   entry +/- ATR[signal_bar] * 2.0
        Target: entry +/- stop_distance * risk_reward
        Exit priority: Stop hit -> Target hit -> Timeout at close after hold_days
        No overlapping trades — loop advances past exit bar before scanning again.

    Returns dict with keys:
        ticker, trades, equity_curve, signal_history,
        win_rate, avg_win_pct, avg_loss_pct, profit_factor,
        total_return_pct, max_drawdown_pct, sharpe_ratio,
        total_trades, best_trade, worst_trade,
        final_capital, initial_capital, error
    """
    try:
        df = generate_signals(get_historical_data(ticker))
    except Exception as exc:
        logger.error("Backtest failed for %s: %s", ticker, exc)
        return _empty_result(ticker, str(exc), initial_capital)

    capital       = float(initial_capital)
    trades        = []
    equity_events = {}   # date_str -> capital after this exit
    n             = len(df)
    i             = 0

    while i < n - 1:
        if not df['watch_flag'].iloc[i]:
            i += 1
            continue

        entry_idx   = i + 1
        entry_price = float(df['Open'].iloc[entry_idx])
        atr_val     = df['atr'].iloc[i]

        if pd.isna(atr_val) or float(atr_val) <= 0 or entry_price <= 0:
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

        # Position sizing — identical to utils/risk_manager.calculate_position_size()
        dollar_risk = capital * (risk_percent / 100)
        shares      = int(math.floor(dollar_risk / stop_dist))
        max_shares  = int(math.floor(capital * MAX_POSITION_PCT / entry_price))
        shares      = min(shares, max_shares)

        if shares <= 0:
            i += 1
            continue

        # Simulate exit: scan bars entry_idx .. entry_idx+hold_days (inclusive)
        exit_price = None
        exit_bar   = min(entry_idx + hold_days, n - 1)
        outcome    = 'Timeout'

        for j in range(entry_idx, exit_bar + 1):
            bar_h = float(df['High'].iloc[j])
            bar_l = float(df['Low'].iloc[j])
            if side == 'long':
                if bar_l <= stop:
                    exit_price = stop;   exit_bar = j; outcome = 'Stop';   break
                elif bar_h >= target:
                    exit_price = target; exit_bar = j; outcome = 'Target'; break
            else:
                if bar_h >= stop:
                    exit_price = stop;   exit_bar = j; outcome = 'Stop';   break
                elif bar_l <= target:
                    exit_price = target; exit_bar = j; outcome = 'Target'; break

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
            'entry_date':   entry_date,
            'exit_date':    exit_date,
            'entry_price':  round(entry_price, 4),
            'exit_price':   round(exit_price,  4),
            'shares':       shares,
            'side':         side,
            'pnl':          round(pnl, 2),
            'pnl_pct':      round(pnl_pct, 2),
            'outcome':      outcome,
            'signals':      str(df['signal_list'].iloc[i]),
            'signal_count': int(df['signal_count'].iloc[i]),
        })

        i = exit_bar + 1   # skip rows that were part of the completed trade

    # ── Equity curve — daily, forward-filled between exits ────────────────────
    equity_curve  = []
    current_eq    = float(initial_capital)
    for dt in df.index:
        date_str = str(dt.date()) if hasattr(dt, 'date') else str(dt)[:10]
        if date_str in equity_events:
            current_eq = equity_events[date_str]
        equity_curve.append((date_str, current_eq))

    # ── Signal history for chart display ──────────────────────────────────────
    signal_history = [
        {
            'date':         str(dt.date()) if hasattr(dt, 'date') else str(dt)[:10],
            'signal_count': int(df['signal_count'].iloc[k]),
            'watch_flag':   bool(df['watch_flag'].iloc[k]),
        }
        for k, dt in enumerate(df.index)
    ]

    # ── Return empty metrics if no trades ────────────────────────────────────
    if not trades:
        result = _empty_result(ticker, None, initial_capital)
        result['equity_curve']   = equity_curve
        result['signal_history'] = signal_history
        result['final_capital']  = round(capital, 2)
        return result

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
    if len(eq_ret) > 1 and eq_ret.std() > 0:
        sharpe = float((eq_ret.mean() / eq_ret.std()) * (252 ** 0.5))
    else:
        sharpe = 0.0

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
        'best_trade':       best_trade,
        'worst_trade':      worst_trade,
        'final_capital':    round(capital, 2),
        'initial_capital':  float(initial_capital),
        'error':            None,
    }


def run_multi_backtest(tickers: list, initial_capital: float = 1000) -> pd.DataFrame:
    """
    Run backtest on multiple tickers; return comparison DataFrame sorted by
    Total Return % descending.
    """
    rows = []
    for t in tickers:
        r = run_backtest(t, initial_capital)
        rows.append({
            'Ticker':          r['ticker'],
            'Total Return %':  r['total_return_pct'],
            'Win Rate %':      r['win_rate'],
            'Profit Factor':   r['profit_factor'],
            'Max Drawdown %':  r['max_drawdown_pct'],
            'Sharpe Ratio':    r['sharpe_ratio'],
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


# ── Private helpers ──────────────────────────────────────────────────────────────

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
        'best_trade':       None,
        'worst_trade':      None,
        'final_capital':    float(initial_capital),
        'initial_capital':  float(initial_capital),
        'error':            error,
    }
