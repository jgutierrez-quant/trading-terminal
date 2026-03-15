# Trading Terminal

**Bloomberg-style equity research and paper-trading dashboard built in Python.**

![Python 3.13](https://img.shields.io/badge/Python-3.13-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.54-FF4B4B)
![Paper Trading Only](https://img.shields.io/badge/Trading-Paper%20Only-yellow)

## Overview

A full-featured trading terminal that pulls real-time market data, options flow, social sentiment, and fundamental analysis into one interface. Powered by a 7-factor quantitative scoring model with academic backing, DCF intrinsic valuation, and integrated Alpaca paper trading. Everything runs locally via Streamlit with a dark Bloomberg-style theme.

## Architecture

```
Data Sources              Analysis Engine           Signal Scoring            Dashboard
─────────────            ────────────────          ──────────────           ─────────
Yahoo Finance  ──┐       ┌─ Technicals             ┌─ 7-Factor Model  ──┐
Polygon.io     ──┼──────►├─ Fundamentals    ──────►├─ Anomaly Detect  ──┼──► Streamlit
Unusual Whales ──┤       ├─ Sentiment (NLP)        ├─ Trade Coach     ──┤    8-Tab UI
FRED (Macro)   ──┤       ├─ DCF Valuation          └─ Risk Manager   ──┘
Alpaca (Paper) ──┘       ├─ Catalyst Detection
                         ├─ Whale Flow Analysis
                         └─ PEAD Tracking
```

## Modules

| Module | File | Purpose |
|--------|------|---------|
| Market Data | `data/market_data.py` | Aggregates price feeds from yfinance + Polygon |
| Technicals | `data/technicals.py` | RSI, MACD, Bollinger Bands, VWAP, SMA crossovers |
| Fundamentals | `data/fundamentals.py` | P/E, FCF, ROE, short interest, insider flow |
| DCF Valuation | `data/dcf.py` | 5-year DCF with base/bull/bear scenarios |
| Factor Model | `data/factor_model.py` | 7-factor quant scoring engine (see below) |
| Anomaly Detector | `data/anomaly_detector.py` | Multi-signal alignment → Watch flag + quality score |
| Sentiment | `sentiment/` | Yahoo News (VADER NLP), Finviz headlines, Google Trends |
| Screener | `data/screener.py` | S&P 500 scan — top movers, unusual volume, anomaly flags |
| Catalyst Detector | `data/catalyst_detector.py` | Earnings, FDA, M&A, insider, and macro catalyst scanning |
| Whale Detector | `data/whale_detector.py` | Unusual options activity and institutional block trades |
| PEAD Tracker | `data/pead_tracker.py` | Post-earnings announcement drift opportunity detection |
| Trade Coach | `data/trade_coach.py` | Actionable trade recommendations with entry/stop/target |
| Backtester | `data/backtester.py` | Historical replay of signal logic with ATR stops |
| Alpaca Client | `data/alpaca_client.py` | Paper trading execution via Alpaca API |
| Trade Logger | `utils/trade_logger.py` | SQLite journal for all trades + performance tracking |
| Risk Manager | `utils/risk_manager.py` | Position sizing, stop-loss, take-profit calculation |
| Market Regime | `utils/market_regime.py` | VIX + breadth regime detection (Risk-On/Risk-Off) |

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | Streamlit, Plotly (candlestick, radar, gauge charts) |
| **Data** | yfinance, Polygon.io, Unusual Whales, FRED |
| **NLP** | VADER Sentiment, BeautifulSoup, pytrends |
| **Analysis** | NumPy, Pandas, custom factor model + DCF engine |
| **Trading** | Alpaca Paper Trading API |
| **Storage** | SQLite (trades.db, alerts.db) |

## 7-Factor Quantitative Model

Each factor is scored as a percentile (0-100) against S&P 500 calibrated distributions. Composite score drives Long / Short / Neutral signals.

| # | Factor | Weight | Academic Basis |
|---|--------|--------|---------------|
| 1 | Price Momentum | 18% | Jegadeesh & Titman (1993) — 12-1 month cross-sectional momentum |
| 2 | Earnings Momentum | 18% | Ball & Brown (1968), Bernard & Thomas (1989) — SUE + PEAD |
| 3 | Value | 14% | Fama & French (1992) — EV/EBITDA, P/E, P/B, FCF yield composite |
| 4 | Quality | 18% | Novy-Marx (2013) — ROE, margins, leverage, EPS consistency |
| 5 | Short Interest | 9% | Asquith, Pathak & Ritter — short float %, squeeze detection |
| 6 | Institutional Flow | 13% | Nofsinger & Sias (1999) — ownership %, insider signals, upgrades |
| 7 | DCF Intrinsic Value | 10% | 5-year DCF margin of safety (perpetuity growth + exit multiple) |

Missing factors are excluded and weights automatically renormalized. Composite > 70 = Long candidate, < 30 = Short candidate.

## Features

- **8-tab dashboard**: Market View, Screener, Whale Flow, Squeeze Scanner, Portfolio, Backtest, Performance, Journal
- **Real-time data**: yfinance + Polygon intraday bars with 60-second auto-refresh
- **7-factor quant model**: Academically grounded composite scoring with radar visualization
- **DCF valuation**: 3-scenario intrinsic value with margin-of-safety signals
- **Trade coaching**: Plain English recommendations with entry, stop, target, and position sizing
- **Backtesting**: Replay signal logic on 2-year daily data with ATR stops and trend exits
- **Performance analytics**: Cumulative P&L curve, win/loss distribution, per-ticker breakdown
- **Paper trading**: Alpaca integration with bracket orders, risk validation, and P&L tracking
- **Catalyst detection**: Earnings, FDA, M&A, insider, and macro event scanning
- **Whale flow**: Unusual options activity and institutional block trade monitoring
- **Sentiment NLP**: Yahoo News + Finviz headline scoring (VADER) + Google Trends
- **Market regime**: VIX + breadth-based Risk-On / Risk-Off / Neutral classification
- **Alerts**: Price-level alerts with SQLite persistence and notification log

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd trading-terminal

# Create virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your API keys (see .env.example for required keys)

# Launch
streamlit run dashboard/app.py
```

### Required API Keys

| Key | Source | Required |
|-----|--------|----------|
| `POLYGON_API_KEY` | [polygon.io](https://polygon.io) | Optional (intraday + options) |
| `ALPACA_API_KEY` + `ALPACA_SECRET_KEY` | [alpaca.markets](https://alpaca.markets) | Optional (paper trading) |
| `UNUSUAL_WHALES_API_KEY` | [unusualwhales.com](https://unusualwhales.com) | Optional (whale flow) |
| `FRED_API_KEY` | [fred.stlouisfed.org](https://fred.stlouisfed.org) | Optional (macro data) |

The terminal runs with yfinance alone (no API key needed). Additional keys unlock premium data sources.

## Screenshots

<!-- screenshot: market-view -->
<!-- screenshot: factor-model -->
<!-- screenshot: performance-tab -->
<!-- screenshot: backtest-tab -->

## Disclaimer

This project is for **educational and paper trading purposes only**. It is not financial advice. No real money is at risk. The factor model and DCF calculations are simplified implementations for learning — they should not be used as the sole basis for real trading decisions. Always do your own research.
