# Trading Terminal

A Bloomberg-style trading terminal built in Python with a Streamlit dashboard.
Pulls real-time market data, options flow, and social sentiment into one interface.

## Stage Roadmap

| Stage | Description                                    | Status      |
|-------|------------------------------------------------|-------------|
| 1     | Project setup, folder structure, dependencies  | Done        |
| 2     | Live market data — price feeds via yfinance + Polygon | Pending |
| 3     | Options flow — unusual activity via Unusual Whales | Pending |
| 4     | Sentiment feed — StockTwits + VADER NLP analysis | Pending |
| 5     | Streamlit dashboard — charts, tables, live refresh | Pending |
| 6     | Scheduler — auto-refresh data on interval      | Pending |
| 7     | Google Trends integration via pytrends         | Pending |
| 8     | Polish — alerts, watchlists, export to CSV     | Pending |

## Project Structure

```
trading-terminal/
├── data/           # Market data API connectors (yfinance, Polygon)
├── sentiment/      # Social sentiment feeds and NLP (StockTwits, VADER)
├── dashboard/      # Streamlit frontend and chart components
├── utils/          # Shared helpers (config, logging, formatting)
├── .env            # API keys (not committed to git)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
# Copy .env and fill in your API keys
```

## Running the Dashboard (Stage 5+)

```bash
streamlit run dashboard/app.py
```
