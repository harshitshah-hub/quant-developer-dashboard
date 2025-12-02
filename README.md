# Quant Developer Assignment — Live Data Dashboard

A Streamlit dashboard for exploring cryptocurrency market data (e.g., **BTCUSDT**, **ETHUSDT**).  
It resamples tick data into bars and provides analytics such as spread & z-score, rolling correlation, ADF stationarity, and a simple mean-reversion backtest. Interactive charts are built with Plotly.

---

## Project Structure
Quant_Developer_Assignment/
├── app.py
├── backend/
│ ├── analytics.py
│ └── data_processor.py
├── data/
│ └── sample_data.ndjson
├── docs/
│ ├── architecture_diagram.png
│ └── screenshots/
│ ├── dashboard_overview.png
│ ├── spread_zscore.png
│ ├── backtest_equity.png
│ └── heatmap.png
├── README.md
├── requirements.txt
├── chatgpt_usage.md
└── SUBMISSION.md



---

## How to Run (Windows)

### 1) Create and activate a virtual environment
```powershell
python -m venv venv
venv\Scripts\Activate.ps1


2) Install dependencies
pip install -r requirements.txt

3) Provide data

Place one or more NDJSON files in the data/ directory. Each line must be valid JSON:

{"symbol":"BTCUSDT","ts":"2025-11-04T10:20:19.000Z","price":103726.4,"size":0.002}


The app automatically loads the most recent file in this folder.

Alternatively, upload a CSV inside the app. It must include at least:

ts, symbol, price


If a column named close is present instead of price, it is renamed automatically.

4) Run the app
python -m streamlit run app.py



Then open your browser at: http://localhost:8501

Architecture Diagram

See docs/architecture_diagram.png for the end-to-end data flow and system components.

Features

Symbol picker for selecting instruments (BTCUSDT, ETHUSDT)

Resampling to multiple timeframes: 1S, 5S, 15S, 30S, 1T, 5T, 15T

Liquidity filter (ignore trades below a chosen min_size)

CSV upload for OHLC-style data with columns ts, symbol, price (or close)

Interactive Plotly charts with zoom and range sliders

Download buttons for resampled bars and analytics (CSV)

Analytics

BTC − β·ETH spread (β estimated via OLS)

Z-score of the spread

Rolling correlation (BTC vs ETH)

ADF stationarity test on spread or a chosen symbol

Alerts on z-score thresholds

Mean-reversion backtest using z-score entry/exit levels (with optional transaction costs)

Rolling correlation heatmap across selected symbols

Per-minute stats table (last/mean/std) with CSV export

Requirements

Installed via requirements.txt:

streamlit
pandas
numpy
plotly
statsmodels

Notes

If a CSV has close instead of price, it is renamed automatically.

If the size column is missing, a dummy value is assigned.

Rolling window sizes are capped by available bars to avoid NaNs.

Pair-based analytics (spread, correlation, backtest) require both BTCUSDT and ETHUSDT.

Submission Summary

This project delivers a complete Streamlit dashboard that loads crypto tick data, resamples it into time bars, and runs core analytics including spread, z-score, correlation, ADF, and a mean-reversion backtest.
It supports data uploads, alerting, interactive charts, a rolling correlation heatmap, and CSV exports.

The system was tested end-to-end with both NDJSON and CSV inputs.

Commands used for environment setup and execution:

python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py


The app runs successfully on http://localhost:8501
.

Included files:

app.py, backend/analytics.py, backend/data_processor.py

requirements.txt

README.md

Example NDJSON file in /data

docs/architecture_diagram.png (system overview)

Methodology & Design

This project is organized for clarity and reuse. Data processing, analytics, and visualization are cleanly separated.

1) Data Flow & Architecture

Data Source: Tick data from .ndjson files or uploaded OHLC CSVs.

Preprocessing: Missing fields (like size) are handled; timestamps are parsed to proper datetimes.

Liquidity Filter: Trades below a user-defined min_size are filtered before analytics for cleaner signals.

Resampling Layer: Ticks are resampled into bars in backend/data_processor.py with flexible timeframes (1s–15min).

2) Analytics Pipeline

Price Alignment: prices_wide() reshapes data for pairwise analysis.

Spread & Z-Score: spread_and_zscore() computes BTC − β·ETH (β via OLS) and its rolling z-score.

Rolling Correlation: rolling_corr() and rolling_corr_matrix() capture evolving relationships.

Stationarity: adf_test() checks mean-reversion assumptions.

Backtesting: backtest_mean_reversion() simulates a simple z-score strategy with optional transaction costs.

3) Visualization & Interactivity

Streamlit frontend with a dark theme, sidebar controls, and tabbed layout:

Prices: interactive price charts

Analytics: spread, z-score, rolling correlation

Strategy: backtest metrics and equity curve

Rolling Corr Heatmap: pairwise rolling correlations

Tables: raw data, returns, and per-minute stats (CSV export)

User-tunable parameters: timeframe, rolling window, liquidity threshold, and backtest settings.

4) Extensibility & Maintainability

Analytics live in backend/analytics.py and can be extended with new indicators/strategies.

The design supports adding more assets or data sources without large changes.

The same pipeline can scale to higher frequency or live feeds.

Author

Harshit Shah
harshitshah045@gmail.com