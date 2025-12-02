# Submission Checklist

- Runnable app: `streamlit run app.py`
- Backend: ingestion → resample → analytics (OLS β, spread, z-score, ADF, rolling corr)
- Frontend: interactive charts, zoom/pan, controls, alerts
- Mini Backtest: z-entry/exit, equity curve, optional TC bps
- Rolling Correlation Heatmap
- Downloads: resampled bars, analytics, backtest series
- README.md (setup + how to run)
- requirements.txt
- Architecture diagram (.png + .drawio)
- ChatGPT usage note (chatgpt_usage.md)
- Sample data (/data/*.ndjson)

## Notes
- Symbols supported: BTCUSDT, ETHUSDT (extensible)
- Typical window: 120 bars; timeframe: 30S/1T
