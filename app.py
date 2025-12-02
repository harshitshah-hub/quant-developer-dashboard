from pathlib import Path
import json
import platform
import pandas as pd
import streamlit as st
import plotly.express as px

from backend.analytics import (
    prices_wide,
    spread_and_zscore,
    pct_returns,
    rolling_corr,
    adf_test,
    backtest_mean_reversion,    # updated in backend
    rolling_corr_matrix,        # heatmap helper
)
from backend.data_processor import resample_ticks

# NEW: modular UI & charts
from frontend.layout import app_layout
from frontend.charts import (
    plot_price_chart,
    plot_spread_zscore,
    plot_heatmap,
)

# ----------------------------- #
# Page config + lightweight dark styling
# ----------------------------- #
st.set_page_config(
    page_title="Quant Developer Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Subtle styling
st.markdown(
    """
    <style>
      .block-container { padding-top: 1rem; padding-bottom: 3rem; }
      .stMetric { background: rgba(255,255,255,0.03); border-radius: 10px; padding: 0.5rem 0.75rem; }
      .stTabs [data-baseweb="tab-list"] { gap: 0.25rem; }
      .stTabs [data-baseweb="tab"] { padding: 8px 14px; border-radius: 8px; }
      [data-testid="stMetricDelta"] svg { display: inline-block !important; opacity: 1 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

PLOTLY_TEMPLATE = "plotly_dark"

# ----------------------------- #
# Header
# ----------------------------- #
st.title("Quant Developer Assignment — Live Data Dashboard")

st.markdown("## Welcome")
st.info(
    "Use this dashboard to explore tick/price data, run analytics on BTCUSDT and ETHUSDT, "
    "and visualize correlation, spreads, alerts, stationarity, backtests, and a rolling correlation heatmap. "
    "Use the sidebar to upload data and choose settings."
)
st.markdown("---")

# ----------------------------- #
# Data loaders
# ----------------------------- #
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / "data"

@st.cache_data
def load_latest_ndjson() -> pd.DataFrame:
    files = sorted(DATA_DIR.glob("*.ndjson"))
    if not files:
        return pd.DataFrame()
    data_file = files[-1]
    with data_file.open("r") as f:
        data = [json.loads(line) for line in f if line.strip()]
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"])
    return df

# ----------------------------- #
# Load data (NDJSON or uploaded CSV)
# ----------------------------- #
df = load_latest_ndjson()
if df.empty:
    st.error("No NDJSON found in /data (or file empty). Add one and rerun.")
    st.stop()

st.caption(f"Active dataset: {len(df):,} ticks (latest NDJSON in /data)")

st.subheader("Raw Data Preview")
# Fixed: only dataframe here (no stray plot call / undefined fig)
st.dataframe(df.head(), use_container_width=True)

st.subheader("Optional: Upload OHLC CSV (overrides NDJSON)")
upl = st.file_uploader(
    "CSV must contain columns: ts, symbol, and either price or close",
    type=["csv"]
)

if upl is not None:
    try:
        df_csv = pd.read_csv(upl)
        if "ts" not in df_csv.columns or "symbol" not in df_csv.columns:
            raise ValueError("CSV must include at least 'ts' and 'symbol' columns.")
        if "price" not in df_csv.columns:
            if "close" in df_csv.columns:
                df_csv = df_csv.rename(columns={"close": "price"})
            else:
                raise ValueError("CSV must have a 'price' column or a 'close' column.")
        df_csv["ts"] = pd.to_datetime(df_csv["ts"])
        keep_cols = [c for c in ["ts", "symbol", "price", "size"] if c in df_csv.columns]
        df = df_csv[keep_cols].dropna(subset=["ts", "symbol", "price"]).copy()
        st.success(f"Loaded {len(df):,} rows from uploaded OHLC CSV.")
    except Exception as e:
        st.error(f"Could not parse CSV: {e}")

# If 'size' missing (typical for OHLC), create a dummy size so resampler can aggregate
if "size" not in df.columns:
    df["size"] = 1.0

# ----------------------------- #
# Sidebar + symbol selection (modular)
# ----------------------------- #
available_syms = sorted(df["symbol"].unique().tolist())
picked, timeframe, window, min_size, alert_thr, max_points = app_layout(available_syms)

# Optional toggle (kept here to avoid cluttering layout.py)
show_returns = st.sidebar.checkbox("Show percentage returns table", value=False)

# If no symbols picked, fall back to all (and warn)
if picked:
    df = df[df["symbol"].isin(picked)]
else:
    st.warning("No symbols selected; showing all available.")

# Apply liquidity filter before resampling
df = df[df["size"].astype(float) >= float(min_size)]

# ----------------------------- #
# Resample
# ----------------------------- #
bars = resample_ticks(df, timeframe)

# ----------------------------- #
# Analytics (safe for single-symbol picks)
# ----------------------------- #
prices_w = prices_wide(bars)            # wide table (BTCUSDT, ETHUSDT as columns)
available = len(prices_w)               # bars available for rolling stats
effective_window = max(2, min(window, available - 1)) if available >= 3 else None

has_btc = "BTCUSDT" in prices_w.columns
has_eth = "ETHUSDT" in prices_w.columns
pairs_ok = has_btc and has_eth
single_symbol_msg = (
    "Only one symbol detected — spread, rolling correlation, z-score alerts, "
    "backtest, and the heatmap require **both BTCUSDT and ETHUSDT**."
)

if available < 3:
    st.info(
        f"Not enough bars for analytics yet (need >= 3). "
        f"Available: {available}. Try a shorter timeframe (e.g., 1S/5S) or load more data."
    )
    corr = pd.Series(dtype=float)
    sz   = pd.DataFrame()
    beta = float("nan")
else:
    st.caption(f"Using rolling window = {effective_window} (available bars: {available})")
    if pairs_ok:
        corr = rolling_corr(prices_w, effective_window)
        sz, beta = spread_and_zscore(prices_w, effective_window)
    else:
        corr = pd.Series(dtype=float)
        sz   = pd.DataFrame()
        beta = float("nan")

# ----------------------------- #
# KPI bar
# ----------------------------- #
k1, k2, k3 = st.columns(3)

def metric_with_delta(col, label, series_df: pd.DataFrame):
    try:
        s = series_df.sort_values("ts")["price"].to_list()
        if len(s) >= 2:
            last, prev = s[-1], s[-2]
            delta = last - prev
            col.metric(label, f"{last:,.2f}", delta=delta, delta_color="normal")
        elif len(s) == 1:
            col.metric(label, f"{s[-1]:,.2f}")
        else:
            col.metric(label, "—")
    except Exception:
        col.metric(label, "—")

metric_with_delta(k1, "BTCUSDT (last)", bars[bars["symbol"] == "BTCUSDT"])
metric_with_delta(k2, "ETHUSDT (last)", bars[bars["symbol"] == "ETHUSDT"])

last_z = None
if not sz.empty and sz["zscore"].notna().any():
    last_z = float(sz["zscore"].dropna().iloc[-1])
k3.metric("Latest Z-Score (spread)", "—" if last_z is None else f"{last_z:.2f}")

if not pairs_ok:
    st.caption("Note: With a single symbol selected, spread/correlation/strategy/heatmap features are disabled.")

# ----------------------------- #
# Alerts + ADF controls
# ----------------------------- #
if available >= 3:
    st.subheader("Alerts (spread-based)")
    if not pairs_ok:
        st.info(single_symbol_msg)
    else:
        colA, colB, colC = st.columns(3)
        metric = colA.selectbox("Metric", ["Z-Score (spread)"])
        op     = colB.selectbox("Condition", [">", "<"])
        thr    = colC.number_input("Threshold", value=float(alert_thr), step=0.5)

        if metric == "Z-Score (spread)" and not sz.empty and sz["zscore"].notna().any():
            last = sz["zscore"].dropna().iloc[-1]
            cond = (last > thr) if op == ">" else (last < thr)
            st.write(f"Latest z-score: {last:.2f}")
            if cond:
                st.success(f"Alert: z-score {op} {thr}")
            else:
                st.info("No alert at the moment.")
        else:
            st.caption("Z-score not available yet.")

    st.subheader("Stationarity (ADF)")
    adf_target = st.selectbox("Run ADF on:", ["Spread", "BTCUSDT", "ETHUSDT"])
    run_adf = st.button("Run ADF test")

    if run_adf:
        if adf_target == "Spread" and pairs_ok and not sz.empty:
            stat, p = adf_test(sz["spread"])
        else:
            series = prices_w[adf_target] if adf_target in prices_w.columns else pd.Series(dtype=float)
            stat, p = adf_test(series)
        if pd.isna(p):
            st.warning("Not enough points to run ADF (need roughly 20+).")
        else:
            st.write(
                f"ADF statistic: {stat:.3f}, p-value: {p:.4f} "
                f"{'(likely stationary)' if p < 0.05 else '(not stationary at 5%)'}"
            )

# ----------------------------- #
# Tabs: Prices | Analytics | Strategy | Heatmap | Tables
# ----------------------------- #
tab_prices, tab_analytics, tab_strategy, tab_heatmap, tab_tables = st.tabs(
    ["Prices", "Analytics", "Strategy", "Rolling Corr Heatmap", "Tables"]
)

def _tail_limit(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.tail(n) if len(x) > n else x

# -------- Prices tab --------
with tab_prices:
    st.subheader("Resampled Prices")
    prices_long = bars[bars["symbol"].isin(picked or available_syms)].sort_values("ts")
    if prices_long.empty:
        st.caption("No bars to plot yet.")
    else:
        prices_long = prices_long[["ts", "symbol", "price"]]
        prices_long = prices_long.groupby("symbol", group_keys=False).apply(lambda d: d.tail(max_points))
        st.plotly_chart(plot_price_chart(prices_long), width="stretch")

# -------- Analytics tab --------
with tab_analytics:
    if not pairs_ok:
        st.info(single_symbol_msg)
    else:
        reg_type = st.selectbox(
            "Regression type",
            ["OLS"],
            index=0,
            help="Hedge ratio β estimated via Ordinary Least Squares."
        )
        st.caption("β (hedge ratio) is currently estimated via OLS. Other methods can be added later.")

        st.subheader(f"Spread & Z-Score (β = {beta:.3f} via {reg_type})" if not pd.isna(beta) else "Spread & Z-Score")

        if not sz.empty and (sz["spread"].notna().any() or sz["zscore"].notna().any()):
            sz_plot = sz.dropna().reset_index().rename(columns={"index": "ts"})
            sz_plot = _tail_limit(sz_plot, max_points)
            st.plotly_chart(plot_spread_zscore(sz_plot), width="stretch")
        else:
            st.caption("Spread/z-score appears once enough bars are available.")

        st.subheader("Rolling Correlation (BTC vs ETH)")
        if hasattr(corr, "empty") and not corr.empty and corr.notna().any():
            df_corr = corr.dropna().reset_index()
            df_corr.columns = ["ts", "corr"]
            df_corr = _tail_limit(df_corr, max_points)
            fig_corr = px.line(df_corr, x="ts", y="corr", title=None, template=PLOTLY_TEMPLATE)
            fig_corr.update_layout(xaxis_rangeslider_visible=True, yaxis_range=[-1, 1], margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_corr, width="stretch")
        else:
            st.caption("Correlation appears once enough bars are available.")

# -------- Strategy tab --------
with tab_strategy:
    st.subheader("Mean-Reversion Backtest on Spread")
    st.caption("Simple z-score entry/exit system on BTC − β·ETH spread.")

    if not pairs_ok:
        st.info(single_symbol_msg)
    else:
        col1, col2, col3, col4 = st.columns(4)
        entry_z = col1.number_input("Entry |z| >", min_value=0.5, max_value=5.0, step=0.1, value=2.0)
        exit_z  = col2.number_input("Exit |z| <",  min_value=0.1, max_value=3.0, step=0.1, value=0.5)
        win_bt  = col3.number_input("Window (bars)", min_value=10, max_value=2000, step=10,
                                    value=max(30, int(window)))
        tc_bps  = col4.number_input("TC (bps, one-way)", min_value=0.0, max_value=20.0, step=0.1, value=0.0)

        run_bt = st.button("Run Backtest")

        if run_bt:
            bt_df, stats = backtest_mean_reversion(
                prices_w, int(win_bt), float(entry_z), float(exit_z), float(tc_bps)
            )

            if bt_df.empty:
                st.info("Not enough data to run the backtest on this window.")
            else:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Trades", f"{stats.get('trades', 0):,}")
                m2.metric("β (OLS)", f"{stats.get('beta', float('nan')):.3f}"
                         if pd.notna(stats.get('beta', float('nan'))) else "—")
                m3.metric("Sharpe (proxy)", f"{stats.get('sharpe', float('nan')):.2f}"
                         if pd.notna(stats.get('sharpe', float('nan'))) else "—")
                m4.metric("Cum PnL", f"{stats.get('cum_pnl', 0.0):.4f}")

                eq_plot = bt_df.reset_index().rename(columns={"index": "ts"})
                fig_cum = px.line(eq_plot, x="ts", y="equity", title="Equity Curve", template=PLOTLY_TEMPLATE)
                fig_cum.update_layout(margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_cum, width="stretch")

                with st.expander("Backtest Table"):
                    st.dataframe(bt_df.tail(500), use_container_width=True)

                csv_bt = eq_plot.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Backtest (CSV)",
                    data=csv_bt,
                    file_name="mean_reversion_backtest.csv",
                    mime="text/csv",
                )

# -------- Heatmap tab --------
with tab_heatmap:
    st.subheader("Rolling Correlation Heatmap")
    st.caption("Computes rolling pairwise correlations across selected symbols.")
    st.caption("Bars per symbol: " + str(bars["symbol"].value_counts().to_dict()))

    if len(picked) < 2:
        st.info("Pick at least two symbols to compute the heatmap.")
    else:
        win_hm = st.number_input("Rolling window (bars)", value=max(30, int(window)), step=10, min_value=10)
        if st.button("Compute Heatmap"):
            heat_df = rolling_corr_matrix(
                bars[["ts", "symbol", "price"]].dropna(),
                value_col="price",
                window=int(win_hm),
                min_periods=2   # allow early correlations even with few bars
            )
            if heat_df.empty:
                st.warning("Not enough data to compute rolling correlations yet.")
            else:
                pivot = heat_df.pivot(index="pair", columns="ts", values="corr")
                st.plotly_chart(plot_heatmap(pivot), width="stretch")

                st.download_button(
                    "Download rolling correlations (CSV)",
                    heat_df.to_csv(index=False).encode(),
                    "rolling_corr.csv",
                    "text/csv",
                )

# -------- Tables tab --------
with tab_tables:
    st.subheader("Tables")
    st.markdown("**Raw preview**")
    st.dataframe(df.head(50), use_container_width=True)

    if show_returns and not prices_w.empty:
        st.markdown("**Percentage Returns (aligned)**")
        rets = pct_returns(prices_w).replace([float("inf"), float("-inf")], pd.NA)
        rets = rets.dropna(how="all")
        if effective_window is not None:
            rets = rets.tail(effective_window)

        if rets.empty:
            st.caption("No returns to display yet (need at least 2 bars per symbol).")
        else:
            st.dataframe(rets.style.format("{:.4%}"), use_container_width=True)
    else:
        st.caption("Turn on 'Show percentage returns table' in the sidebar to display returns here.")

    st.subheader("Per-minute Stats")
    minute_stats = (
        bars.groupby(["symbol", pd.Grouper(key="ts", freq="1min")])["price"]  # '1T' -> '1min'
            .agg(price_last="last", price_mean="mean", price_std="std")
            .reset_index()
            .dropna(subset=["price_last"])
    )
    if minute_stats.empty:
        st.caption("Not enough data for per-minute stats yet.")
    else:
        st.dataframe(minute_stats.tail(300), use_container_width=True)
        st.download_button(
            "Download per-minute stats (CSV)",
            minute_stats.to_csv(index=False).encode(),
            "minute_stats.csv",
            "text/csv",
        )

# ----------------------------- #
# Downloads (under expander)
# ----------------------------- #
with st.expander("Downloads", expanded=False):
    st.caption("Export resampled bars and analytics as CSV files.")

    bars_csv = bars.sort_values(["symbol", "ts"]).to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Resampled Bars (CSV)",
        data=bars_csv,
        file_name="resampled_bars.csv",
        mime="text/csv",
    )

    analytics_frames = []
    if not sz.empty:
        tmp = sz[["spread", "zscore"]].copy()
        tmp = tmp.reset_index().rename(columns={"index": "ts"})
        analytics_frames.append(tmp)

    if hasattr(corr, "empty") and not corr.empty:
        corr_df = corr.to_frame(name="corr").reset_index().rename(columns={"index": "ts"})
        analytics_frames.append(corr_df)

    if analytics_frames:
        analytics_df = analytics_frames[0]
        for extra in analytics_frames[1:]:
            analytics_df = pd.merge(analytics_df, extra, on="ts", how="outer")
        analytics_df = analytics_df.sort_values("ts")
    else:
        analytics_df = pd.DataFrame(columns=["ts", "spread", "zscore", "corr"])

    analytics_csv = analytics_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Analytics (CSV)",
        data=analytics_csv,
        file_name="analytics.csv",
        mime="text/csv",
    )

# ----------------------------- #
# Footer
# ----------------------------- #
st.markdown("---")
c1, c2, c3 = st.columns(3)
with c1:
    st.caption(f"Python {platform.python_version()}")
with c2:
    st.caption(f"Streamlit {st.__version__}")
with c3:
    try:
        import pandas as _pd
        st.caption(f"Pandas {_pd.__version__}")
    except Exception:
        st.caption("Pandas")

st.markdown(
    "<div style='text-align:center; color:gray'>"
    "Developed by Harshit Shah • © 2025 Quant Developer Assignment"
    "</div>",
    unsafe_allow_html=True,
)
