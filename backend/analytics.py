# backend/analytics.py
from typing import Tuple, Dict, Optional
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


# -------------------------------
# Wide prices & basic analytics
# -------------------------------

def prices_wide(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot to wide form and forward-fill so symbols are aligned by timestamp.
    Result index = ts, columns = symbols (e.g., BTCUSDT, ETHUSDT), values = price.
    """
    if "ts" not in bars.columns or "symbol" not in bars.columns or "price" not in bars.columns:
        return pd.DataFrame()

    # Ensure datetime + sorted
    p = bars.copy()
    p["ts"] = pd.to_datetime(p["ts"], utc=True, errors="coerce")
    p = p.dropna(subset=["ts"]).sort_values("ts")

    p = p.pivot(index="ts", columns="symbol", values="price")
    p = p.ffill()  # align/forward fill between symbols
    return p


def hedge_ratio_ols(y: pd.Series, x: pd.Series) -> float:
    """
    OLS: y = a + b*x  -> return b (hedge ratio).
    Robust for short series; returns NaN if not enough points.
    """
    data = pd.concat([y, x], axis=1).dropna()
    if len(data) < 3:
        return float("nan")

    y1 = data.iloc[:, 0]
    x1 = data.iloc[:, 1]
    X = sm.add_constant(x1)
    try:
        model = sm.OLS(y1, X).fit()
        b = float(model.params[1]) if len(model.params) >= 2 else float("nan")
        return b
    except Exception:
        return float("nan")


def spread_and_zscore(
    prices_w: pd.DataFrame,
    window: int = 60,
    y_sym: str = "BTCUSDT",
    x_sym: str = "ETHUSDT",
) -> Tuple[pd.DataFrame, float]:
    """
    Compute spread = y - beta*x and its rolling Z-score.
    Returns (DataFrame[spread,zscore], beta). Safe for short series.
    """
    need = {y_sym, x_sym}
    if prices_w is None or prices_w.empty or not need.issubset(prices_w.columns):
        return pd.DataFrame(), float("nan")

    df = prices_w[[y_sym, x_sym]].dropna(how="any")
    if df.empty:
        return pd.DataFrame(), float("nan")

    beta = hedge_ratio_ols(df[y_sym], df[x_sym])
    if pd.isna(beta):
        return pd.DataFrame(), float("nan")

    spread = df[y_sym] - beta * df[x_sym]

    # rolling stats need at least 2 obs; be defensive on short series
    w = max(2, min(int(window), len(spread) - 1)) if len(spread) > 2 else 2
    m = spread.rolling(w).mean()
    s = spread.rolling(w).std()
    z = (spread - m) / s

    out = pd.DataFrame({"spread": spread, "zscore": z})
    return out, float(beta)


def pct_returns(prices_w: pd.DataFrame) -> pd.DataFrame:
    """Simple percentage returns per symbol (aligned index)."""
    if prices_w is None or prices_w.empty:
        return pd.DataFrame()
    return prices_w.pct_change()


def rolling_corr(prices_w: pd.DataFrame, window: int = 60) -> pd.Series:
    """
    Rolling correlation between BTCUSDT and ETHUSDT.
    Safe if one of the symbols is missing.
    """
    if prices_w is None or prices_w.empty or not {"BTCUSDT", "ETHUSDT"}.issubset(prices_w.columns):
        return pd.Series(dtype=float)
    w = max(2, int(window))
    return prices_w["BTCUSDT"].rolling(w).corr(prices_w["ETHUSDT"])


def adf_test(series: pd.Series) -> Tuple[float, float]:
    """Return (statistic, pvalue) for ADF; handles short/empty series."""
    if series is None:
        return float("nan"), float("nan")
    s = series.dropna()
    if len(s) < 20:
        return float("nan"), float("nan")
    stat, p, *_ = adfuller(s.values, autolag="AIC")
    return float(stat), float(p)


# -------------------------------------------------
# Rolling correlation matrix for heatmap (bonus)
# -------------------------------------------------

def rolling_corr_matrix(
    bars_long: pd.DataFrame,
    value_col: str = "price",
    window: int = 120,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build a time/pair rolling correlation frame for heatmaps.

    bars_long must have columns: ['ts','symbol', value_col]
    Returns long-form DataFrame with columns: ['ts','pair','corr'].

    Example usage (Streamlit):
        heat = rolling_corr_matrix(resampled_long_df, "price", 120)
        # pivot to heatmap: rows=pair, cols=ts, values=corr
    """
    if bars_long is None or bars_long.empty:
        return pd.DataFrame(columns=["ts", "pair", "corr"])

    df = bars_long[["ts", "symbol", value_col]].copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts")
    if df.empty:
        return pd.DataFrame(columns=["ts", "pair", "corr"])

    pivot = df.pivot_table(index="ts", columns="symbol", values=value_col)
    if pivot.shape[1] < 2:
        return pd.DataFrame(columns=["ts", "pair", "corr"])

    w = max(2, int(window))
    mp = min_periods if min_periods is not None else max(2, w // 3)

    # Rolling correlation yields a 3D-like object (ts, symbol) x symbol
    roll = pivot.rolling(window=w, min_periods=mp).corr()

    # Collect upper-triangle pairs for each timestamp
    symbols = list(pivot.columns)
    rows = []
    for t in pivot.index:
        # guard: rolling result is not available for earliest rows
        if (t, symbols[0]) not in roll.index:
            continue
        mat = roll.loc[t]
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                a, b = symbols[i], symbols[j]
                corr_val = mat.loc[a, b] if (a in mat.index and b in mat.columns) else np.nan
                if pd.notna(corr_val):
                    rows.append({"ts": t, "pair": f"{a}–{b}", "corr": float(corr_val)})

    return pd.DataFrame(rows, columns=["ts", "pair", "corr"])


# ---------------------------------------
# Mean-Reversion Backtest on the spread
# ---------------------------------------

def backtest_mean_reversion(
    prices_w: pd.DataFrame,
    window: int,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    tc_bps: float = 0.0,
    y_sym: str = "BTCUSDT",
    x_sym: str = "ETHUSDT",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Simple mean-reversion backtest on the (y_sym - β*x_sym) spread.

    Rules
    -----
    - Go LONG spread when zscore <= -entry_z
    - Go SHORT spread when zscore >=  entry_z
    - Exit to FLAT when |zscore| <= exit_z

    PnL (approx)
    ------------
    ret(t) = position(t-1) * Δspread(t)  -  tc
      where tc is applied when position changes: tc = bps / 1e4

    Returns
    -------
    df : DataFrame with columns ['spread','zscore','position','ret','equity']
         indexed by timestamp; 'ret' includes transaction costs (if any).
    stats : dict(trades, cum_pnl, sharpe, beta)
    """
    # Guards
    empty_stats = {"trades": 0, "cum_pnl": 0.0, "sharpe": float("nan"), "beta": float("nan")}
    if prices_w is None or prices_w.empty:
        return pd.DataFrame(), empty_stats

    sz, beta = spread_and_zscore(prices_w, window, y_sym=y_sym, x_sym=x_sym)
    if sz.empty or not sz["zscore"].notna().any():
        s = empty_stats.copy()
        s["beta"] = float(beta) if beta == beta else float("nan")
        return pd.DataFrame(), s

    df = sz[["spread", "zscore"]].dropna().copy()
    if df.empty:
        s = empty_stats.copy()
        s["beta"] = float(beta) if beta == beta else float("nan")
        return pd.DataFrame(), s

    # Position: -1 short spread, 0 flat, +1 long spread
    pos = []
    cur = 0
    e_in = float(entry_z)
    e_out = float(exit_z)

    for z in df["zscore"]:
        # Exit to flat inside the band first
        if abs(z) <= e_out:
            cur = 0
        elif z >= e_in:
            cur = -1
        elif z <= -e_in:
            cur = 1
        pos.append(cur)

    df["position"] = pd.Series(pos, index=df.index)

    # Returns: position(t-1) * Δspread(t)
    df["d_spread"] = df["spread"].diff().fillna(0.0)
    gross = df["position"].shift(1).fillna(0) * df["d_spread"]

    # Transaction cost when you change position
    turns = (df["position"] != df["position"].shift(1).fillna(0)).astype(int)
    tc = turns * (tc_bps / 1e4) if tc_bps and tc_bps > 0 else 0.0

    df["ret"] = gross - tc
    df["equity"] = df["ret"].cumsum()

    # Stats
    trades = int(turns.sum())
    cum_pnl = float(df["equity"].iloc[-1]) if len(df) else 0.0
    r = df["ret"].replace([np.inf, -np.inf], np.nan).dropna()
    sharpe = float(r.mean() / r.std()) if r.std() and not np.isnan(r.std()) else float("nan")

    stats: Dict[str, float] = {
        "trades": trades,
        "cum_pnl": cum_pnl,
        "sharpe": sharpe,
        "beta": float(beta) if beta == beta else float("nan"),
    }
    # Clean up columns users will likely plot
    return df[["spread", "zscore", "position", "ret", "equity"]], stats
