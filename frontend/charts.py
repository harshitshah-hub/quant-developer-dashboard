import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

PLOTLY_TEMPLATE = "plotly_dark"
MARGINS = dict(l=10, r=10, t=10, b=10)


def plot_price_chart(df: pd.DataFrame) -> go.Figure:
    """
    Line chart of price over time by symbol.
    Expects columns: ['ts', 'price', 'symbol'].
    """
    if df.empty or not {"ts", "price", "symbol"}.issubset(df.columns):
        return go.Figure()

    fig = px.line(
        df,
        x="ts",
        y="price",
        color="symbol",
        title=None,
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=True,
        margin=MARGINS,
    )
    return fig


def plot_spread_zscore(df: pd.DataFrame) -> go.Figure:
    """
    Dual-axis plot: spread (left y) & z-score (right y).
    Expects columns: ['ts', 'spread', 'zscore'].
    """
    needed = {"ts", "spread", "zscore"}
    if df.empty or not needed.issubset(df.columns):
        return go.Figure()

    fig = go.Figure()

    # Spread on primary y-axis
    fig.add_trace(
        go.Scatter(
            x=df["ts"],
            y=df["spread"],
            name="Spread",
            mode="lines",
        )
    )

    # Z-score on secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=df["ts"],
            y=df["zscore"],
            name="Z-Score",
            mode="lines",
            yaxis="y2",
        )
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        margin=MARGINS,
        title=None,
        xaxis=dict(title="Time"),
        yaxis=dict(title="Spread"),
        yaxis2=dict(title="Z-Score", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_correlation_heatmap(matrix) -> go.Figure:
    """
    Heatmap for rolling correlations.
    Accepts:
      - a pivoted DataFrame with index = pair, columns = ts, values = corr
      - a 2D numpy array (falls back to generic axes)
    """
    if isinstance(matrix, pd.DataFrame):
        if matrix.empty:
            return go.Figure()
        fig = px.imshow(
            matrix,
            aspect="auto",
            origin="lower",
            zmin=-1,
            zmax=1,
            labels=dict(x="Time", y="Pair", color="ρ (rolling)"),
            template=PLOTLY_TEMPLATE,
            title=None,
        )
    else:
        arr = np.asarray(matrix)
        if arr.size == 0:
            return go.Figure()
        fig = px.imshow(
            arr,
            aspect="auto",
            origin="lower",
            zmin=-1,
            zmax=1,
            labels=dict(x="Column", y="Row", color="ρ (rolling)"),
            template=PLOTLY_TEMPLATE,
            title=None,
        )

    fig.update_layout(margin=MARGINS)
    return fig


# Optional alias to avoid import mismatches (both names valid)
plot_heatmap = plot_correlation_heatmap
