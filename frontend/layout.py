import streamlit as st
from typing import List, Tuple


def app_layout(symbol_options: List[str]) -> Tuple[list, str, int, float, float, int]:
    """
    Sidebar controls for the dashboard.

    Parameters
    ----------
    symbol_options : list[str]
        Symbols detected in the loaded dataset (e.g., ["BTCUSDT", "ETHUSDT"]).

    Returns
    -------
    picked : list[str]
        Symbols selected by the user.
    timeframe : str
        Resampling timeframe (e.g., "1S", "1T", "5T").
    window : int
        Rolling window size (in bars).
    min_size : float
        Liquidity filter – ignore trades below this size.
    alert_thr : float
        Z-score alert threshold.
    max_points : int
        Max bars to plot per chart (for performance).
    """

    st.sidebar.header("Dashboard Controls")

    # Symbols to include
    default_syms = [s for s in ("BTCUSDT", "ETHUSDT") if s in symbol_options] or symbol_options[:2]
    picked = st.sidebar.multiselect(
        "Symbols to include",
        options=sorted(symbol_options),
        default=default_syms,
        help="Choose one or more instruments to analyze.",
    )

    # Resampling timeframe
    timeframe = st.sidebar.selectbox(
        "Resample timeframe",
        ["1S", "5S", "15S", "30S", "1T", "5T", "15T"],
        index=3,
    )

    # Rolling window
    window = st.sidebar.slider(
        "Rolling window (bars)",
        min_value=20,
        max_value=600,
        value=120,
        step=10,
        help="Used for z-score, rolling correlation, etc.",
    )

    # Liquidity filter
    min_size = st.sidebar.number_input(
        "Min trade size (liquidity filter)",
        value=0.0,
        step=0.001,
        help="Ignore trades smaller than this size before analytics.",
    )

    # Alert threshold
    alert_thr = st.sidebar.number_input(
        "Z-score alert threshold",
        value=2.0,
        step=0.5,
        help="Trigger an alert when |z| crosses this value.",
    )

    # Plot density cap
    max_points = st.sidebar.slider(
        "Max bars to plot (per chart)",
        min_value=200,
        max_value=5000,
        value=1000,
        step=100,
        help="Limits points for smoother charts and faster rendering.",
    )

    st.sidebar.divider()
    st.sidebar.caption("Tip: Use the range slider on charts to zoom specific periods.")

    return picked, timeframe, int(window), float(min_size), float(alert_thr), int(max_points)
