"""Streamlit dashboard for the AI Trading Bot."""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from src.data.storage import DataStorage
from src.utils.helpers import load_config, format_currency, format_pct


def create_dashboard():
    """Main Streamlit dashboard entry point."""
    st.set_page_config(
        page_title="AI Trading Bot",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("AI Trading Bot Dashboard")

    # Load config
    try:
        config = load_config()
    except FileNotFoundError:
        st.error("Configuration file not found. Please ensure config/settings.yaml exists.")
        return

    # Sidebar
    st.sidebar.header("Settings")
    mode = st.sidebar.selectbox("Trading Mode", ["paper", "live", "backtest"], index=0)
    symbols = st.sidebar.multiselect(
        "Symbols",
        config["trading"]["symbols"],
        default=config["trading"]["symbols"],
    )
    timeframe = st.sidebar.selectbox(
        "Timeframe",
        ["1m", "5m", "15m", "1h", "4h", "1d"],
        index=3,
    )

    # Initialize storage
    try:
        storage = DataStorage(config["database"]["url"])
    except Exception as e:
        st.warning(f"Database not available: {e}")
        storage = None

    # Main layout
    col1, col2, col3, col4 = st.columns(4)

    # Portfolio metrics (placeholder values when no live data)
    with col1:
        st.metric("Portfolio Value", format_currency(10000), format_pct(0))
    with col2:
        st.metric("Daily P&L", format_currency(0), format_pct(0))
    with col3:
        st.metric("Open Positions", "0")
    with col4:
        st.metric("Win Rate", "0%")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Chart", "Trades", "Backtest", "Settings"])

    with tab1:
        _render_chart_tab(storage, symbols, timeframe)

    with tab2:
        _render_trades_tab(storage, mode)

    with tab3:
        _render_backtest_tab(config)

    with tab4:
        _render_settings_tab(config)


def _render_chart_tab(storage, symbols, timeframe):
    """Render the price chart tab."""
    st.subheader("Price Chart")

    if not symbols:
        st.info("Select a symbol from the sidebar")
        return

    selected = st.selectbox("Symbol", symbols, key="chart_symbol")

    if storage:
        df = storage.load_ohlcv(selected, timeframe)
        if not df.empty:
            fig = _create_candlestick_chart(df, selected)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No data available for {selected}. Run the bot to fetch data.")
    else:
        st.info("Database not connected. Start the bot to begin collecting data.")


def _create_candlestick_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """Create a candlestick chart with volume."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
        ),
        row=1, col=1,
    )

    # Volume
    colors = ["red" if c < o else "green" for c, o in zip(df["close"], df["open"])]
    fig.add_trace(
        go.Bar(x=df.index, y=df["volume"], marker_color=colors, name="Volume", opacity=0.5),
        row=2, col=1,
    )

    fig.update_layout(
        title=f"{symbol} Price Chart",
        xaxis_rangeslider_visible=False,
        height=600,
        template="plotly_dark",
    )

    return fig


def _render_trades_tab(storage, mode):
    """Render the trades history tab."""
    st.subheader("Trade History")

    if storage:
        trades = storage.get_trades(mode=mode, limit=50)
        if trades:
            df = pd.DataFrame(trades)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No trades recorded yet.")
    else:
        st.info("Database not connected.")


def _render_backtest_tab(config):
    """Render the backtest configuration and results tab."""
    st.subheader("Backtesting")

    col1, col2 = st.columns(2)
    with col1:
        bt_symbol = st.selectbox("Symbol", config["trading"]["symbols"], key="bt_symbol")
        bt_timeframe = st.selectbox("Timeframe", ["1h", "4h", "1d"], key="bt_tf")
        bt_capital = st.number_input("Initial Capital ($)", value=10000, min_value=100)

    with col2:
        bt_strategy = st.selectbox("Strategy", ["TrendFollowing", "MeanReversion", "Momentum"])
        bt_fee = st.number_input("Fee Rate (%)", value=0.1, min_value=0.0, max_value=1.0) / 100
        bt_slippage = st.number_input("Slippage (%)", value=0.05, min_value=0.0, max_value=1.0) / 100

    if st.button("Run Backtest", type="primary"):
        st.info("Backtest functionality — use the CLI: `python -m src.main backtest`")


def _render_settings_tab(config):
    """Render the settings tab."""
    st.subheader("Current Configuration")
    st.json(config)


def _create_equity_curve(equity_df: pd.DataFrame) -> go.Figure:
    """Create an equity curve chart."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=equity_df.index,
            y=equity_df["value"],
            mode="lines",
            name="Portfolio Value",
            line=dict(color="cyan", width=2),
        )
    )
    fig.update_layout(
        title="Equity Curve",
        yaxis_title="Portfolio Value ($)",
        template="plotly_dark",
        height=400,
    )
    return fig


if __name__ == "__main__":
    create_dashboard()
