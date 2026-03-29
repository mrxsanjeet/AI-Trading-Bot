"""TradingView-style interactive chart generation using Plotly."""

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils.logger import get_logger

logger = get_logger("dashboard.charts")


def generate_tradingview_chart(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str = "1d",
    output_dir: str = "charts",
) -> str:
    """Generate an interactive TradingView-style HTML chart.

    Includes candlesticks, volume, EMA, Bollinger Bands, RSI, and MACD.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=[
            f"{symbol} ({timeframe})",
            "Volume",
            "RSI (14)",
            "MACD",
        ],
    )

    # --- Row 1: Candlestick + Overlays ---
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1, col=1,
    )

    # EMA 12 & 26
    if "ema_12" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["ema_12"], name="EMA 12",
                       line=dict(color="#2196F3", width=1)),
            row=1, col=1,
        )
    if "ema_26" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["ema_26"], name="EMA 26",
                       line=dict(color="#FF9800", width=1)),
            row=1, col=1,
        )

    # Bollinger Bands
    if "bb_upper" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["bb_upper"], name="BB Upper",
                       line=dict(color="rgba(150,150,150,0.5)", width=1, dash="dot")),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["bb_lower"], name="BB Lower",
                       line=dict(color="rgba(150,150,150,0.5)", width=1, dash="dot"),
                       fill="tonexty", fillcolor="rgba(150,150,150,0.05)"),
            row=1, col=1,
        )

    # --- Row 2: Volume ---
    colors = [
        "#26a69a" if c >= o else "#ef5350"
        for c, o in zip(df["close"], df["open"])
    ]
    fig.add_trace(
        go.Bar(x=df.index, y=df["volume"], name="Volume",
               marker_color=colors, opacity=0.7),
        row=2, col=1,
    )

    # --- Row 3: RSI ---
    if "rsi" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["rsi"], name="RSI",
                       line=dict(color="#AB47BC", width=1.5)),
            row=3, col=1,
        )
        # Overbought/Oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=3, col=1)

    # --- Row 4: MACD ---
    if "macd" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["macd"], name="MACD",
                       line=dict(color="#2196F3", width=1.5)),
            row=4, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["macd_signal"], name="Signal",
                       line=dict(color="#FF9800", width=1.5)),
            row=4, col=1,
        )
        # MACD histogram
        hist_colors = [
            "#26a69a" if v >= 0 else "#ef5350"
            for v in df["macd_histogram"].fillna(0)
        ]
        fig.add_trace(
            go.Bar(x=df.index, y=df["macd_histogram"], name="MACD Hist",
                   marker_color=hist_colors, opacity=0.5),
            row=4, col=1,
        )

    # --- Layout ---
    fig.update_layout(
        template="plotly_dark",
        height=900,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        margin=dict(l=60, r=30, t=80, b=30),
        title=dict(
            text=f"{symbol} — TradingView Analysis ({timeframe})",
            font=dict(size=18),
        ),
        font=dict(family="Arial, sans-serif", size=12),
    )

    # Y-axis labels
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=4, col=1)

    # Save
    safe_symbol = symbol.replace("/", "_").replace(".", "_")
    filename = f"{safe_symbol}_{timeframe}_chart.html"
    filepath = os.path.join(output_dir, filename)
    fig.write_html(filepath, include_plotlyjs=True)

    logger.info(f"TradingView chart saved: {filepath}")
    return os.path.abspath(filepath)


def generate_backtest_chart(
    df: pd.DataFrame,
    backtest_result: dict,
    symbol: str,
    output_dir: str = "charts",
) -> str:
    """Generate a backtest results chart with equity curve and trade markers."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    equity_df = backtest_result.get("equity_curve", pd.DataFrame())
    trades = backtest_result.get("trades", [])
    metrics = backtest_result.get("metrics", {})

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.45, 0.35, 0.2],
        subplot_titles=[
            f"{symbol} — Price & Trade Markers",
            "Equity Curve",
            "Drawdown",
        ],
    )

    # --- Row 1: Candlestick with buy/sell markers ---
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name="Price",
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        ),
        row=1, col=1,
    )

    # Buy markers
    buys = [t for t in trades if t.get("side") == "BUY"]
    if buys:
        buy_times = [t["timestamp"] for t in buys]
        buy_prices = [t["price"] for t in buys]
        fig.add_trace(
            go.Scatter(
                x=buy_times, y=buy_prices, mode="markers", name="BUY",
                marker=dict(symbol="triangle-up", size=12, color="#26a69a",
                            line=dict(width=1, color="white")),
            ),
            row=1, col=1,
        )

    # Sell markers
    sells = [t for t in trades if t.get("side") == "SELL" or t.get("action") == "CLOSE"]
    if sells:
        sell_times = [t["timestamp"] for t in sells]
        sell_prices = [t.get("exit_price", t.get("price", 0)) for t in sells]
        fig.add_trace(
            go.Scatter(
                x=sell_times, y=sell_prices, mode="markers", name="SELL",
                marker=dict(symbol="triangle-down", size=12, color="#ef5350",
                            line=dict(width=1, color="white")),
            ),
            row=1, col=1,
        )

    # --- Row 2: Equity curve ---
    if not equity_df.empty:
        fig.add_trace(
            go.Scatter(
                x=equity_df.index, y=equity_df["value"], name="Portfolio Value",
                line=dict(color="#00BCD4", width=2),
                fill="tozeroy", fillcolor="rgba(0,188,212,0.1)",
            ),
            row=2, col=1,
        )

        # Drawdown
        cummax = equity_df["value"].cummax()
        drawdown = ((equity_df["value"] - cummax) / cummax) * 100
        fig.add_trace(
            go.Scatter(
                x=equity_df.index, y=drawdown, name="Drawdown %",
                line=dict(color="#ef5350", width=1.5),
                fill="tozeroy", fillcolor="rgba(239,83,80,0.15)",
            ),
            row=3, col=1,
        )

    # --- Metrics annotation ---
    metrics_text = (
        f"Return: {metrics.get('total_return_pct', 0):+.2f}%  |  "
        f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}  |  "
        f"Win Rate: {metrics.get('win_rate', 0):.1f}%  |  "
        f"Max DD: {metrics.get('max_drawdown_pct', 0):.2f}%  |  "
        f"Trades: {metrics.get('total_trades', 0)}"
    )

    fig.update_layout(
        template="plotly_dark",
        height=950,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        margin=dict(l=60, r=30, t=100, b=30),
        title=dict(
            text=f"Backtest: {symbol}<br><sub>{metrics_text}</sub>",
            font=dict(size=16),
        ),
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Value ($)", row=2, col=1)
    fig.update_yaxes(title_text="DD %", row=3, col=1)

    safe_symbol = symbol.replace("/", "_").replace(".", "_")
    filename = f"{safe_symbol}_backtest_chart.html"
    filepath = os.path.join(output_dir, filename)
    fig.write_html(filepath, include_plotlyjs=True)

    logger.info(f"Backtest chart saved: {filepath}")
    return os.path.abspath(filepath)
