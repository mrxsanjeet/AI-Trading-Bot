"""Tests for the backtesting engine."""

import numpy as np
import pandas as pd
import pytest

from src.data.features import FeatureEngineer
from src.execution.backtest import BacktestEngine
from src.strategy.mean_reversion import MeanReversionStrategy
from src.strategy.momentum import MomentumStrategy
from src.strategy.trend import TrendFollowingStrategy


@pytest.fixture
def sample_df():
    """Create sample OHLCV data with indicators for backtesting."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    base_price = 50000

    close = base_price + np.cumsum(np.random.randn(n) * 100)
    df = pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 50,
            "high": close + abs(np.random.randn(n) * 100),
            "low": close - abs(np.random.randn(n) * 100),
            "close": close,
            "volume": np.random.uniform(100, 10000, n),
        },
        index=dates,
    )

    eng = FeatureEngineer()
    return eng.add_all_indicators(df)


class TestBacktestEngine:
    def test_trend_following_backtest(self, sample_df):
        engine = BacktestEngine(initial_capital=10000)
        strategy = TrendFollowingStrategy()
        result = engine.run(sample_df, strategy, "BTC/USDT")

        assert "metrics" in result
        assert "equity_curve" in result
        assert "trades" in result
        assert result["metrics"]["initial_capital"] == 10000
        assert result["metrics"]["final_value"] > 0

    def test_mean_reversion_backtest(self, sample_df):
        engine = BacktestEngine(initial_capital=10000)
        strategy = MeanReversionStrategy()
        result = engine.run(sample_df, strategy, "BTC/USDT")
        assert result["metrics"]["final_value"] > 0

    def test_momentum_backtest(self, sample_df):
        engine = BacktestEngine(initial_capital=10000)
        strategy = MomentumStrategy()
        result = engine.run(sample_df, strategy, "BTC/USDT")
        assert result["metrics"]["final_value"] > 0

    def test_custom_parameters(self, sample_df):
        engine = BacktestEngine(
            initial_capital=50000,
            fee_rate=0.002,
            slippage_pct=0.001,
        )
        strategy = TrendFollowingStrategy()
        result = engine.run(sample_df, strategy, "BTC/USDT")
        assert result["metrics"]["initial_capital"] == 50000

    def test_metrics_completeness(self, sample_df):
        engine = BacktestEngine()
        strategy = TrendFollowingStrategy()
        result = engine.run(sample_df, strategy, "BTC/USDT")

        metrics = result["metrics"]
        required_keys = [
            "initial_capital", "final_value", "total_return_pct",
            "sharpe_ratio", "max_drawdown_pct", "total_trades",
            "win_rate", "profit_factor",
        ]
        for key in required_keys:
            assert key in metrics, f"Missing metric: {key}"

    def test_equity_curve_shape(self, sample_df):
        engine = BacktestEngine()
        strategy = TrendFollowingStrategy()
        result = engine.run(sample_df, strategy, "BTC/USDT")
        assert not result["equity_curve"].empty
        assert "value" in result["equity_curve"].columns
