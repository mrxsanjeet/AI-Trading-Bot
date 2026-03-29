"""Tests for the strategy engine modules."""

import numpy as np
import pandas as pd
import pytest

from src.data.features import FeatureEngineer
from src.strategy.base import Signal, TradeSignal
from src.strategy.mean_reversion import MeanReversionStrategy
from src.strategy.momentum import MomentumStrategy
from src.strategy.risk import RiskManager
from src.strategy.signals import SignalAggregator
from src.strategy.trend import TrendFollowingStrategy


@pytest.fixture
def sample_df():
    """Create sample OHLCV data with indicators."""
    np.random.seed(42)
    n = 200
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


class TestTrendFollowingStrategy:
    def test_generates_valid_signal(self, sample_df):
        strategy = TrendFollowingStrategy()
        signal = strategy.generate_signal(sample_df, "BTC/USDT")
        assert isinstance(signal, TradeSignal)
        assert signal.signal in [Signal.BUY, Signal.SELL, Signal.HOLD]
        assert signal.symbol == "BTC/USDT"
        assert 0 <= signal.confidence <= 1

    def test_insufficient_data(self, sample_df):
        strategy = TrendFollowingStrategy()
        signal = strategy.generate_signal(sample_df.head(5), "BTC/USDT")
        assert signal.signal == Signal.HOLD
        assert "Insufficient" in signal.reason

    def test_custom_config(self):
        config = {"ema_short": 8, "ema_long": 21}
        strategy = TrendFollowingStrategy(config)
        assert strategy.ema_short == 8
        assert strategy.ema_long == 21


class TestMeanReversionStrategy:
    def test_generates_valid_signal(self, sample_df):
        strategy = MeanReversionStrategy()
        signal = strategy.generate_signal(sample_df, "ETH/USDT")
        assert isinstance(signal, TradeSignal)
        assert signal.signal in [Signal.BUY, Signal.SELL, Signal.HOLD]

    def test_custom_rsi_thresholds(self):
        config = {"rsi_oversold": 25, "rsi_overbought": 75}
        strategy = MeanReversionStrategy(config)
        assert strategy.rsi_oversold == 25
        assert strategy.rsi_overbought == 75


class TestMomentumStrategy:
    def test_generates_valid_signal(self, sample_df):
        strategy = MomentumStrategy()
        signal = strategy.generate_signal(sample_df, "BTC/USDT")
        assert isinstance(signal, TradeSignal)
        assert signal.signal in [Signal.BUY, Signal.SELL, Signal.HOLD]


class TestRiskManager:
    @pytest.fixture
    def risk_manager(self):
        return RiskManager({
            "max_position_pct": 5,
            "stop_loss_pct": 2,
            "take_profit_pct": 6,
            "max_drawdown_pct": 15,
            "daily_loss_limit": 500,
            "max_open_positions": 3,
            "cooldown_periods": 3,
        })

    def test_approve_valid_signal(self, risk_manager):
        signal = TradeSignal(
            signal=Signal.BUY, symbol="BTC/USDT", price=50000,
            confidence=0.8, strategy="test", reason="test"
        )
        result = risk_manager.evaluate(signal, 10000, 10000, 0, 0)
        assert result["approved"]
        assert result["position_size"] > 0

    def test_reject_max_positions(self, risk_manager):
        signal = TradeSignal(
            signal=Signal.BUY, symbol="BTC/USDT", price=50000,
            confidence=0.8, strategy="test", reason="test"
        )
        result = risk_manager.evaluate(signal, 10000, 10000, 3, 0)  # 3 open positions
        assert not result["approved"]
        assert "Max open positions" in result["reason"]

    def test_reject_max_drawdown(self, risk_manager):
        signal = TradeSignal(
            signal=Signal.BUY, symbol="BTC/USDT", price=50000,
            confidence=0.8, strategy="test", reason="test"
        )
        result = risk_manager.evaluate(signal, 8000, 8000, 0, 20)  # 20% drawdown
        assert not result["approved"]

    def test_hold_always_approved(self, risk_manager):
        signal = TradeSignal(
            signal=Signal.HOLD, symbol="BTC/USDT", price=50000,
            confidence=0.0, strategy="test", reason="test"
        )
        result = risk_manager.evaluate(signal, 10000, 10000, 5, 50)
        assert result["approved"]

    def test_cooldown_after_losses(self, risk_manager):
        # Record 3 consecutive losses
        for _ in range(3):
            risk_manager.record_trade_result(-100)

        signal = TradeSignal(
            signal=Signal.BUY, symbol="BTC/USDT", price=50000,
            confidence=0.8, strategy="test", reason="test"
        )
        result = risk_manager.evaluate(signal, 10000, 10000, 0, 0)
        assert not result["approved"]
        assert "Cooldown" in result["reason"]

    def test_win_resets_consecutive_losses(self, risk_manager):
        risk_manager.record_trade_result(-100)
        risk_manager.record_trade_result(-100)
        risk_manager.record_trade_result(200)  # Win resets counter
        assert risk_manager._consecutive_losses == 0


class TestSignalAggregator:
    def test_aggregator_with_strategies(self, sample_df):
        strategies = [
            TrendFollowingStrategy(),
            MeanReversionStrategy(),
            MomentumStrategy(),
        ]
        aggregator = SignalAggregator(strategies, min_agreement=2)
        result = aggregator.generate_signals(sample_df, "BTC/USDT")

        assert "individual_signals" in result
        assert "aggregated_signal" in result
        assert len(result["individual_signals"]) == 3
        assert isinstance(result["aggregated_signal"], TradeSignal)
