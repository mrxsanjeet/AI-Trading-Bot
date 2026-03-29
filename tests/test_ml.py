"""Tests for the ML engine modules."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.features import FeatureEngineer
from src.ml.features import MLFeatureEngineer
from src.ml.models import LSTMPredictor, TransformerPredictor, create_model


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


class TestLSTMPredictor:
    def test_forward_pass(self):
        model = LSTMPredictor(input_size=10, hidden_size=32, num_layers=1)
        x = torch.randn(4, 60, 10)  # batch=4, seq=60, features=10
        output = model(x)
        assert output.shape == (4, 1)
        assert (output >= 0).all() and (output <= 1).all()  # Sigmoid output

    def test_different_configs(self):
        model = LSTMPredictor(input_size=20, hidden_size=64, num_layers=3, dropout=0.3)
        x = torch.randn(2, 30, 20)
        output = model(x)
        assert output.shape == (2, 1)


class TestTransformerPredictor:
    def test_forward_pass(self):
        model = TransformerPredictor(input_size=10, d_model=64, nhead=4, num_layers=2)
        x = torch.randn(4, 60, 10)
        output = model(x)
        assert output.shape == (4, 1)
        assert (output >= 0).all() and (output <= 1).all()


class TestCreateModel:
    def test_create_lstm(self):
        model = create_model("lstm", 10, {"hidden_size": 32, "num_layers": 1})
        assert isinstance(model, LSTMPredictor)

    def test_create_transformer(self):
        model = create_model("transformer", 10, {"hidden_size": 64, "nhead": 4})
        assert isinstance(model, TransformerPredictor)

    def test_invalid_type(self):
        with pytest.raises(ValueError):
            create_model("invalid", 10, {})


class TestMLFeatureEngineer:
    def test_prepare_features(self, sample_df):
        eng = MLFeatureEngineer(lookback_window=30)
        result = eng.prepare_features(sample_df)
        assert "X" in result
        assert "y" in result
        assert result["X"].shape[1] == 30  # lookback window
        assert result["X"].shape[2] > 5  # multiple features
        assert len(result["X"]) == len(result["y"])

    def test_train_test_split(self, sample_df):
        eng = MLFeatureEngineer(lookback_window=30)
        result = eng.prepare_features(sample_df)
        X_train, X_test, y_train, y_test = eng.train_test_split(
            result["X"], result["y"], test_ratio=0.2
        )
        assert len(X_train) > len(X_test)
        assert len(X_train) + len(X_test) == len(result["X"])

    def test_binary_labels(self, sample_df):
        eng = MLFeatureEngineer(lookback_window=30)
        result = eng.prepare_features(sample_df)
        assert set(np.unique(result["y"])).issubset({0, 1})
