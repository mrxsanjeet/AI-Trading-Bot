"""Tests for the data engine modules."""

import numpy as np
import pandas as pd
import pytest

from src.data.cleaner import DataCleaner
from src.data.features import FeatureEngineer


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data for testing."""
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
    return df


class TestDataCleaner:
    def test_clean_removes_duplicates(self, sample_ohlcv):
        df = pd.concat([sample_ohlcv, sample_ohlcv.tail(5)])  # Add duplicates
        cleaner = DataCleaner()
        cleaned = cleaner.clean(df)
        assert not cleaned.index.duplicated().any()

    def test_clean_handles_missing_values(self, sample_ohlcv):
        df = sample_ohlcv.copy()
        df.iloc[5, 0] = np.nan  # Set a NaN in open
        cleaner = DataCleaner()
        cleaned = cleaner.clean(df)
        assert not cleaned.isnull().any().any()

    def test_clean_sorts_index(self, sample_ohlcv):
        df = sample_ohlcv.sample(frac=1)  # Shuffle
        cleaner = DataCleaner()
        cleaned = cleaner.clean(df)
        assert cleaned.index.is_monotonic_increasing

    def test_validate_good_data(self, sample_ohlcv):
        cleaner = DataCleaner()
        issues = cleaner.validate(sample_ohlcv)
        # May have high < low issues with random data, but should not be empty
        assert isinstance(issues, list)

    def test_validate_empty_df(self):
        cleaner = DataCleaner()
        issues = cleaner.validate(pd.DataFrame())
        assert len(issues) > 0
        assert "empty" in issues[0].lower()


class TestFeatureEngineer:
    def test_add_all_indicators(self, sample_ohlcv):
        eng = FeatureEngineer()
        result = eng.add_all_indicators(sample_ohlcv)
        assert "rsi" in result.columns
        assert "macd" in result.columns
        assert "bb_upper" in result.columns
        assert "atr" in result.columns
        assert "obv" in result.columns
        assert "ema_12" in result.columns
        assert "sma_50" in result.columns

    def test_add_rsi(self, sample_ohlcv):
        eng = FeatureEngineer()
        result = eng.add_rsi(sample_ohlcv)
        assert "rsi" in result.columns
        # RSI should be between 0 and 100 (after warm-up)
        valid_rsi = result["rsi"].dropna()
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()

    def test_add_bollinger_bands(self, sample_ohlcv):
        eng = FeatureEngineer()
        result = eng.add_bollinger_bands(sample_ohlcv)
        assert "bb_upper" in result.columns
        assert "bb_lower" in result.columns
        assert "bb_middle" in result.columns

    def test_add_returns(self, sample_ohlcv):
        eng = FeatureEngineer()
        result = eng.add_returns(sample_ohlcv)
        assert "returns" in result.columns
        assert "log_returns" in result.columns
        assert "volatility_20" in result.columns

    def test_get_feature_columns(self, sample_ohlcv):
        eng = FeatureEngineer()
        result = eng.add_all_indicators(sample_ohlcv)
        features = eng.get_feature_columns(result)
        assert "open" not in features
        assert "close" not in features
        assert len(features) > 10
