"""Technical indicator computation for trading signals.

All indicators are implemented using pandas/numpy directly,
no external ta-lib dependency required.
"""

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("data.features")


class FeatureEngineer:
    """Computes technical indicators from OHLCV data."""

    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to the DataFrame."""
        df = df.copy()

        df = self.add_moving_averages(df)
        df = self.add_rsi(df)
        df = self.add_macd(df)
        df = self.add_bollinger_bands(df)
        df = self.add_atr(df)
        df = self.add_obv(df)
        df = self.add_stochastic(df)
        df = self.add_fibonacci_levels(df)
        df = self.add_returns(df)

        logger.info(f"Added {len(df.columns)} total columns ({len(df.columns) - 5} indicators)")
        return df

    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add EMA and SMA indicators."""
        for period in [7, 12, 21, 26, 50, 200]:
            df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()
            df[f"sma_{period}"] = df["close"].rolling(window=period).mean()
        return df

    def add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Relative Strength Index."""
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))
        return df

    def add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD, signal line, and histogram."""
        ema_fast = df["close"].ewm(span=12, adjust=False).mean()
        ema_slow = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]
        return df

    def add_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
        """Add Bollinger Bands."""
        df["bb_middle"] = df["close"].rolling(window=period).mean()
        rolling_std = df["close"].rolling(window=period).std()
        df["bb_upper"] = df["bb_middle"] + std * rolling_std
        df["bb_lower"] = df["bb_middle"] - std * rolling_std
        bb_range = df["bb_upper"] - df["bb_lower"]
        df["bb_width"] = bb_range / df["bb_middle"]
        df["bb_pct"] = (df["close"] - df["bb_lower"]) / bb_range.replace(0, np.nan)
        return df

    def add_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average True Range."""
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = true_range.rolling(window=period).mean()
        return df

    def add_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add On-Balance Volume."""
        direction = np.sign(df["close"].diff())
        df["obv"] = (df["volume"] * direction).cumsum()
        return df

    def add_stochastic(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Stochastic Oscillator (%K and %D)."""
        low_min = df["low"].rolling(window=period).min()
        high_max = df["high"].rolling(window=period).max()
        denom = (high_max - low_min).replace(0, np.nan)
        df["stoch_k"] = ((df["close"] - low_min) / denom) * 100
        df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()
        return df

    def add_fibonacci_levels(self, df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
        """Add Fibonacci retracement levels based on rolling high/low."""
        rolling_high = df["high"].rolling(window=lookback).max()
        rolling_low = df["low"].rolling(window=lookback).min()
        diff = rolling_high - rolling_low

        df["fib_0"] = rolling_low
        df["fib_236"] = rolling_low + 0.236 * diff
        df["fib_382"] = rolling_low + 0.382 * diff
        df["fib_500"] = rolling_low + 0.500 * diff
        df["fib_618"] = rolling_low + 0.618 * diff
        df["fib_1"] = rolling_high
        return df

    def add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add return-based features."""
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        df["volatility_20"] = df["returns"].rolling(window=20).std()
        df["volume_sma_20"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"].replace(0, np.nan)
        return df

    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Get list of indicator/feature column names (excludes OHLCV)."""
        base_cols = {"open", "high", "low", "close", "volume"}
        return [c for c in df.columns if c not in base_cols]
