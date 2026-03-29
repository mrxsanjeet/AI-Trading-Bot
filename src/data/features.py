"""Technical indicator computation for trading signals."""

import numpy as np
import pandas as pd
import ta

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
            df[f"ema_{period}"] = ta.trend.ema_indicator(df["close"], window=period)
            df[f"sma_{period}"] = ta.trend.sma_indicator(df["close"], window=period)
        return df

    def add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Relative Strength Index."""
        df["rsi"] = ta.momentum.rsi(df["close"], window=period)
        return df

    def add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD, signal line, and histogram."""
        macd = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_histogram"] = macd.macd_diff()
        return df

    def add_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
        """Add Bollinger Bands."""
        bb = ta.volatility.BollingerBands(df["close"], window=period, window_dev=std)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        return df

    def add_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average True Range."""
        df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=period)
        return df

    def add_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add On-Balance Volume."""
        df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])
        return df

    def add_stochastic(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Stochastic Oscillator."""
        stoch = ta.momentum.StochasticOscillator(
            df["high"], df["low"], df["close"], window=period, smooth_window=3
        )
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()
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
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]
        return df

    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Get list of indicator/feature column names (excludes OHLCV)."""
        base_cols = {"open", "high", "low", "close", "volume"}
        return [c for c in df.columns if c not in base_cols]
