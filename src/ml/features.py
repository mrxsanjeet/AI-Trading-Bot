"""ML feature engineering — preparing data for model training and inference."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

logger = get_logger("ml.features")


class MLFeatureEngineer:
    """Prepares features for ML model training and inference."""

    def __init__(self, lookback_window: int = 60):
        self.lookback_window = lookback_window
        self.scaler = StandardScaler()
        self.feature_columns: list[str] = []

    def prepare_features(self, df: pd.DataFrame, fit_scaler: bool = True) -> dict:
        """Prepare feature matrix and labels from OHLCV + indicators DataFrame.

        Returns:
            dict with keys: X (np.ndarray), y (np.ndarray), feature_columns (list),
            scaler (StandardScaler), timestamps (list).
        """
        df = df.copy().dropna()

        # Select feature columns (everything except base OHLCV)
        exclude = {"open", "high", "low", "close", "volume"}
        self.feature_columns = [c for c in df.columns if c not in exclude]

        # Add normalized OHLCV
        self.feature_columns = list(df.columns)

        # Create binary label: 1 if next close > current close, else 0
        df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
        df.dropna(inplace=True)

        # Scale features
        feature_data = df[self.feature_columns].values
        if fit_scaler:
            feature_data = self.scaler.fit_transform(feature_data)
        else:
            feature_data = self.scaler.transform(feature_data)

        labels = df["target"].values
        timestamps = df.index.tolist()

        # Create sequences for LSTM/Transformer input
        X, y, ts = self._create_sequences(feature_data, labels, timestamps)

        logger.info(
            f"Prepared {len(X)} sequences | "
            f"Features: {len(self.feature_columns)} | "
            f"Window: {self.lookback_window}"
        )

        return {
            "X": X,
            "y": y,
            "feature_columns": self.feature_columns,
            "timestamps": ts,
        }

    def _create_sequences(
        self, data: np.ndarray, labels: np.ndarray, timestamps: list
    ) -> tuple[np.ndarray, np.ndarray, list]:
        """Create sliding window sequences for time series models."""
        X, y, ts = [], [], []
        for i in range(self.lookback_window, len(data)):
            X.append(data[i - self.lookback_window : i])
            y.append(labels[i])
            ts.append(timestamps[i])

        return np.array(X), np.array(y), ts

    def prepare_inference(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for inference (single prediction).

        Takes the last `lookback_window` rows and returns a scaled sequence.
        """
        df = df.copy().dropna()

        if len(df) < self.lookback_window:
            raise ValueError(
                f"Need at least {self.lookback_window} rows for inference, got {len(df)}"
            )

        feature_data = df[self.feature_columns].values[-self.lookback_window :]
        scaled = self.scaler.transform(feature_data)

        return scaled.reshape(1, self.lookback_window, -1)  # (1, seq_len, features)

    def train_test_split(
        self, X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data chronologically (no shuffling for time series)."""
        split_idx = int(len(X) * (1 - test_ratio))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        logger.info(f"Train/test split: {len(X_train)} train, {len(X_test)} test")
        return X_train, X_test, y_train, y_test
