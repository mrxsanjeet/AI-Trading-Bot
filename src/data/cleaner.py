"""Data cleaning and normalization for market data."""

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("data.cleaner")


class DataCleaner:
    """Cleans and normalizes OHLCV market data."""

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all cleaning steps on OHLCV data."""
        df = df.copy()
        initial_len = len(df)

        df = self._remove_duplicates(df)
        df = self._handle_missing_values(df)
        df = self._remove_outliers(df)
        df = self._ensure_types(df)
        df = self._sort_index(df)

        removed = initial_len - len(df)
        if removed > 0:
            logger.info(f"Cleaned data: removed {removed} rows ({initial_len} -> {len(df)})")

        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate index entries, keeping the last."""
        duplicates = df.index.duplicated(keep="last")
        if duplicates.any():
            logger.debug(f"Removing {duplicates.sum()} duplicate timestamps")
            df = df[~duplicates]
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in OHLCV data."""
        if df.isnull().any().any():
            missing_counts = df.isnull().sum()
            logger.debug(f"Missing values found: {missing_counts[missing_counts > 0].to_dict()}")

            # Forward-fill price data (use last known price)
            price_cols = ["open", "high", "low", "close"]
            existing_price_cols = [c for c in price_cols if c in df.columns]
            df[existing_price_cols] = df[existing_price_cols].ffill()

            # Fill volume with 0 for missing entries
            if "volume" in df.columns:
                df["volume"] = df["volume"].fillna(0)

            # Drop any remaining rows with NaN
            df.dropna(inplace=True)

        return df

    def _remove_outliers(self, df: pd.DataFrame, z_threshold: float = 10.0) -> pd.DataFrame:
        """Remove extreme outliers using z-score on returns."""
        if len(df) < 20:
            return df

        returns = df["close"].pct_change().dropna()
        if len(returns) == 0:
            return df

        z_scores = np.abs((returns - returns.mean()) / returns.std())
        outlier_mask = z_scores > z_threshold

        if outlier_mask.any():
            outlier_count = outlier_mask.sum()
            logger.warning(f"Removing {outlier_count} outlier candles (z-score > {z_threshold})")
            valid_indices = returns[~outlier_mask].index
            # Keep the first row (which has no return) plus non-outlier rows
            df = df.loc[df.index.isin(valid_indices) | (df.index == df.index[0])]

        return df

    def _ensure_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all columns have correct types."""
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _sort_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure data is sorted by timestamp."""
        return df.sort_index()

    def validate(self, df: pd.DataFrame) -> list[str]:
        """Validate OHLCV data and return list of issues found."""
        issues = []

        if df.empty:
            issues.append("DataFrame is empty")
            return issues

        required_cols = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            issues.append(f"Missing columns: {missing}")

        if df.isnull().any().any():
            issues.append(f"Contains NaN values: {df.isnull().sum().to_dict()}")

        if not df.index.is_monotonic_increasing:
            issues.append("Index is not sorted chronologically")

        # Check high >= low
        if "high" in df.columns and "low" in df.columns:
            invalid = (df["high"] < df["low"]).sum()
            if invalid > 0:
                issues.append(f"{invalid} candles have high < low")

        # Check for negative volumes
        if "volume" in df.columns:
            neg_vol = (df["volume"] < 0).sum()
            if neg_vol > 0:
                issues.append(f"{neg_vol} candles have negative volume")

        return issues
