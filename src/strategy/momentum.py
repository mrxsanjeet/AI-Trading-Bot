"""Momentum and breakout trading strategies."""

import pandas as pd

from src.strategy.base import BaseStrategy, TradeSignal
from src.utils.logger import get_logger

logger = get_logger("strategy.momentum")


class MomentumStrategy(BaseStrategy):
    """Momentum strategy using breakouts and volume spikes."""

    def __init__(self, config: dict | None = None):
        super().__init__("Momentum", config)
        self.breakout_period = self.config.get("breakout_period", 20)
        self.volume_multiplier = self.config.get("volume_multiplier", 1.5)
        self.atr_period = self.config.get("atr_period", 14)

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> TradeSignal:
        if len(df) < self.breakout_period + 5:
            return self._hold_signal(symbol, df["close"].iloc[-1], "Insufficient data")

        current = df.iloc[-1]
        price = current["close"]

        # Calculate breakout levels
        lookback = df.iloc[-self.breakout_period - 1 : -1]
        resistance = lookback["high"].max()
        support = lookback["low"].min()

        reasons = []
        confidence = 0.0

        # Breakout detection
        is_bullish_breakout = price > resistance
        is_bearish_breakout = price < support

        # Volume confirmation
        volume_ratio = current.get("volume_ratio", 1.0)
        has_volume = volume_ratio is not None and volume_ratio > self.volume_multiplier

        # Stochastic momentum
        stoch_k = current.get("stoch_k")
        stoch_d = current.get("stoch_d")

        atr = current.get("atr", price * 0.02)

        if is_bullish_breakout:
            confidence = 0.55
            reasons.append(f"Bullish breakout above {resistance:.2f}")

            if has_volume:
                confidence += 0.2
                reasons.append(f"Volume spike ({volume_ratio:.1f}x avg)")

            if stoch_k is not None and stoch_k > stoch_d:
                confidence += 0.1
                reasons.append("Stochastic confirms momentum")

            # Stronger breakout = more confidence
            breakout_strength = (price - resistance) / atr
            confidence += min(breakout_strength * 0.05, 0.1)
            confidence = min(confidence, 0.95)

            return self._buy_signal(
                symbol=symbol,
                price=price,
                confidence=confidence,
                reason=", ".join(reasons),
                stop_loss=resistance - 0.5 * atr,  # Just below breakout level
                take_profit=price + 3 * atr,
            )

        if is_bearish_breakout:
            confidence = 0.55
            reasons.append(f"Bearish breakdown below {support:.2f}")

            if has_volume:
                confidence += 0.2
                reasons.append(f"Volume spike ({volume_ratio:.1f}x avg)")

            if stoch_k is not None and stoch_k < stoch_d:
                confidence += 0.1
                reasons.append("Stochastic confirms momentum")

            confidence = min(confidence, 0.95)

            return self._sell_signal(
                symbol=symbol,
                price=price,
                confidence=confidence,
                reason=", ".join(reasons),
            )

        return self._hold_signal(symbol, price, "No breakout detected")
