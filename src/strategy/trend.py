"""Trend-following trading strategies."""

import pandas as pd

from src.strategy.base import BaseStrategy, TradeSignal
from src.utils.logger import get_logger

logger = get_logger("strategy.trend")


class TrendFollowingStrategy(BaseStrategy):
    """Trend-following strategy using EMA crossover and MACD confirmation."""

    def __init__(self, config: dict | None = None):
        super().__init__("TrendFollowing", config)
        self.ema_short = self.config.get("ema_short", 12)
        self.ema_long = self.config.get("ema_long", 26)

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> TradeSignal:
        if len(df) < self.ema_long + 2:
            return self._hold_signal(symbol, df["close"].iloc[-1], "Insufficient data")

        current = df.iloc[-1]
        prev = df.iloc[-2]
        price = current["close"]

        ema_short_col = f"ema_{self.ema_short}"
        ema_long_col = f"ema_{self.ema_long}"

        if ema_short_col not in df.columns or ema_long_col not in df.columns:
            return self._hold_signal(symbol, price, "Missing EMA indicators")

        ema_short_now = current[ema_short_col]
        ema_long_now = current[ema_long_col]
        ema_short_prev = prev[ema_short_col]
        ema_long_prev = prev[ema_long_col]

        # EMA crossover detection
        bullish_crossover = ema_short_prev <= ema_long_prev and ema_short_now > ema_long_now
        bearish_crossover = ema_short_prev >= ema_long_prev and ema_short_now < ema_long_now

        # MACD confirmation
        macd_bullish = current.get("macd_histogram", 0) > 0
        macd_bearish = current.get("macd_histogram", 0) < 0

        # Calculate confidence based on trend strength
        ema_spread = abs(ema_short_now - ema_long_now) / ema_long_now
        base_confidence = min(0.5 + ema_spread * 50, 0.95)

        if bullish_crossover:
            confidence = base_confidence + (0.15 if macd_bullish else 0.0)
            confidence = min(confidence, 0.95)
            reasons = ["EMA bullish crossover"]
            if macd_bullish:
                reasons.append("MACD confirms")

            atr = current.get("atr", price * 0.02)
            return self._buy_signal(
                symbol=symbol,
                price=price,
                confidence=confidence,
                reason=", ".join(reasons),
                stop_loss=price - 2 * atr,
                take_profit=price + 3 * atr,
            )

        if bearish_crossover:
            confidence = base_confidence + (0.15 if macd_bearish else 0.0)
            confidence = min(confidence, 0.95)
            reasons = ["EMA bearish crossover"]
            if macd_bearish:
                reasons.append("MACD confirms")

            return self._sell_signal(
                symbol=symbol,
                price=price,
                confidence=confidence,
                reason=", ".join(reasons),
            )

        # Trend continuation signals
        if ema_short_now > ema_long_now and macd_bullish:
            trend_strength = ema_spread * 30
            if trend_strength > 0.3:
                return self._hold_signal(symbol, price, f"Uptrend intact (strength: {trend_strength:.2f})")

        return self._hold_signal(symbol, price, "No trend signal")
