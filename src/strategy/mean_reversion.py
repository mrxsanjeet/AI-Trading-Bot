"""Mean reversion trading strategies."""

import pandas as pd

from src.strategy.base import BaseStrategy, TradeSignal
from src.utils.logger import get_logger

logger = get_logger("strategy.mean_reversion")


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy using Bollinger Bands and RSI extremes."""

    def __init__(self, config: dict | None = None):
        super().__init__("MeanReversion", config)
        self.rsi_oversold = self.config.get("rsi_oversold", 30)
        self.rsi_overbought = self.config.get("rsi_overbought", 70)
        self.bb_period = self.config.get("bb_period", 20)

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> TradeSignal:
        if len(df) < self.bb_period + 5:
            return self._hold_signal(symbol, df["close"].iloc[-1], "Insufficient data")

        current = df.iloc[-1]
        price = current["close"]

        rsi = current.get("rsi")
        bb_lower = current.get("bb_lower")
        bb_upper = current.get("bb_upper")
        bb_middle = current.get("bb_middle")
        bb_pct = current.get("bb_pct")

        if rsi is None or bb_lower is None:
            return self._hold_signal(symbol, price, "Missing indicators")

        reasons = []
        buy_score = 0.0
        sell_score = 0.0

        # RSI extreme check
        if rsi < self.rsi_oversold:
            buy_score += 0.4
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi > self.rsi_overbought:
            sell_score += 0.4
            reasons.append(f"RSI overbought ({rsi:.1f})")

        # Bollinger Band check
        if price <= bb_lower:
            buy_score += 0.4
            reasons.append("Price at lower Bollinger Band")
        elif price >= bb_upper:
            sell_score += 0.4
            reasons.append("Price at upper Bollinger Band")

        # BB percentage position (how far from mean)
        if bb_pct is not None:
            if bb_pct < 0.1:
                buy_score += 0.2
            elif bb_pct > 0.9:
                sell_score += 0.2

        # Volume confirmation — higher volume at extremes increases confidence
        volume_ratio = current.get("volume_ratio", 1.0)
        if volume_ratio and volume_ratio > 1.5:
            buy_score *= 1.15
            sell_score *= 1.15

        atr = current.get("atr", price * 0.02)

        if buy_score >= 0.5:
            return self._buy_signal(
                symbol=symbol,
                price=price,
                confidence=min(buy_score, 0.95),
                reason=", ".join(reasons) or "Mean reversion buy",
                stop_loss=price - 1.5 * atr,
                take_profit=bb_middle if bb_middle else price + 2 * atr,
            )

        if sell_score >= 0.5:
            return self._sell_signal(
                symbol=symbol,
                price=price,
                confidence=min(sell_score, 0.95),
                reason=", ".join(reasons) or "Mean reversion sell",
            )

        return self._hold_signal(symbol, price, "No mean reversion signal")
