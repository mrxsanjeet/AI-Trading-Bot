"""Signal generation and aggregation from multiple strategies."""

import pandas as pd

from src.strategy.base import BaseStrategy, Signal, TradeSignal
from src.utils.logger import get_logger

logger = get_logger("strategy.signals")


class SignalAggregator:
    """Aggregates signals from multiple strategies using weighted voting."""

    def __init__(self, strategies: list[BaseStrategy], min_agreement: int = 2):
        self.strategies = [s for s in strategies if s.enabled]
        self.min_agreement = min_agreement
        logger.info(
            f"SignalAggregator initialized with {len(self.strategies)} strategies, "
            f"min_agreement={min_agreement}"
        )

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> dict:
        """Generate signals from all strategies and aggregate.

        Returns:
            dict with 'individual_signals', 'aggregated_signal', and 'details'.
        """
        individual_signals = {}
        for strategy in self.strategies:
            try:
                signal = strategy.generate_signal(df, symbol)
                individual_signals[strategy.name] = signal
                logger.debug(f"  {signal}")
            except Exception as e:
                logger.error(f"Error in strategy {strategy.name}: {e}")
                individual_signals[strategy.name] = strategy._hold_signal(
                    symbol, df["close"].iloc[-1], f"Error: {e}"
                )

        aggregated = self._aggregate(individual_signals, symbol, df["close"].iloc[-1])

        return {
            "individual_signals": individual_signals,
            "aggregated_signal": aggregated,
            "timestamp": df.index[-1] if len(df) > 0 else None,
        }

    def _aggregate(
        self, signals: dict[str, TradeSignal], symbol: str, price: float
    ) -> TradeSignal:
        """Aggregate multiple signals using weighted voting."""
        buy_weight = 0.0
        sell_weight = 0.0
        buy_signals = []
        sell_signals = []

        for name, signal in signals.items():
            strategy = next((s for s in self.strategies if s.name == name), None)
            weight = strategy.weight if strategy else 1.0

            if signal.signal == Signal.BUY:
                buy_weight += weight * signal.confidence
                buy_signals.append(name)
            elif signal.signal == Signal.SELL:
                sell_weight += weight * signal.confidence
                sell_signals.append(name)

        total_weight = sum(s.weight for s in self.strategies)

        # Determine final signal
        if len(buy_signals) >= self.min_agreement and buy_weight > sell_weight:
            confidence = buy_weight / total_weight
            # Use stop/take from highest-confidence buy signal
            best_buy = max(
                [s for s in signals.values() if s.signal == Signal.BUY],
                key=lambda s: s.confidence,
            )
            return TradeSignal(
                signal=Signal.BUY,
                symbol=symbol,
                price=price,
                confidence=min(confidence, 0.95),
                strategy="Aggregated",
                reason=f"BUY consensus from: {', '.join(buy_signals)}",
                stop_loss=best_buy.stop_loss,
                take_profit=best_buy.take_profit,
            )

        if len(sell_signals) >= self.min_agreement and sell_weight > buy_weight:
            confidence = sell_weight / total_weight
            return TradeSignal(
                signal=Signal.SELL,
                symbol=symbol,
                price=price,
                confidence=min(confidence, 0.95),
                strategy="Aggregated",
                reason=f"SELL consensus from: {', '.join(sell_signals)}",
            )

        return TradeSignal(
            signal=Signal.HOLD,
            symbol=symbol,
            price=price,
            confidence=0.0,
            strategy="Aggregated",
            reason="No consensus reached",
        )
