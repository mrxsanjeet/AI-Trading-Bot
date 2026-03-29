"""Hybrid ensemble strategy combining multiple signal sources."""

import pandas as pd

from src.strategy.base import BaseStrategy, Signal, TradeSignal
from src.strategy.signals import SignalAggregator
from src.utils.logger import get_logger

logger = get_logger("strategy.hybrid")


class HybridStrategy(BaseStrategy):
    """Ensemble strategy that combines signals from multiple sub-strategies."""

    def __init__(
        self,
        strategies: list[BaseStrategy],
        config: dict | None = None,
    ):
        super().__init__("Hybrid", config)
        self.strategies = strategies
        self.min_agreement = self.config.get("min_agreement", 2)
        self.weighted_voting = self.config.get("weighted_voting", True)
        self.aggregator = SignalAggregator(strategies, self.min_agreement)

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> TradeSignal:
        result = self.aggregator.generate_signals(df, symbol)
        aggregated = result["aggregated_signal"]

        # Log individual signals for transparency
        for name, signal in result["individual_signals"].items():
            logger.debug(f"  [{name}] {signal.signal.value} (conf: {signal.confidence:.2f})")

        logger.info(
            f"Hybrid signal: {aggregated.signal.value} | "
            f"Confidence: {aggregated.confidence:.2f} | "
            f"Reason: {aggregated.reason}"
        )

        return aggregated
