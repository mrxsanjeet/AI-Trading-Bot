"""Abstract base class for all trading strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import pandas as pd


class Signal(Enum):
    """Trading signal types."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradeSignal:
    """Represents a trading signal with metadata."""

    signal: Signal
    symbol: str
    price: float
    confidence: float  # 0.0 to 1.0
    strategy: str
    reason: str
    stop_loss: float | None = None
    take_profit: float | None = None
    timestamp: str | None = None

    def __str__(self) -> str:
        return (
            f"[{self.strategy}] {self.signal.value} {self.symbol} "
            f"@ {self.price:.2f} (confidence: {self.confidence:.1%}) — {self.reason}"
        )


class BaseStrategy(ABC):
    """Abstract base class that all strategies must implement."""

    def __init__(self, name: str, config: dict | None = None):
        self.name = name
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.weight = self.config.get("weight", 1.0)

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> TradeSignal:
        """Analyze market data and produce a trading signal.

        Args:
            df: DataFrame with OHLCV data and technical indicators.
            symbol: The trading pair symbol.

        Returns:
            A TradeSignal with BUY, SELL, or HOLD.
        """
        ...

    def _hold_signal(self, symbol: str, price: float, reason: str = "No clear signal") -> TradeSignal:
        """Helper to create a HOLD signal."""
        return TradeSignal(
            signal=Signal.HOLD,
            symbol=symbol,
            price=price,
            confidence=0.0,
            strategy=self.name,
            reason=reason,
        )

    def _buy_signal(
        self, symbol: str, price: float, confidence: float, reason: str,
        stop_loss: float | None = None, take_profit: float | None = None,
    ) -> TradeSignal:
        """Helper to create a BUY signal."""
        return TradeSignal(
            signal=Signal.BUY,
            symbol=symbol,
            price=price,
            confidence=confidence,
            strategy=self.name,
            reason=reason,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    def _sell_signal(
        self, symbol: str, price: float, confidence: float, reason: str,
        stop_loss: float | None = None, take_profit: float | None = None,
    ) -> TradeSignal:
        """Helper to create a SELL signal."""
        return TradeSignal(
            signal=Signal.SELL,
            symbol=symbol,
            price=price,
            confidence=confidence,
            strategy=self.name,
            reason=reason,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
