"""ML model-driven trading strategy."""

import pandas as pd

from src.ml.predict import Predictor
from src.strategy.base import BaseStrategy, TradeSignal
from src.utils.logger import get_logger

logger = get_logger("strategy.ml_strategy")


class MLStrategy(BaseStrategy):
    """Strategy that uses ML model predictions to generate signals."""

    def __init__(self, predictor: Predictor, config: dict | None = None):
        super().__init__("MLStrategy", config)
        self.predictor = predictor
        self.confidence_threshold = self.config.get("confidence_threshold", 0.65)

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> TradeSignal:
        price = df["close"].iloc[-1]

        try:
            result = self.predictor.predict(df)
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return self._hold_signal(symbol, price, f"Prediction error: {e}")

        direction = result["direction"]
        confidence = result["confidence"]
        probability = result["probability"]

        if confidence < self.confidence_threshold:
            return self._hold_signal(
                symbol, price,
                f"ML confidence too low ({confidence:.2f} < {self.confidence_threshold})"
            )

        # Use ATR for dynamic stop/take levels
        atr = df["atr"].iloc[-1] if "atr" in df.columns else price * 0.02

        if direction == "UP":
            return self._buy_signal(
                symbol=symbol,
                price=price,
                confidence=confidence,
                reason=f"ML predicts UP (prob={probability:.3f}, conf={confidence:.3f})",
                stop_loss=price - 2 * atr,
                take_profit=price + 3 * atr,
            )
        elif direction == "DOWN":
            return self._sell_signal(
                symbol=symbol,
                price=price,
                confidence=confidence,
                reason=f"ML predicts DOWN (prob={probability:.3f}, conf={confidence:.3f})",
            )

        return self._hold_signal(symbol, price, "ML prediction neutral")
