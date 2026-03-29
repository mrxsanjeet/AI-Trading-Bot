"""Inference pipeline for making predictions with trained models."""

import numpy as np
import torch
import torch.nn as nn

from src.ml.features import MLFeatureEngineer
from src.utils.logger import get_logger

logger = get_logger("ml.predict")


class Predictor:
    """Runs inference using a trained model."""

    def __init__(self, model: nn.Module, feature_engineer: MLFeatureEngineer):
        self.model = model
        self.feature_engineer = feature_engineer
        self.device = next(model.parameters()).device
        self.model.eval()

    def predict(self, df) -> dict:
        """Make a prediction on the latest data.

        Args:
            df: DataFrame with OHLCV + indicator columns.

        Returns:
            dict with: prediction (0 or 1), probability (0-1),
            direction ('UP' or 'DOWN'), confidence (0-1).
        """
        try:
            X = self.feature_engineer.prepare_inference(df)
            X_tensor = torch.FloatTensor(X).to(self.device)

            with torch.no_grad():
                probability = self.model(X_tensor).squeeze().item()

            prediction = 1 if probability > 0.5 else 0
            direction = "UP" if prediction == 1 else "DOWN"
            confidence = abs(probability - 0.5) * 2  # Map 0.5-1.0 to 0-1 confidence

            logger.info(
                f"Prediction: {direction} | Probability: {probability:.3f} | "
                f"Confidence: {confidence:.3f}"
            )

            return {
                "prediction": prediction,
                "probability": probability,
                "direction": direction,
                "confidence": confidence,
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                "prediction": 0,
                "probability": 0.5,
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "error": str(e),
            }

    def predict_batch(self, sequences: np.ndarray) -> list[dict]:
        """Make predictions on a batch of sequences."""
        X_tensor = torch.FloatTensor(sequences).to(self.device)

        with torch.no_grad():
            probabilities = self.model(X_tensor).squeeze().cpu().numpy()

        results = []
        for prob in probabilities:
            p = float(prob)
            results.append({
                "prediction": 1 if p > 0.5 else 0,
                "probability": p,
                "direction": "UP" if p > 0.5 else "DOWN",
                "confidence": abs(p - 0.5) * 2,
            })

        return results
