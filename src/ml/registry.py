"""Model versioning and storage registry."""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import torch

from src.ml.features import MLFeatureEngineer
from src.ml.models import create_model
from src.utils.logger import get_logger

logger = get_logger("ml.registry")


class ModelRegistry:
    """Manages versioning, storage, and retrieval of trained models."""

    def __init__(self, registry_dir: str = "models"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.registry_dir / "manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> dict:
        """Load or create the model manifest."""
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                return json.load(f)
        return {"models": [], "latest": None}

    def _save_manifest(self) -> None:
        """Save the manifest to disk."""
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2, default=str)

    def register_model(
        self,
        model: torch.nn.Module,
        feature_engineer: MLFeatureEngineer,
        metrics: dict,
        config: dict,
        symbol: str = "all",
    ) -> str:
        """Save and register a trained model with metadata.

        Returns:
            The version string of the registered model.
        """
        version = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")
        model_dir = self.registry_dir / version
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights
        model_path = model_dir / "model.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_type": config.get("model_type", "lstm"),
            "config": config,
        }, model_path)

        # Save feature engineer (scaler + columns)
        scaler_path = model_dir / "scaler.joblib"
        joblib.dump(feature_engineer.scaler, scaler_path)

        feature_meta = {
            "feature_columns": feature_engineer.feature_columns,
            "lookback_window": feature_engineer.lookback_window,
        }
        with open(model_dir / "features.json", "w") as f:
            json.dump(feature_meta, f, indent=2)

        # Update manifest
        entry = {
            "version": version,
            "model_type": config.get("model_type", "lstm"),
            "symbol": symbol,
            "metrics": metrics,
            "config": config,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "path": str(model_dir),
        }
        self.manifest["models"].append(entry)
        self.manifest["latest"] = version
        self._save_manifest()

        logger.info(f"Model registered: {version} | Accuracy: {metrics.get('best_accuracy', 'N/A')}")
        return version

    def load_model(self, version: str | None = None) -> dict:
        """Load a model by version (or latest).

        Returns:
            dict with: model, feature_engineer, metadata.
        """
        if version is None:
            version = self.manifest.get("latest")
        if version is None:
            raise ValueError("No models registered")

        model_dir = self.registry_dir / version
        if not model_dir.exists():
            raise FileNotFoundError(f"Model version {version} not found")

        # Load model
        save_data = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=False)
        model_type = save_data["model_type"]
        config = save_data.get("config", {})

        # Load feature metadata
        with open(model_dir / "features.json") as f:
            feature_meta = json.load(f)

        input_size = len(feature_meta["feature_columns"])
        model = create_model(model_type, input_size, config)
        model.load_state_dict(save_data["model_state_dict"])
        model.eval()

        # Load scaler
        scaler = joblib.load(model_dir / "scaler.joblib")
        feature_engineer = MLFeatureEngineer(lookback_window=feature_meta["lookback_window"])
        feature_engineer.scaler = scaler
        feature_engineer.feature_columns = feature_meta["feature_columns"]

        # Find metadata
        entry = next((m for m in self.manifest["models"] if m["version"] == version), {})

        logger.info(f"Model loaded: {version}")
        return {"model": model, "feature_engineer": feature_engineer, "metadata": entry}

    def list_models(self) -> list[dict]:
        """List all registered models."""
        return self.manifest.get("models", [])

    def get_best_model(self, metric: str = "best_accuracy") -> str | None:
        """Get the version of the best-performing model by a given metric."""
        models = self.manifest.get("models", [])
        if not models:
            return None

        best = max(models, key=lambda m: m.get("metrics", {}).get(metric, 0))
        return best["version"]
