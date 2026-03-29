"""Training pipeline for ML models."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.ml.models import create_model
from src.utils.logger import get_logger

logger = get_logger("ml.train")


class ModelTrainer:
    """Handles training, evaluation, and saving of ML models."""

    def __init__(self, config: dict):
        self.config = config
        self.model_type = config.get("model_type", "lstm")
        self.epochs = config.get("epochs", 50)
        self.batch_size = config.get("batch_size", 32)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"ModelTrainer initialized | device: {self.device} | model: {self.model_type}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict:
        """Train the model and return training metrics.

        Returns:
            dict with: model, train_losses, val_losses, best_accuracy.
        """
        input_size = X_train.shape[2]  # Number of features

        model = create_model(self.model_type, input_size, self.config)
        model = model.to(self.device)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        train_loader = self._create_dataloader(X_train, y_train)
        val_loader = self._create_dataloader(X_val, y_val)

        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        best_accuracy = 0.0
        best_model_state = None
        patience_counter = 0
        patience = 10

        for epoch in range(self.epochs):
            # Training phase
            model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                predictions = model(X_batch).squeeze()
                loss = criterion(predictions, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation phase
            val_loss, val_accuracy = self._evaluate(model, val_loader, criterion)
            val_losses.append(val_loss)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_accuracy = val_accuracy
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{self.epochs} | "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_accuracy:.2%}"
                )

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Restore best model
        if best_model_state:
            model.load_state_dict(best_model_state)

        logger.info(f"Training complete | Best Val Accuracy: {best_accuracy:.2%}")

        return {
            "model": model,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_accuracy": best_accuracy,
            "best_val_loss": best_val_loss,
            "epochs_trained": len(train_losses),
        }

    def _evaluate(
        self, model: nn.Module, dataloader: DataLoader, criterion: nn.Module
    ) -> tuple[float, float]:
        """Evaluate model on a dataset."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                predictions = model(X_batch).squeeze()
                loss = criterion(predictions, y_batch)
                total_loss += loss.item()

                predicted_labels = (predictions > 0.5).float()
                correct += (predicted_labels == y_batch).sum().item()
                total += y_batch.size(0)

        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        accuracy = correct / total if total > 0 else 0

        return avg_loss, accuracy

    def _create_dataloader(self, X: np.ndarray, y: np.ndarray) -> DataLoader:
        """Create a PyTorch DataLoader from numpy arrays."""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def save_model(self, model: nn.Module, path: str, metadata: dict | None = None) -> None:
        """Save model weights and metadata."""
        save_dir = Path(path).parent
        save_dir.mkdir(parents=True, exist_ok=True)

        save_data = {
            "model_state_dict": model.state_dict(),
            "model_type": self.model_type,
            "config": self.config,
        }
        if metadata:
            save_data["metadata"] = metadata

        torch.save(save_data, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str, input_size: int) -> nn.Module:
        """Load a saved model."""
        save_data = torch.load(path, map_location=self.device, weights_only=False)

        model = create_model(save_data["model_type"], input_size, save_data.get("config", {}))
        model.load_state_dict(save_data["model_state_dict"])
        model = model.to(self.device)
        model.eval()

        logger.info(f"Model loaded from {path}")
        return model
