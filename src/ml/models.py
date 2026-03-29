"""Neural network model architectures for price prediction."""

import math

import torch
import torch.nn as nn


class LSTMPredictor(nn.Module):
    """LSTM-based model for time series price prediction.

    Predicts the probability of price going up in the next candle.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            Probability tensor of shape (batch_size, 1).
        """
        lstm_out, _ = self.lstm(x)
        # Use the last time step output
        last_hidden = lstm_out[:, -1, :]
        return self.classifier(last_hidden)


class TransformerPredictor(nn.Module):
    """Transformer-based model for time series price prediction."""

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dropout: float = 0.2,
        max_seq_len: int = 200,
    ):
        super().__init__()
        self.d_model = d_model

        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size).
        """
        x = self.input_projection(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        # Use the last time step
        x = x[:, -1, :]
        return self.classifier(x)


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer models."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 200):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


def create_model(model_type: str, input_size: int, config: dict) -> nn.Module:
    """Factory function to create a model by type."""
    if model_type == "lstm":
        return LSTMPredictor(
            input_size=input_size,
            hidden_size=config.get("hidden_size", 128),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.2),
        )
    elif model_type == "transformer":
        return TransformerPredictor(
            input_size=input_size,
            d_model=config.get("hidden_size", 128),
            nhead=config.get("nhead", 8),
            num_layers=config.get("num_layers", 3),
            dropout=config.get("dropout", 0.2),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Must be 'lstm' or 'transformer'.")
