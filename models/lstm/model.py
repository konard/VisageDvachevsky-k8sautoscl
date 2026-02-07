#!/usr/bin/env python3
"""
LSTM model for time series forecasting.

Architecture:
  - Multi-layer LSTM with dropout
  - Batch normalization on input
  - Dense output head
  - Support for bidirectional LSTM
"""
import torch
import torch.nn as nn


class TimeSeriesLSTM(nn.Module):
    """LSTM model for predicting future metric values.

    Args:
        input_size: Number of input features per timestep
        hidden_size: LSTM hidden state dimensionality
        num_layers: Number of stacked LSTM layers
        dropout: Dropout rate between LSTM layers
        bidirectional: Whether to use bidirectional LSTM
        output_size: Number of output values (1 for single-step forecast)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        output_size: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Input batch normalization — helps with varying feature scales
        self.input_bn = nn.BatchNorm1d(input_size)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Output head
        lstm_output_size = hidden_size * self.num_directions
        self.output_head = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, features)

        Returns:
            Predictions of shape (batch, output_size)
        """
        batch_size, seq_len, features = x.shape

        # Apply batch norm across features (permute to batch, features, seq)
        x = x.permute(0, 2, 1)
        x = self.input_bn(x)
        x = x.permute(0, 2, 1)

        # LSTM forward
        lstm_out, _ = self.lstm(x)

        # Use last timestep output
        last_output = lstm_out[:, -1, :]

        # Dense output
        prediction = self.output_head(last_output)
        return prediction.squeeze(-1)


class TimeSeriesGRU(nn.Module):
    """GRU variant — lighter than LSTM with comparable performance.

    Args:
        input_size: Number of input features per timestep
        hidden_size: GRU hidden state dimensionality
        num_layers: Number of stacked GRU layers
        dropout: Dropout rate between GRU layers
        output_size: Number of output values
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
    ):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_size)

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, features = x.shape

        x = x.permute(0, 2, 1)
        x = self.input_bn(x)
        x = x.permute(0, 2, 1)

        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]

        prediction = self.output_head(last_output)
        return prediction.squeeze(-1)


def create_model(
    model_type: str,
    input_size: int,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    bidirectional: bool = False,
) -> nn.Module:
    """Factory function for creating a model.

    Args:
        model_type: 'lstm' or 'gru'
        input_size: Number of input features
        hidden_size: Hidden layer size
        num_layers: Number of recurrent layers
        dropout: Dropout rate
        bidirectional: Use bidirectional (LSTM only)

    Returns:
        Initialized model
    """
    if model_type == "lstm":
        return TimeSeriesLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
    elif model_type == "gru":
        return TimeSeriesGRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'lstm' or 'gru'.")
