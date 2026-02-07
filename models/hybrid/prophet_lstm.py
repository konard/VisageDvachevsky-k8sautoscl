#!/usr/bin/env python3
"""
Hybrid Prophet+LSTM model for time series forecasting.

Architecture:
  Stage 1: Prophet captures trend + seasonality
  Stage 2: LSTM learns to predict residuals (actual - prophet_prediction)
  Combined: y_pred = prophet_pred + lstm_residual_pred

This approach leverages Prophet's strength in decomposing time series
into interpretable components and LSTM's ability to capture complex
sequential patterns in the residuals.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lstm"))
from model import TimeSeriesLSTM, create_model


class HybridProphetLSTM:
    """Hybrid model combining Prophet trend/seasonality with LSTM residuals.

    The pipeline:
    1. Prophet generates base predictions (trend + seasonality)
    2. Residuals = actual - prophet_predictions
    3. LSTM is trained to predict these residuals from sequence features
    4. Final prediction = prophet_prediction + lstm_residual_prediction
    """

    def __init__(
        self,
        prophet_model=None,
        lstm_model: nn.Module | None = None,
        scaler=None,
        target_metric: str = "cpu_usage",
        seq_length: int = 24,
        device: torch.device | None = None,
    ):
        self.prophet_model = prophet_model
        self.lstm_model = lstm_model
        self.scaler = scaler
        self.target_metric = target_metric
        self.seq_length = seq_length
        self.device = device or torch.device("cpu")

    def compute_prophet_predictions(self, df: pd.DataFrame) -> np.ndarray:
        """Generate Prophet predictions for the given dataframe.

        Args:
            df: DataFrame with a datetime index

        Returns:
            Array of Prophet predictions (in normalized scale)
        """
        # Prepare Prophet format
        prophet_df = pd.DataFrame({
            "ds": df.index.tz_localize(None) if df.index.tz else df.index,
            "y": df[self.target_metric].values if self.target_metric in df.columns else 0,
        })

        forecast = self.prophet_model.predict(prophet_df)
        return forecast["yhat"].values

    def compute_residuals(
        self, actual: np.ndarray, prophet_preds: np.ndarray
    ) -> np.ndarray:
        """Compute residuals: actual - prophet_prediction."""
        return actual - prophet_preds

    def build_residual_sequences(
        self,
        features: np.ndarray,
        residuals: np.ndarray,
        stride: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build sliding-window sequences where the target is the residual.

        Args:
            features: Feature matrix (n_samples, n_features)
            residuals: Residual values to predict
            stride: Step between windows

        Returns:
            (sequences, residual_targets)
        """
        seqs, tgts = [], []
        for i in range(0, len(features) - self.seq_length, stride):
            seqs.append(features[i : i + self.seq_length])
            tgts.append(residuals[i + self.seq_length - 1])

        return np.array(seqs, dtype=np.float32), np.array(tgts, dtype=np.float32)

    def predict_residuals(self, sequences: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """Predict residuals using the LSTM model.

        Args:
            sequences: Input sequences (n_samples, seq_length, n_features)
            batch_size: Batch size for inference

        Returns:
            Predicted residuals
        """
        self.lstm_model.eval()
        predictions = []

        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch = torch.FloatTensor(sequences[i : i + batch_size]).to(self.device)
                pred = self.lstm_model(batch)
                predictions.append(pred.cpu().numpy())

        return np.concatenate(predictions)

    def predict(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        stride: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate hybrid predictions.

        Args:
            df: DataFrame with features and datetime index
            feature_cols: Feature column names for LSTM
            stride: Sliding window stride

        Returns:
            (hybrid_predictions, prophet_predictions, lstm_residual_predictions)
            All aligned to the sequence timestamps
        """
        # Stage 1: Prophet predictions
        prophet_preds_full = self.compute_prophet_predictions(df)

        # Build sequences for LSTM
        features = df[feature_cols].values.astype(np.float32)
        seqs = []
        indices = []
        for i in range(0, len(features) - self.seq_length, stride):
            seqs.append(features[i : i + self.seq_length])
            indices.append(i + self.seq_length - 1)

        sequences = np.array(seqs, dtype=np.float32)

        # Stage 2: LSTM residual predictions
        lstm_residuals = self.predict_residuals(sequences)

        # Align prophet predictions to sequence indices
        prophet_preds_aligned = prophet_preds_full[indices]

        # Combined prediction
        hybrid_preds = prophet_preds_aligned + lstm_residuals

        return hybrid_preds, prophet_preds_aligned, lstm_residuals


def load_hybrid_model(
    prophet_path: Path,
    lstm_path: Path,
    scaler_path: Path | None = None,
    device: torch.device | None = None,
) -> HybridProphetLSTM:
    """Load a trained hybrid model from saved artifacts.

    Args:
        prophet_path: Path to prophet_model.pkl
        lstm_path: Path to LSTM best_model.pt
        scaler_path: Path to scaler.pkl (optional)
        device: Torch device

    Returns:
        Initialized HybridProphetLSTM
    """
    device = device or torch.device("cpu")

    # Load Prophet
    prophet_model = joblib.load(prophet_path)

    # Load LSTM
    checkpoint = torch.load(lstm_path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]

    lstm_model = create_model(
        model_type=config["model_type"],
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        bidirectional=config.get("bidirectional", False),
    )
    lstm_model.load_state_dict(checkpoint["model_state_dict"])
    lstm_model = lstm_model.to(device)
    lstm_model.eval()

    # Load scaler
    scaler = None
    if scaler_path and scaler_path.exists():
        scaler = joblib.load(scaler_path)

    return HybridProphetLSTM(
        prophet_model=prophet_model,
        lstm_model=lstm_model,
        scaler=scaler,
        device=device,
    )
