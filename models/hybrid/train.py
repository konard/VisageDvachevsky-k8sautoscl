#!/usr/bin/env python3
"""
Train the hybrid Prophet+LSTM model.

Pipeline:
  1. Load pre-trained Prophet model
  2. Generate Prophet predictions on train/val data
  3. Compute residuals (actual - prophet_pred)
  4. Train a new LSTM to predict these residuals
  5. Save the residual-LSTM model

Usage:
    python models/hybrid/train.py
    python models/hybrid/train.py --hidden-size 128 --epochs 200
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lstm"))
from model import create_model


def load_prophet_model(path: Path):
    """Load the pre-trained Prophet model."""
    print(f"Loading Prophet model from {path}...")
    model = joblib.load(path)
    return model


def generate_prophet_predictions(prophet_model, df: pd.DataFrame, target_col: str) -> np.ndarray:
    """Generate Prophet predictions for a dataframe."""
    timestamps = pd.to_datetime(df.index)
    if timestamps.tz is not None:
        timestamps = timestamps.tz_localize(None)

    prophet_df = pd.DataFrame({
        "ds": timestamps,
        "y": df[target_col].values,
    })

    forecast = prophet_model.predict(prophet_df)
    return forecast["yhat"].values


def build_residual_sequences(
    features: np.ndarray,
    residuals: np.ndarray,
    seq_length: int = 24,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding-window sequences where target = residual."""
    seqs, tgts = [], []
    for i in range(0, len(features) - seq_length, stride):
        seqs.append(features[i : i + seq_length])
        tgts.append(residuals[i + seq_length - 1])
    return np.array(seqs, dtype=np.float32), np.array(tgts, dtype=np.float32)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)
            total_loss += loss.item()
            n += 1
    return total_loss / max(n, 1)


def main():
    parser = argparse.ArgumentParser(description="Train hybrid Prophet+LSTM model")
    parser.add_argument(
        "--prophet-model",
        type=Path,
        default=Path("models/prophet/artifacts/prophet_model.pkl"),
    )
    parser.add_argument("--train-data", type=Path, default=Path("data/processed/train.csv"))
    parser.add_argument("--val-data", type=Path, default=Path("data/processed/validation.csv"))
    parser.add_argument("--target", default="cpu_usage")
    parser.add_argument("--output-dir", type=Path, default=Path("models/hybrid/artifacts"))
    parser.add_argument("--seq-length", type=int, default=24)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 70)
    print("HYBRID PROPHET+LSTM TRAINING")
    print("=" * 70)
    print(f"Device: {device}")

    # Load metadata
    meta_path = Path("data/processed/metadata.json")
    with open(meta_path) as f:
        metadata = json.load(f)
    feature_cols = metadata["feature_columns"]
    target_col_name = f"target_{args.target}_t+{metadata['forecast_horizon']}"

    # Stage 1: Load Prophet and compute residuals
    print("\n[Stage 1] Prophet predictions...")
    prophet_model = load_prophet_model(args.prophet_model)

    # Load train and val data
    train_df = pd.read_csv(args.train_data, index_col="timestamp", parse_dates=True)
    val_df = pd.read_csv(args.val_data, index_col="timestamp", parse_dates=True)

    print(f"  Train: {train_df.shape}, Val: {val_df.shape}")

    # Generate Prophet predictions (Prophet works on the target directly)
    train_prophet_preds = generate_prophet_predictions(prophet_model, train_df, args.target)
    val_prophet_preds = generate_prophet_predictions(prophet_model, val_df, args.target)

    # Compute actual target values (from the forecast target column)
    train_actual = train_df[target_col_name].values
    val_actual = val_df[target_col_name].values

    # Compute residuals
    train_residuals = train_actual - train_prophet_preds
    val_residuals = val_actual - val_prophet_preds

    print(f"  Train residuals: mean={train_residuals.mean():.4f}, std={train_residuals.std():.4f}")
    print(f"  Val residuals:   mean={val_residuals.mean():.4f}, std={val_residuals.std():.4f}")

    # Stage 2: Build sequences with residual targets
    print("\n[Stage 2] Building residual sequences...")
    train_features = train_df[feature_cols].values.astype(np.float32)
    val_features = val_df[feature_cols].values.astype(np.float32)

    train_seqs, train_res_targets = build_residual_sequences(
        train_features, train_residuals, args.seq_length, args.stride
    )
    val_seqs, val_res_targets = build_residual_sequences(
        val_features, val_residuals, args.seq_length, args.stride
    )

    print(f"  Train sequences: {train_seqs.shape}")
    print(f"  Val sequences:   {val_seqs.shape}")
    print(f"  Residual target range: [{train_res_targets.min():.4f}, {train_res_targets.max():.4f}]")

    # Create dataloaders
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(train_seqs), torch.FloatTensor(train_res_targets)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(val_seqs), torch.FloatTensor(val_res_targets)),
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Stage 3: Train residual LSTM
    print("\n[Stage 3] Training residual LSTM...")
    input_size = train_seqs.shape[2]
    model = create_model(
        model_type="lstm",
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Residual LSTM params: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    print("\n" + "-" * 70)
    print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | {'LR':>10} | {'Best':>5}")
    print("-" * 70)

    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    train_losses, val_losses = [], []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        v_loss = validate(model, val_loader, criterion, device)
        train_losses.append(t_loss)
        val_losses.append(v_loss)

        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(v_loss)

        is_best = v_loss < best_val_loss
        if is_best:
            best_val_loss = v_loss
            best_epoch = epoch
            no_improve = 0
            args.output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": v_loss,
                    "train_loss": t_loss,
                    "model_config": {
                        "model_type": "lstm",
                        "input_size": input_size,
                        "hidden_size": args.hidden_size,
                        "num_layers": args.num_layers,
                        "dropout": args.dropout,
                        "bidirectional": False,
                    },
                },
                args.output_dir / "residual_lstm.pt",
            )
        else:
            no_improve += 1

        marker = "*" if is_best else ""
        if epoch % 5 == 0 or epoch == 1 or is_best:
            print(f"{epoch:>6} | {t_loss:>12.6f} | {v_loss:>12.6f} | {lr:>10.6f} | {marker:>5}")

        if no_improve >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("HYBRID TRAINING COMPLETE!")
    print("=" * 70)
    print(f"  Best val loss (residual MSE): {best_val_loss:.6f} (epoch {best_epoch})")
    print(f"  Training time: {elapsed:.1f}s")

    # Save training history
    history = {
        "model_type": "hybrid_prophet_lstm",
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": float(best_val_loss),
        "best_epoch": best_epoch,
        "training_time_seconds": elapsed,
        "hyperparameters": {
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seq_length": args.seq_length,
            "stride": args.stride,
            "epochs_trained": len(train_losses),
            "patience": args.patience,
        },
        "residual_stats": {
            "train_mean": float(train_residuals.mean()),
            "train_std": float(train_residuals.std()),
            "val_mean": float(val_residuals.mean()),
            "val_std": float(val_residuals.std()),
        },
    }
    with open(args.output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"  Artifacts saved to: {args.output_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
