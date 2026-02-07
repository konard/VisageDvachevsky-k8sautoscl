#!/usr/bin/env python3
"""
Train LSTM/GRU model for time series forecasting.

Usage:
    python models/lstm/train.py
    python models/lstm/train.py --model-type gru --hidden-size 128
    python models/lstm/train.py --bidirectional --epochs 200
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import create_model


def load_sequences(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load preprocessed sequences from .npz file."""
    data = np.load(path)
    return data["sequences"], data["targets"]


def create_dataloader(
    sequences: np.ndarray, targets: np.ndarray, batch_size: int, shuffle: bool = True
) -> DataLoader:
    """Create a PyTorch DataLoader from numpy arrays."""
    X = torch.FloatTensor(sequences)
    y = torch.FloatTensor(targets)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Validate model. Returns average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Train LSTM/GRU model")
    parser.add_argument(
        "--train-data",
        type=Path,
        default=Path("data/processed/sequences_train.npz"),
    )
    parser.add_argument(
        "--val-data",
        type=Path,
        default=Path("data/processed/sequences_validation.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/lstm/artifacts"),
    )
    parser.add_argument("--model-type", choices=["lstm", "gru"], default="lstm")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 70)
    print(f"{args.model_type.upper()} MODEL TRAINING")
    print("=" * 70)
    print(f"Device: {device}")

    # Load data
    print("\nLoading data...")
    train_seqs, train_targets = load_sequences(args.train_data)
    val_seqs, val_targets = load_sequences(args.val_data)

    print(f"  Train sequences: {train_seqs.shape}")
    print(f"  Val sequences:   {val_seqs.shape}")
    print(f"  Features per step: {train_seqs.shape[2]}")
    print(f"  Sequence length:   {train_seqs.shape[1]}")
    print(f"  Target range: [{train_targets.min():.4f}, {train_targets.max():.4f}]")

    # Create dataloaders
    train_loader = create_dataloader(train_seqs, train_targets, args.batch_size, shuffle=True)
    val_loader = create_dataloader(val_seqs, val_targets, args.batch_size, shuffle=False)

    # Create model
    input_size = train_seqs.shape[2]
    model = create_model(
        model_type=args.model_type,
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {args.model_type.upper()}")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  Layers: {args.num_layers}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Bidirectional: {args.bidirectional}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    # Training loop with early stopping
    print("\n" + "-" * 70)
    print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | {'LR':>10} | {'Best':>5}")
    print("-" * 70)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            # Save best model
            args.output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                    "model_config": {
                        "model_type": args.model_type,
                        "input_size": input_size,
                        "hidden_size": args.hidden_size,
                        "num_layers": args.num_layers,
                        "dropout": args.dropout,
                        "bidirectional": args.bidirectional,
                    },
                },
                args.output_dir / "best_model.pt",
            )
        else:
            epochs_without_improvement += 1

        marker = "*" if is_best else ""
        if epoch % 5 == 0 or epoch == 1 or is_best:
            print(
                f"{epoch:>6} | {train_loss:>12.6f} | {val_loss:>12.6f} | "
                f"{current_lr:>10.6f} | {marker:>5}"
            )

        # Early stopping
        if epochs_without_improvement >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"  Best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")
    print(f"  Training time: {elapsed:.1f}s")
    print(f"  Model saved to: {args.output_dir / 'best_model.pt'}")

    # Save training history
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": float(best_val_loss),
        "best_epoch": best_epoch,
        "training_time_seconds": elapsed,
        "hyperparameters": {
            "model_type": args.model_type,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "bidirectional": args.bidirectional,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "epochs_trained": len(train_losses),
            "patience": args.patience,
        },
    }
    with open(args.output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"  History saved to: {args.output_dir / 'training_history.json'}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
