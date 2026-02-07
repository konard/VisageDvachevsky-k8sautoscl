#!/usr/bin/env python3
"""
Evaluate trained LSTM/GRU model on test set.
Generate predictions, metrics, and visualizations.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import create_model

sns.set_theme(style="darkgrid")


def load_model(model_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]

    model = create_model(
        model_type=config["model_type"],
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        bidirectional=config.get("bidirectional", False),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded {config['model_type'].upper()} model from epoch {checkpoint['epoch']}")
    print(f"  Val loss at save: {checkpoint['val_loss']:.6f}")
    return model, config


def load_test_data(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load test sequences."""
    data = np.load(path)
    return data["sequences"], data["targets"], data["timestamps"]


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate forecasting metrics."""
    mse = ((y_true - y_pred) ** 2).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(y_true - y_pred).mean()

    # MAPE with epsilon
    epsilon = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100

    # R-squared
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1 - (ss_res / (ss_tot + epsilon))

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "r2": float(r2),
        "mse": float(mse),
    }


def predict(
    model: torch.nn.Module, sequences: np.ndarray, device: torch.device, batch_size: int = 64
) -> np.ndarray:
    """Generate predictions for all sequences."""
    model.eval()
    predictions = []

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = torch.FloatTensor(sequences[i : i + batch_size]).to(device)
            pred = model(batch)
            predictions.append(pred.cpu().numpy())

    return np.concatenate(predictions)


def create_visualizations(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: np.ndarray,
    output_dir: Path,
    model_type: str,
):
    """Create evaluation visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    title_prefix = model_type.upper()

    # 1. Predictions vs Actual
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    idx = np.arange(len(y_true))
    axes[0].plot(idx, y_true, label="Actual", alpha=0.7, linewidth=1)
    axes[0].plot(idx, y_pred, label="Predicted", alpha=0.7, linewidth=1)
    axes[0].set_title(
        f"{title_prefix} Predictions vs Actual (Full Test Set)",
        fontsize=14,
        fontweight="bold",
    )
    axes[0].set_ylabel("Value (normalized)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Zoomed view
    zoom = min(200, len(y_true))
    axes[1].plot(idx[:zoom], y_true[:zoom], label="Actual", alpha=0.7, linewidth=1.5)
    axes[1].plot(idx[:zoom], y_pred[:zoom], label="Predicted", alpha=0.7, linewidth=1.5)
    axes[1].set_title(
        f"{title_prefix} Predictions vs Actual (First {zoom} points)",
        fontsize=14,
        fontweight="bold",
    )
    axes[1].set_xlabel("Sample index")
    axes[1].set_ylabel("Value (normalized)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "predictions_vs_actual.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: predictions_vs_actual.png")
    plt.close()

    # 2. Error distribution
    errors = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].hist(errors, bins=50, edgecolor="black", alpha=0.7)
    axes[0].axvline(0, color="red", linestyle="--", linewidth=2, label="Zero error")
    axes[0].set_title("Prediction Error Distribution", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Error (Actual - Predicted)")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    # Q-Q plot
    from scipy import stats

    stats.probplot(errors, dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q Plot (Error Normality)", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / "error_analysis.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: error_analysis.png")
    plt.close()

    # 3. Scatter plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(y_true, y_pred, alpha=0.4, s=10)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect prediction")

    ax.set_xlabel("Actual", fontsize=12)
    ax.set_ylabel("Predicted", fontsize=12)
    ax.set_title(f"{title_prefix} Predicted vs Actual", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "scatter_plot.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: scatter_plot.png")
    plt.close()

    # 4. Training history (if available)
    history_path = output_dir.parent / "artifacts" / "training_history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

        fig, ax = plt.subplots(figsize=(12, 5))
        epochs = range(1, len(history["train_losses"]) + 1)
        ax.plot(epochs, history["train_losses"], label="Train Loss", alpha=0.8)
        ax.plot(epochs, history["val_losses"], label="Val Loss", alpha=0.8)
        ax.axvline(
            history["best_epoch"], color="green", linestyle="--", alpha=0.5, label="Best epoch"
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE)")
        ax.set_title(f"{title_prefix} Training History", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "training_history.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: training_history.png")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate LSTM/GRU model")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/lstm/artifacts/best_model.pt"),
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=Path("data/processed/sequences_test.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/lstm/results"),
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 70)
    print("LSTM/GRU MODEL EVALUATION")
    print("=" * 70)

    # Load model and data
    model, config = load_model(args.model_path, device)
    sequences, targets, timestamps = load_test_data(args.test_data)
    print(f"\nTest data: {sequences.shape[0]} sequences")

    # Generate predictions
    print("\nGenerating predictions...")
    predictions = predict(model, sequences, device)

    # Calculate metrics
    print("\n" + "=" * 70)
    print("TEST SET METRICS")
    print("=" * 70)
    metrics = calculate_metrics(targets, predictions)

    for name, value in metrics.items():
        if name == "mape":
            print(f"  {name.upper()}: {value:.2f}%")
        else:
            print(f"  {name.upper()}: {value:.6f}")

    # Create visualizations
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)
    create_visualizations(targets, predictions, timestamps, args.output_dir, config["model_type"])

    # Save results
    results = {
        "model_type": config["model_type"],
        "model_config": config,
        "test_metrics": metrics,
        "test_samples": len(targets),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Save predictions for comparison
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        y_true=targets,
        y_pred=predictions,
        timestamps=timestamps,
    )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
