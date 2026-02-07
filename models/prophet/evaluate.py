#!/usr/bin/env python3
"""
Evaluate trained Prophet model on test set.
Generate predictions and calculate metrics.
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid")


def load_model(model_path: Path):
    """Load trained Prophet model."""
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    print("[OK] Model loaded")
    return model


def load_test_data(data_path: Path, target_col: str) -> pd.DataFrame:
    """Load test data."""
    print(f"Loading test data from {data_path}...")
    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    print(f"Loaded {len(df):,} samples")

    # Remove timezone for Prophet compatibility
    timestamps = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

    # Prepare Prophet format
    prophet_df = pd.DataFrame({"ds": timestamps, "y": df[target_col]})
    return prophet_df


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate forecasting metrics."""
    mse = ((y_true - y_pred) ** 2).mean()
    rmse = mse**0.5
    mae = abs(y_true - y_pred).mean()

    # MAPE with epsilon to avoid division by zero
    mape = (abs((y_true - y_pred) / (y_true + 1e-8))).mean() * 100

    # R-squared
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "r2": float(r2),
        "mse": float(mse),
    }


def create_visualizations(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.Series,
    output_dir: Path,
):
    """Create evaluation visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Predictions vs Actual
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # Full time series
    axes[0].plot(timestamps, y_true, label="Actual", alpha=0.7, linewidth=1)
    axes[0].plot(timestamps, y_pred, label="Predicted", alpha=0.7, linewidth=1)
    axes[0].set_title(
        "Prophet Predictions vs Actual (Full Test Set)", fontsize=14, fontweight="bold"
    )
    axes[0].set_ylabel("Value (normalized)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Zoomed view (first 500 points)
    zoom_size = min(500, len(y_true))
    axes[1].plot(
        timestamps[:zoom_size], y_true[:zoom_size], label="Actual", alpha=0.7, linewidth=1.5
    )
    axes[1].plot(
        timestamps[:zoom_size], y_pred[:zoom_size], label="Predicted", alpha=0.7, linewidth=1.5
    )
    axes[1].set_title(
        f"Prophet Predictions vs Actual (First {zoom_size} points)", fontsize=14, fontweight="bold"
    )
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Value (normalized)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "predictions_vs_actual.png", dpi=150, bbox_inches="tight")
    print(f"[OK] Saved: {output_dir / 'predictions_vs_actual.png'}")
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
    axes[0].grid(True, alpha=0.3, axis="y")

    # Q-Q plot for normality check
    from scipy import stats

    stats.probplot(errors, dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q Plot (Error Normality Check)", fontsize=14, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "error_analysis.png", dpi=150, bbox_inches="tight")
    print(f"[OK] Saved: {output_dir / 'error_analysis.png'}")
    plt.close()

    # 3. Scatter plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(y_true, y_pred, alpha=0.3, s=1)

    # Perfect prediction line
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect prediction")

    ax.set_xlabel("Actual", fontsize=12)
    ax.set_ylabel("Predicted", fontsize=12)
    ax.set_title("Predicted vs Actual Scatter Plot", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "scatter_plot.png", dpi=150, bbox_inches="tight")
    print(f"[OK] Saved: {output_dir / 'scatter_plot.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Prophet model")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/prophet/artifacts/prophet_model.pkl"),
        help="Path to trained model",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=Path("data/processed/test.csv"),
        help="Path to test data",
    )
    parser.add_argument(
        "--target",
        default="request_rate",
        help="Target column",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/prophet/results"),
        help="Directory to save results",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("PROPHET MODEL EVALUATION")
    print("=" * 70)

    # Load model and data
    model = load_model(args.model_path)
    test_df = load_test_data(args.test_data, args.target)

    # Make predictions
    print("\nGenerating predictions...")
    forecast = model.predict(test_df)

    y_true = test_df["y"].values
    y_pred = forecast["yhat"].values
    timestamps = test_df["ds"]

    print(f"[OK] Generated {len(y_pred):,} predictions")

    # Calculate metrics
    print("\n" + "=" * 70)
    print("TEST SET METRICS")
    print("=" * 70)
    metrics = calculate_metrics(y_true, y_pred)

    for metric_name, value in metrics.items():
        if metric_name == "mape":
            print(f"{metric_name.upper()}: {value:.2f}%")
        else:
            print(f"{metric_name.upper()}: {value:.4f}")

    # Create visualizations
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)
    create_visualizations(y_true, y_pred, timestamps, args.output_dir)

    # Save results
    results = {
        "model_type": "prophet",
        "test_metrics": metrics,
        "test_samples": len(y_true),
        "target_column": args.target,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results saved to: {results_path}")

    # Save predictions for model comparison
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        y_true=y_true,
        y_pred=y_pred,
        timestamps=timestamps.values.astype(str),
    )
    print(f"[OK] Predictions saved to: {args.output_dir / 'predictions.npz'}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)
    print(f"\nResults directory: {args.output_dir}")
    print(f"  - test_results.json")
    print(f"  - predictions_vs_actual.png")
    print(f"  - error_analysis.png")
    print(f"  - scatter_plot.png")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
