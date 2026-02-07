#!/usr/bin/env python3
"""
Evaluate the hybrid Prophet+LSTM model on test data.
Compare with standalone Prophet and LSTM results.

Usage:
    python models/hybrid/evaluate.py
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lstm"))
from model import create_model

sns.set_theme(style="darkgrid")


def generate_prophet_predictions(prophet_model, df, target_col):
    timestamps = pd.to_datetime(df.index)
    if timestamps.tz is not None:
        timestamps = timestamps.tz_localize(None)
    prophet_df = pd.DataFrame({"ds": timestamps, "y": df[target_col].values})
    forecast = prophet_model.predict(prophet_df)
    return forecast["yhat"].values


def build_sequences_with_indices(features, seq_length, stride=1):
    seqs, indices = [], []
    for i in range(0, len(features) - seq_length, stride):
        seqs.append(features[i : i + seq_length])
        indices.append(i + seq_length - 1)
    return np.array(seqs, dtype=np.float32), np.array(indices)


def predict_residuals(model, sequences, device, batch_size=64):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = torch.FloatTensor(sequences[i : i + batch_size]).to(device)
            pred = model(batch)
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds)


def calculate_metrics(y_true, y_pred):
    mse = ((y_true - y_pred) ** 2).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(y_true - y_pred).mean()
    epsilon = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1 - (ss_res / (ss_tot + epsilon))
    return {"rmse": float(rmse), "mae": float(mae), "mape": float(mape), "r2": float(r2), "mse": float(mse)}


def create_visualizations(
    y_true, hybrid_pred, prophet_pred, lstm_residual_pred, output_dir, prophet_only=None, lstm_only=None
):
    output_dir.mkdir(parents=True, exist_ok=True)
    idx = np.arange(len(y_true))

    # 1. Hybrid predictions vs actual
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    axes[0].plot(idx, y_true, label="Actual", color="black", linewidth=1.5, alpha=0.8)
    axes[0].plot(idx, hybrid_pred, label="Hybrid", color="tab:green", linewidth=1, alpha=0.8)
    axes[0].plot(idx, prophet_pred, label="Prophet component", color="tab:orange", linewidth=0.8, alpha=0.5, linestyle="--")
    axes[0].set_title("Hybrid Prophet+LSTM: Predictions vs Actual", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Value (normalized)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    zoom = min(200, len(y_true))
    axes[1].plot(idx[:zoom], y_true[:zoom], label="Actual", color="black", linewidth=2, alpha=0.8)
    axes[1].plot(idx[:zoom], hybrid_pred[:zoom], label="Hybrid", color="tab:green", linewidth=1.5, alpha=0.8)
    axes[1].plot(idx[:zoom], prophet_pred[:zoom], label="Prophet", color="tab:orange", linewidth=1, alpha=0.6, linestyle="--")
    axes[1].set_title(f"Zoomed View (First {zoom} points)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Sample index")
    axes[1].set_ylabel("Value (normalized)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "hybrid_predictions.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: hybrid_predictions.png")
    plt.close()

    # 2. Residual analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    actual_residuals = y_true - prophet_pred
    axes[0].plot(idx, actual_residuals, label="Actual residuals", alpha=0.6, linewidth=0.5)
    axes[0].plot(idx, lstm_residual_pred, label="LSTM predicted", alpha=0.6, linewidth=0.5)
    axes[0].set_title("Residual Predictions", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Sample")
    axes[0].set_ylabel("Residual value")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    hybrid_errors = y_true - hybrid_pred
    axes[1].hist(hybrid_errors, bins=50, edgecolor="black", alpha=0.7)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=2)
    axes[1].set_title("Hybrid Error Distribution", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Error")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "residual_analysis.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: residual_analysis.png")
    plt.close()

    # 3. All models comparison (if standalone results available)
    all_models = {"Hybrid": hybrid_pred}
    if prophet_only is not None:
        all_models["Prophet"] = prophet_only
    if lstm_only is not None:
        all_models["LSTM"] = lstm_only

    if len(all_models) > 1:
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(idx[:zoom], y_true[:zoom], label="Actual", color="black", linewidth=2, alpha=0.9)
        colors = {"Hybrid": "tab:green", "Prophet": "tab:orange", "LSTM": "tab:blue"}
        for name, preds in all_models.items():
            ax.plot(idx[:zoom], preds[:zoom], label=name, color=colors.get(name, "gray"), linewidth=1.2, alpha=0.7)
        ax.set_title("All Models Comparison (Zoomed)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Value (normalized)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "all_models_comparison.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: all_models_comparison.png")
        plt.close()

    # 4. Scatter plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(y_true, hybrid_pred, alpha=0.4, s=10, color="tab:green")
    mn, mx = min(y_true.min(), hybrid_pred.min()), max(y_true.max(), hybrid_pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=2, label="Perfect prediction")
    ax.set_xlabel("Actual", fontsize=12)
    ax.set_ylabel("Predicted", fontsize=12)
    ax.set_title("Hybrid: Predicted vs Actual", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "scatter_plot.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: scatter_plot.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate hybrid model")
    parser.add_argument("--prophet-model", type=Path, default=Path("models/prophet/artifacts/prophet_model.pkl"))
    parser.add_argument("--residual-lstm", type=Path, default=Path("models/hybrid/artifacts/residual_lstm.pt"))
    parser.add_argument("--test-data", type=Path, default=Path("data/processed/test.csv"))
    parser.add_argument("--target", default="cpu_usage")
    parser.add_argument("--output-dir", type=Path, default=Path("models/hybrid/results"))
    parser.add_argument("--seq-length", type=int, default=24)
    parser.add_argument("--lstm-standalone-preds", type=Path, default=Path("models/lstm/results/predictions.npz"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 70)
    print("HYBRID PROPHET+LSTM EVALUATION")
    print("=" * 70)

    # Load metadata
    with open("data/processed/metadata.json") as f:
        metadata = json.load(f)
    feature_cols = metadata["feature_columns"]
    target_col_name = f"target_{args.target}_t+{metadata['forecast_horizon']}"

    # Load test data
    test_df = pd.read_csv(args.test_data, index_col="timestamp", parse_dates=True)
    print(f"Test data: {test_df.shape}")

    # Stage 1: Prophet predictions
    print("\n[Stage 1] Prophet predictions on test set...")
    prophet_model = joblib.load(args.prophet_model)
    prophet_preds_full = generate_prophet_predictions(prophet_model, test_df, args.target)

    # Stage 2: Build sequences and predict residuals
    print("[Stage 2] LSTM residual predictions...")
    checkpoint = torch.load(args.residual_lstm, map_location=device, weights_only=False)
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

    features = test_df[feature_cols].values.astype(np.float32)
    sequences, seq_indices = build_sequences_with_indices(features, args.seq_length)
    print(f"  Sequences: {sequences.shape}")

    lstm_residuals = predict_residuals(lstm_model, sequences, device)

    # Align everything to sequence indices
    prophet_preds_aligned = prophet_preds_full[seq_indices]
    actual_aligned = test_df[target_col_name].values[seq_indices]

    # Combined prediction
    hybrid_preds = prophet_preds_aligned + lstm_residuals

    # Calculate metrics
    print("\n" + "=" * 70)
    print("TEST SET METRICS")
    print("=" * 70)

    hybrid_metrics = calculate_metrics(actual_aligned, hybrid_preds)
    prophet_metrics = calculate_metrics(actual_aligned, prophet_preds_aligned)

    print("\n  Hybrid Prophet+LSTM:")
    for name, val in hybrid_metrics.items():
        if name == "mape":
            print(f"    {name.upper()}: {val:.2f}%")
        else:
            print(f"    {name.upper()}: {val:.6f}")

    print("\n  Prophet only (on same samples):")
    for name, val in prophet_metrics.items():
        if name == "mape":
            print(f"    {name.upper()}: {val:.2f}%")
        else:
            print(f"    {name.upper()}: {val:.6f}")

    # Calculate improvement
    rmse_improvement = (prophet_metrics["rmse"] - hybrid_metrics["rmse"]) / prophet_metrics["rmse"] * 100
    r2_improvement = hybrid_metrics["r2"] - prophet_metrics["r2"]
    print(f"\n  RMSE improvement over Prophet: {rmse_improvement:.1f}%")
    print(f"  R2 improvement over Prophet: +{r2_improvement:.4f}")

    # Visualizations
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)

    # Load LSTM standalone predictions for comparison
    lstm_only_preds = None
    if args.lstm_standalone_preds.exists():
        lstm_data = np.load(args.lstm_standalone_preds)
        lstm_only_preds = lstm_data["y_pred"]
        # Align sizes
        min_len = min(len(actual_aligned), len(lstm_only_preds))
        lstm_only_preds = lstm_only_preds[:min_len]

    min_len = len(actual_aligned)
    if lstm_only_preds is not None:
        min_len = min(min_len, len(lstm_only_preds))

    create_visualizations(
        actual_aligned[:min_len],
        hybrid_preds[:min_len],
        prophet_preds_aligned[:min_len],
        lstm_residuals[:min_len],
        args.output_dir,
        prophet_only=prophet_preds_aligned[:min_len],
        lstm_only=lstm_only_preds[:min_len] if lstm_only_preds is not None else None,
    )

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "model_type": "hybrid_prophet_lstm",
        "test_metrics": hybrid_metrics,
        "prophet_only_metrics": prophet_metrics,
        "rmse_improvement_over_prophet_pct": float(rmse_improvement),
        "r2_improvement_over_prophet": float(r2_improvement),
        "test_samples": len(actual_aligned),
    }

    with open(args.output_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    np.savez_compressed(
        args.output_dir / "predictions.npz",
        y_true=actual_aligned,
        y_pred=hybrid_preds,
        prophet_pred=prophet_preds_aligned,
        lstm_residual_pred=lstm_residuals,
    )

    print(f"\nResults saved to: {args.output_dir}")
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
