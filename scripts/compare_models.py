#!/usr/bin/env python3
"""
Compare Prophet and LSTM/GRU model results.
Generates comparison visualizations and summary tables.

Usage:
    python scripts/compare_models.py
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="darkgrid")


def load_results(results_dir: Path) -> dict | None:
    """Load test results JSON from a model's results directory."""
    path = results_dir / "test_results.json"
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping")
        return None
    with open(path) as f:
        return json.load(f)


def load_predictions(results_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load predictions npz if available."""
    path = results_dir / "predictions.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return data["y_true"], data["y_pred"]


def create_comparison_table(results: dict[str, dict]) -> str:
    """Generate a markdown comparison table."""
    lines = [
        "| Metric | " + " | ".join(results.keys()) + " |",
        "| --- | " + " | ".join(["---"] * len(results)) + " |",
    ]

    metrics = ["rmse", "mae", "mape", "r2", "mse"]
    metric_labels = {
        "rmse": "RMSE",
        "mae": "MAE",
        "mape": "MAPE (%)",
        "r2": "R2",
        "mse": "MSE",
    }

    for metric in metrics:
        values = []
        raw_values = []
        for name, res in results.items():
            val = res["test_metrics"].get(metric, float("nan"))
            raw_values.append(val)
            if metric == "mape":
                values.append(f"{val:.2f}%")
            elif metric == "r2":
                values.append(f"{val:.4f}")
            else:
                values.append(f"{val:.6f}")

        # Bold the best value
        if metric == "mape" or metric == "mse" or metric == "rmse" or metric == "mae":
            best_idx = np.nanargmin(raw_values)
        else:  # r2 — higher is better
            best_idx = np.nanargmax(raw_values)

        formatted = []
        for i, v in enumerate(values):
            formatted.append(f"**{v}**" if i == best_idx else v)

        label = metric_labels.get(metric, metric.upper())
        lines.append(f"| {label} | " + " | ".join(formatted) + " |")

    return "\n".join(lines)


def create_comparison_visualizations(
    results: dict[str, dict],
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
):
    """Generate comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Bar chart of metrics
    metrics_to_plot = ["rmse", "mae", "r2"]
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5 * len(metrics_to_plot), 6))

    if len(metrics_to_plot) == 1:
        axes = [axes]

    model_names = list(results.keys())
    colors = sns.color_palette("Set2", n_colors=len(model_names))

    for ax, metric in zip(axes, metrics_to_plot):
        values = [res["test_metrics"].get(metric, 0) for res in results.values()]
        bars = ax.bar(model_names, values, color=colors, edgecolor="black", alpha=0.8)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_title(metric.upper(), fontsize=14, fontweight="bold")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Model Comparison — Key Metrics", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_comparison.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: metrics_comparison.png")
    plt.close()

    # 2. Overlay predictions
    if len(predictions) > 1:
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))

        # Full view
        for name, (y_true, y_pred) in predictions.items():
            idx = np.arange(len(y_true))
            if name == list(predictions.keys())[0]:
                axes[0].plot(idx, y_true, label="Actual", color="black", linewidth=1.5, alpha=0.8)
            axes[0].plot(idx, y_pred, label=f"{name} Predicted", alpha=0.7, linewidth=1)

        axes[0].set_title(
            "All Models — Predictions vs Actual", fontsize=14, fontweight="bold"
        )
        axes[0].set_ylabel("Value (normalized)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Zoomed
        zoom = min(150, min(len(v[0]) for v in predictions.values()))
        for name, (y_true, y_pred) in predictions.items():
            idx = np.arange(zoom)
            if name == list(predictions.keys())[0]:
                axes[1].plot(
                    idx, y_true[:zoom], label="Actual", color="black", linewidth=2, alpha=0.8
                )
            axes[1].plot(idx, y_pred[:zoom], label=f"{name}", alpha=0.7, linewidth=1.5)

        axes[1].set_title(
            f"Zoomed View (First {zoom} points)", fontsize=14, fontweight="bold"
        )
        axes[1].set_xlabel("Sample index")
        axes[1].set_ylabel("Value (normalized)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "predictions_overlay.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: predictions_overlay.png")
        plt.close()

    # 3. Error comparison boxplot
    if len(predictions) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        error_data = []
        labels = []
        for name, (y_true, y_pred) in predictions.items():
            errors = y_true - y_pred
            error_data.append(errors)
            labels.append(name)

        ax.boxplot(error_data, tick_labels=labels, showfliers=True, patch_artist=True)
        ax.axhline(0, color="red", linestyle="--", alpha=0.5)
        ax.set_title("Error Distribution Comparison", fontsize=14, fontweight="bold")
        ax.set_ylabel("Error (Actual - Predicted)")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(output_dir / "error_boxplot.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: error_boxplot.png")
        plt.close()


def generate_markdown_report(
    results: dict[str, dict], output_path: Path, comparison_table: str
):
    """Generate the model comparison markdown document."""
    lines = [
        "# Model Comparison Results",
        "",
        "## Overview",
        "",
        "Comparison of baseline models for Kubernetes workload prediction.",
        "All models are trained on real production data (Alibaba 2018 + Azure v2).",
        "",
        "## Datasets",
        "",
        "| Dataset | Samples | Source |",
        "| --- | --- | --- |",
        "| Alibaba 2018 | 2,243 | Cluster trace data (CPU, memory, network, disk) |",
        "| Azure v2 | 8,640 | VM workload data (CPU, memory) |",
        "| Combined | ~10,800 | Merged with temporal continuity |",
        "",
        "## Target Metric",
        "",
        "- **Target**: `cpu_usage` (CPU utilization percentage)",
        "- **Forecast horizon**: 3 steps ahead (15 minutes)",
        "- **Sequence length**: 24 timesteps (2 hours of 5-min data) for LSTM",
        "",
        "## Results",
        "",
        comparison_table,
        "",
        "## Visualizations",
        "",
        "### Metrics Comparison",
        "![Metrics Comparison](../models/comparison/metrics_comparison.png)",
        "",
        "### Predictions Overlay",
        "![Predictions Overlay](../models/comparison/predictions_overlay.png)",
        "",
        "### Error Distribution",
        "![Error Boxplot](../models/comparison/error_boxplot.png)",
        "",
        "## Analysis",
        "",
    ]

    # Add per-model analysis
    for name, res in results.items():
        m = res["test_metrics"]
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- **R2**: {m['r2']:.4f}")
        lines.append(f"- **RMSE**: {m['rmse']:.6f}")
        lines.append(f"- **MAE**: {m['mae']:.6f}")

        if m["r2"] > 0.9:
            lines.append("- Excellent fit — captures the main patterns well")
        elif m["r2"] > 0.7:
            lines.append("- Good fit — reasonable prediction quality")
        elif m["r2"] > 0.0:
            lines.append("- Moderate fit — captures some patterns but has room for improvement")
        else:
            lines.append("- Poor fit — predictions are worse than mean baseline")
        lines.append("")

    lines.extend(
        [
            "## Recommendations",
            "",
            "Based on the comparison results:",
            "",
            "1. **Best single model** should be selected for Phase 3 hybrid approach",
            "2. For the hybrid model (Phase 3): Prophet captures trend/seasonality, "
            "LSTM captures residual patterns",
            "3. The hybrid approach is expected to improve RMSE by 6-15%",
            "",
            "## Next Steps",
            "",
            "- [ ] Phase 3: Implement hybrid Prophet+LSTM model",
            "- [ ] Phase 3: Export models to ONNX format",
            "- [ ] Phase 4: Build FastAPI inference service",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compare model results")
    parser.add_argument(
        "--prophet-results",
        type=Path,
        default=Path("models/prophet/results"),
    )
    parser.add_argument(
        "--lstm-results",
        type=Path,
        default=Path("models/lstm/results"),
    )
    parser.add_argument(
        "--hybrid-results",
        type=Path,
        default=Path("models/hybrid/results"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/comparison"),
    )
    parser.add_argument(
        "--docs-output",
        type=Path,
        default=Path("docs/model-comparison-results.md"),
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    # Load results
    results = {}
    predictions = {}

    print("\nLoading model results...")

    prophet_res = load_results(args.prophet_results)
    if prophet_res:
        results["Prophet"] = prophet_res
        prophet_preds = load_predictions(args.prophet_results)
        if prophet_preds:
            predictions["Prophet"] = prophet_preds

    lstm_res = load_results(args.lstm_results)
    if lstm_res:
        model_type = lstm_res.get("model_type", "lstm").upper()
        results[model_type] = lstm_res
        lstm_preds = load_predictions(args.lstm_results)
        if lstm_preds:
            predictions[model_type] = lstm_preds

    hybrid_res = load_results(args.hybrid_results)
    if hybrid_res:
        results["Hybrid"] = hybrid_res
        hybrid_preds = load_predictions(args.hybrid_results)
        if hybrid_preds:
            predictions["Hybrid"] = hybrid_preds

    if not results:
        print("ERROR: No model results found. Train models first.")
        return

    # Generate comparison table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    table = create_comparison_table(results)
    print(table)

    # Create visualizations
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)
    create_comparison_visualizations(results, predictions, args.output_dir)

    # Generate markdown report
    print("\n" + "=" * 70)
    print("GENERATING REPORT")
    print("=" * 70)
    generate_markdown_report(results, args.docs_output, table)
    print(f"  Report saved to: {args.docs_output}")

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
