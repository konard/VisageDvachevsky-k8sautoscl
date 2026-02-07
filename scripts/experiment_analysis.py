"""Generate static experiment charts and CSV summaries for documentation.

Runs all 4 scenarios through the simulator, produces:
  - Per-scenario PNG charts (CPU + replicas, replica difference)
  - Combined comparison bar chart (8 KPIs × 4 scenarios)
  - Summary CSV with all metrics
  - LaTeX-ready table for the thesis

Usage:
    python scripts/experiment_analysis.py [--output-dir results/experiments]
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # non-interactive backend

# Add project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.experiment_scenarios import Scenario, get_all_scenarios
from scripts.experiment_simulator import ExperimentMetrics, ExperimentRunner

# ---------------------------------------------------------------------------
# Color palette (shared with GUI and Grafana)
# ---------------------------------------------------------------------------

C = {
    "bg": "#1a1a2e",
    "bg_card": "#16213e",
    "green": "#73BF69",
    "yellow": "#FADE2A",
    "orange": "#FF9830",
    "red": "#F2495C",
    "blue": "#5794F2",
    "purple": "#B877D9",
    "text": "#e0e0e0",
    "text_dim": "#8892b0",
    "grid": "#2a2a4a",
}

# Apply dark style globally
plt.rcParams.update(
    {
        "figure.facecolor": C["bg_card"],
        "axes.facecolor": C["bg"],
        "axes.edgecolor": C["grid"],
        "axes.labelcolor": C["text"],
        "text.color": C["text"],
        "xtick.color": C["text_dim"],
        "ytick.color": C["text_dim"],
        "grid.color": C["grid"],
        "grid.alpha": 0.3,
        "font.size": 10,
        "legend.facecolor": C["bg_card"],
        "legend.edgecolor": C["grid"],
    }
)


# ---------------------------------------------------------------------------
# Per-scenario chart
# ---------------------------------------------------------------------------


def plot_scenario(
    scenario: Scenario,
    history: list,
    metrics: ExperimentMetrics,
    output_path: Path,
) -> None:
    """Generate a 3-subplot chart for a single scenario."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), dpi=120)
    fig.suptitle(
        f"Scenario: {scenario.name}  ({scenario.duration_hours:.1f}h, {scenario.steps} steps)",
        fontsize=14,
        fontweight="bold",
        color=C["text"],
    )

    n = len(history)
    x = np.arange(n)
    actual = np.array([r.actual_cpu for r in history])
    predicted = np.array([r.predicted_cpu for r in history])
    hpa_r = np.array([r.hpa_replicas for r in history])
    ml_r = np.array([r.ml_replicas for r in history])

    # --- Subplot 1: CPU trace ---
    ax1.fill_between(x, actual, alpha=0.12, color=C["text_dim"])
    ax1.plot(x, actual, color=C["green"], lw=1.2, label="Actual CPU", alpha=0.8)
    ax1.plot(x, predicted, color=C["purple"], lw=1.0, ls="--", label="ML Predicted", alpha=0.7)

    # Thresholds
    ax1.axhline(60, color=C["yellow"], ls=":", lw=0.8, alpha=0.6, label="Target (60%)")
    ax1.axhline(66, color=C["red"], ls="--", lw=0.7, alpha=0.5, label="Up thresh (66%)")
    ax1.axhline(42, color=C["orange"], ls="--", lw=0.7, alpha=0.5, label="Down thresh (42%)")
    ax1.axhspan(42, 66, alpha=0.05, color=C["yellow"])

    ax1.set_ylabel("CPU Utilization (%)")
    ax1.set_ylim(0, 105)
    ax1.legend(loc="upper right", fontsize=8, ncol=3)
    ax1.grid(True, alpha=0.2)
    ax1.set_title("CPU Trace + Prediction + Algorithm Thresholds", fontsize=11)

    # --- Subplot 2: Replicas comparison ---
    ax2.step(x, hpa_r, color=C["red"], lw=2, where="post", label="HPA Replicas", alpha=0.8)
    ax2.step(x, ml_r, color=C["blue"], lw=2, where="post", label="ML Replicas", alpha=0.9)
    ax2.axhline(2, color=C["text_dim"], ls=":", lw=0.6, alpha=0.4, label="Min (2)")
    ax2.axhline(8, color=C["text_dim"], ls=":", lw=0.6, alpha=0.4, label="Max (8)")
    ax2.fill_between(x, hpa_r, ml_r, alpha=0.1, color=C["blue"])

    ax2.set_ylabel("Replica Count")
    ax2.set_ylim(0, 10)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.2)
    ax2.set_title("Replica Count: HPA (reactive) vs ML (predictive)", fontsize=11)

    # --- Subplot 3: Difference (ML - HPA) ---
    diff = ml_r - hpa_r
    colors_bar = [C["green"] if d < 0 else C["red"] if d > 0 else C["text_dim"] for d in diff]
    ax3.bar(x, diff, color=colors_bar, alpha=0.7, width=1.0)
    ax3.axhline(0, color=C["text_dim"], lw=0.5)
    ax3.set_ylabel("ML - HPA (replicas)")
    ax3.set_xlabel(f"Step (each = 5 min)")
    ax3.grid(True, alpha=0.2)
    ax3.set_title(
        f"Replica Difference (green = ML saves resources) | Cost savings: {metrics.cost_savings_pct:+.1f}%",
        fontsize=11,
    )

    # Annotation: key metrics
    summary = (
        f"HPA avg: {metrics.hpa_avg_replicas:.1f}R | ML avg: {metrics.ml_avg_replicas:.1f}R | "
        f"HPA SLA viol: {metrics.hpa_sla_violations} | ML SLA viol: {metrics.ml_sla_violations} | "
        f"Cost savings: {metrics.cost_savings_pct:+.1f}%"
    )
    fig.text(0.5, 0.01, summary, ha="center", fontsize=9, color=C["text_dim"])

    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Combined bar chart (8 KPIs × 4 scenarios)
# ---------------------------------------------------------------------------


def plot_combined_bars(
    names: list[str],
    all_metrics: list[ExperimentMetrics],
    output_path: Path,
) -> None:
    """Generate a multi-metric bar chart comparing HPA vs ML across scenarios."""
    metric_pairs = [
        ("Avg Replicas", "hpa_avg_replicas", "ml_avg_replicas"),
        ("SLA Violations", "hpa_sla_violations", "ml_sla_violations"),
        ("Over-Prov %", "hpa_over_provisioning_pct", "ml_over_provisioning_pct"),
        ("Under-Prov %", "hpa_under_provisioning_pct", "ml_under_provisioning_pct"),
        ("Scaling Events", "hpa_scaling_events", "ml_scaling_events"),
        ("Avg Reaction", "hpa_avg_reaction_time", "ml_avg_reaction_time"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=120)
    fig.suptitle(
        "HPA vs ML Autoscaler — Metric Comparison Across Scenarios",
        fontsize=14,
        fontweight="bold",
        color=C["text"],
    )

    x = np.arange(len(names))
    width = 0.35

    for idx, (title, hpa_key, ml_key) in enumerate(metric_pairs):
        ax = axes[idx // 3, idx % 3]

        hpa_vals = [getattr(m, hpa_key) for m in all_metrics]
        ml_vals = [getattr(m, ml_key) for m in all_metrics]

        bars_hpa = ax.bar(x - width / 2, hpa_vals, width, color=C["red"], alpha=0.8, label="HPA")
        bars_ml = ax.bar(x + width / 2, ml_vals, width, color=C["blue"], alpha=0.8, label="ML")

        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.2)

        # Value labels
        for bar in bars_hpa:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h,
                    f"{h:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=C["red"],
                )
        for bar in bars_ml:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h,
                    f"{h:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=C["blue"],
                )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Cost savings chart
# ---------------------------------------------------------------------------


def plot_cost_savings(
    names: list[str],
    all_metrics: list[ExperimentMetrics],
    output_path: Path,
) -> None:
    """Generate a cost savings bar chart."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=120)

    savings = [m.cost_savings_pct for m in all_metrics]
    colors = [C["green"] if s > 0 else C["red"] for s in savings]
    x = np.arange(len(names))

    bars = ax.bar(x, savings, color=colors, alpha=0.85, width=0.5)
    ax.axhline(0, color=C["text_dim"], lw=0.5)

    for bar, val in zip(bars, savings):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (1 if val >= 0 else -2),
            f"{val:+.1f}%",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=12,
            fontweight="bold",
            color=C["green"] if val > 0 else C["red"],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=10)
    ax.set_ylabel("Cost Savings (%)")
    ax.set_title(
        "ML Autoscaler Cost Savings vs HPA (positive = ML cheaper)",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, axis="y", alpha=0.2)

    avg_savings = np.mean(savings)
    ax.axhline(avg_savings, color=C["yellow"], ls="--", lw=1.2, alpha=0.7)
    ax.text(
        len(names) - 0.5,
        avg_savings + 1,
        f"avg: {avg_savings:+.1f}%",
        fontsize=10,
        color=C["yellow"],
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Replica-minutes chart
# ---------------------------------------------------------------------------


def plot_replica_minutes(
    names: list[str],
    all_metrics: list[ExperimentMetrics],
    output_path: Path,
) -> None:
    """Generate a replica-minutes comparison chart."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=120)

    x = np.arange(len(names))
    width = 0.35
    hpa_rm = [m.hpa_replica_minutes for m in all_metrics]
    ml_rm = [m.ml_replica_minutes for m in all_metrics]

    ax.bar(x - width / 2, hpa_rm, width, color=C["red"], alpha=0.8, label="HPA")
    ax.bar(x + width / 2, ml_rm, width, color=C["blue"], alpha=0.8, label="ML")

    # Percentage labels
    for i, (h, m) in enumerate(zip(hpa_rm, ml_rm)):
        pct = (h - m) / h * 100 if h > 0 else 0
        ax.text(
            i,
            max(h, m) + 50,
            f"{pct:+.1f}%",
            ha="center",
            fontsize=10,
            fontweight="bold",
            color=C["green"] if pct > 0 else C["red"],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=10)
    ax.set_ylabel("Replica-Minutes")
    ax.set_title("Total Resource Consumption (Replica-Minutes)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# CSV / LaTeX export
# ---------------------------------------------------------------------------


def export_csv(
    names: list[str],
    all_metrics: list[ExperimentMetrics],
    output_path: Path,
) -> None:
    """Export metrics to CSV."""
    rows = []
    for name, m in zip(names, all_metrics):
        row = {"scenario": name}
        row.update(asdict(m))
        rows.append(row)

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {output_path}")


def export_latex_table(
    names: list[str],
    all_metrics: list[ExperimentMetrics],
    output_path: Path,
) -> None:
    """Generate a LaTeX-ready comparison table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{HPA vs ML Autoscaler — Experiment Results}",
        r"\label{tab:experiment-results}",
        r"\small",
        r"\begin{tabular}{l|rr|rr|rr|r}",
        r"\hline",
        r"\textbf{Scenario} & \multicolumn{2}{c|}{\textbf{Avg Replicas}} "
        r"& \multicolumn{2}{c|}{\textbf{SLA Violations}} "
        r"& \multicolumn{2}{c|}{\textbf{Scaling Events}} "
        r"& \textbf{Cost} \\",
        r"& HPA & ML & HPA & ML & HPA & ML & \textbf{Savings} \\",
        r"\hline",
    ]

    for name, m in zip(names, all_metrics):
        display_name = name.replace("_", " ").title()
        lines.append(
            f"{display_name} & {m.hpa_avg_replicas:.1f} & {m.ml_avg_replicas:.1f} "
            f"& {m.hpa_sla_violations} & {m.ml_sla_violations} "
            f"& {m.hpa_scaling_events} & {m.ml_scaling_events} "
            f"& {m.cost_savings_pct:+.1f}\\% \\\\"
        )

    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "experiments",
        help="Directory for output charts and CSV",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=ROOT / "data" / "processed" / "test.csv",
    )
    parser.add_argument(
        "--scaler",
        type=Path,
        default=ROOT / "data" / "processed" / "scaler.pkl",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=ROOT / "data" / "processed" / "metadata.json",
    )
    args = parser.parse_args(argv)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ML Autoscaler — Experiment Analysis")
    print("=" * 60)

    # Load scenarios
    print("\n[1/5] Loading scenarios...")
    try:
        scenarios = get_all_scenarios(args.test_csv, args.scaler, args.metadata)
        print(f"  Loaded {len(scenarios)} scenarios from real data")
    except Exception as e:
        print(f"  Warning: Could not load real data ({e})")
        print("  Using synthetic fallback scenarios")
        scenarios = _generate_fallback_scenarios()

    # Run experiments
    print("\n[2/5] Running experiments...")
    all_names: list[str] = []
    all_metrics: list[ExperimentMetrics] = []
    all_histories: list[list] = []

    for sc in scenarios:
        runner = ExperimentRunner(sc.actual_cpu, sc.predicted_cpu)
        history, metrics = runner.run()
        all_names.append(sc.name)
        all_metrics.append(metrics)
        all_histories.append(history)
        print(
            f"  {sc.name:20s} | steps={sc.steps:>4} | "
            f"HPA avg={metrics.hpa_avg_replicas:.1f}R | ML avg={metrics.ml_avg_replicas:.1f}R | "
            f"savings={metrics.cost_savings_pct:+.1f}%"
        )

    # Per-scenario charts
    print("\n[3/5] Generating per-scenario charts...")
    for sc, history, metrics in zip(scenarios, all_histories, all_metrics):
        plot_scenario(sc, history, metrics, out / f"scenario_{sc.name}.png")

    # Combined charts
    print("\n[4/5] Generating combined charts...")
    plot_combined_bars(all_names, all_metrics, out / "combined_metrics.png")
    plot_cost_savings(all_names, all_metrics, out / "cost_savings.png")
    plot_replica_minutes(all_names, all_metrics, out / "replica_minutes.png")

    # Export data
    print("\n[5/5] Exporting data...")
    export_csv(all_names, all_metrics, out / "experiment_results.csv")
    export_latex_table(all_names, all_metrics, out / "experiment_table.tex")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    avg_savings = np.mean([m.cost_savings_pct for m in all_metrics])
    avg_hpa_sla = np.mean([m.hpa_sla_violations for m in all_metrics])
    avg_ml_sla = np.mean([m.ml_sla_violations for m in all_metrics])
    avg_hpa_events = np.mean([m.hpa_scaling_events for m in all_metrics])
    avg_ml_events = np.mean([m.ml_scaling_events for m in all_metrics])

    print(f"  Average cost savings:    {avg_savings:+.1f}%")
    print(f"  Average SLA violations:  HPA={avg_hpa_sla:.0f}  ML={avg_ml_sla:.0f}")
    print(f"  Average scaling events:  HPA={avg_hpa_events:.0f}  ML={avg_ml_events:.0f}")
    print(f"\n  Output directory: {out.resolve()}")
    print("  Files generated:")
    for f in sorted(out.iterdir()):
        size = f.stat().st_size
        print(f"    {f.name:40s} {size:>8,} bytes")

    return 0


def _generate_fallback_scenarios() -> list[Scenario]:
    """Generate simple synthetic scenarios when data files are unavailable."""
    rng = np.random.RandomState(42)
    scenarios = []

    # Stable
    n = 300
    actual = 50 + rng.normal(0, 4, n)
    predicted = np.roll(actual, -3) + rng.normal(0, 2, n)
    scenarios.append(
        Scenario("stable_workload", "Stable ~50% CPU", np.clip(actual, 10, 100), np.clip(predicted, 10, 100), n * 5 / 60)
    )

    # Gradual ramp
    n = 400
    t = np.linspace(0, 1, n)
    actual = 30 + 55 * t + rng.normal(0, 3, n)
    predicted = np.roll(actual, -3) + rng.normal(0, 2.5, n)
    scenarios.append(
        Scenario("gradual_ramp", "30%->85% ramp", np.clip(actual, 10, 100), np.clip(predicted, 10, 100), n * 5 / 60)
    )

    # Spike
    n = 350
    actual = np.full(n, 42.0) + rng.normal(0, 3, n)
    for i in range(60, 90):
        actual[i] = 85 + rng.normal(0, 3)
    for i in range(180, 220):
        actual[i] = 90 + rng.normal(0, 3)
    predicted = np.roll(actual, -3) + rng.normal(0, 3, n)
    scenarios.append(
        Scenario("spike_pattern", "Spike bursts", np.clip(actual, 10, 100), np.clip(predicted, 10, 100), n * 5 / 60)
    )

    # Diurnal
    n = 576
    actual = np.zeros(n)
    for i in range(n):
        hour = (i * 5 / 60) % 24
        factor = 0.3 if hour < 6 else (0.3 + 0.7 * (hour - 6) / 3 if hour < 9 else 1.0 if hour < 17 else 0.5)
        actual[i] = 20 + factor * 65 + rng.normal(0, 4)
    predicted = np.roll(actual, -3) + rng.normal(0, 2.5, n)
    scenarios.append(
        Scenario("diurnal_cycle", "48h day/night", np.clip(actual, 10, 100), np.clip(predicted, 10, 100), n * 5 / 60)
    )

    return scenarios


if __name__ == "__main__":
    raise SystemExit(main())
