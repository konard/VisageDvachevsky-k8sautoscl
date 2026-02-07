#!/usr/bin/env python3
"""
Exploratory Data Analysis for collected metrics data.
Analyzes patterns, seasonality, statistics, and creates visualizations.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
sns.set_theme(style="darkgrid")
plt.rcParams["figure.figsize"] = (15, 8)

# Load data
print("Loading data...")
train_df = pd.read_csv("data/processed/train.csv", parse_dates=["timestamp"])
val_df = pd.read_csv("data/processed/validation.csv", parse_dates=["timestamp"])
test_df = pd.read_csv("data/processed/test.csv", parse_dates=["timestamp"])

print(f"Train shape: {train_df.shape}")
print(f"Validation shape: {val_df.shape}")
print(f"Test shape: {test_df.shape}")

# Combine for overall statistics
full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
full_df = full_df.sort_values("timestamp").reset_index(drop=True)

print(f"\nFull dataset shape: {full_df.shape}")
print(f"Date range: {full_df['timestamp'].min()} to {full_df['timestamp'].max()}")
print(f"Duration: {full_df['timestamp'].max() - full_df['timestamp'].min()}")

# Basic statistics
print("\n" + "=" * 80)
print("BASIC STATISTICS")
print("=" * 80)

metrics = ["request_rate", "latency_p50", "latency_p95", "latency_p99", "active_jobs"]
stats_summary = full_df[metrics].describe()
print(stats_summary)

# Missing values
print("\n" + "=" * 80)
print("MISSING VALUES")
print("=" * 80)
missing = full_df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values")

# Temporal patterns
print("\n" + "=" * 80)
print("TEMPORAL PATTERNS")
print("=" * 80)

# Group by hour
hourly_stats = full_df.groupby("hour")[metrics].agg(["mean", "std", "min", "max"])
print("\nRequest rate by hour of day:")
print(hourly_stats["request_rate"])

# Group by day of week
daily_stats = full_df.groupby("day_of_week")[metrics].agg(["mean", "std"])
print("\nRequest rate by day of week:")
print(daily_stats["request_rate"])

# Weekend vs weekday
weekend_stats = full_df.groupby("is_weekend")[metrics].mean()
print("\nWeekend vs Weekday comparison:")
print(weekend_stats)

# Correlation analysis
print("\n" + "=" * 80)
print("CORRELATIONS")
print("=" * 80)
correlation_matrix = full_df[metrics].corr()
print(correlation_matrix)

# Create visualizations
print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)

output_dir = Path("docs/eda_figures")
output_dir.mkdir(exist_ok=True, parents=True)

# 1. Time series plot
fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# Plot request rate
axes[0].plot(train_df["timestamp"], train_df["request_rate"], label="Train", alpha=0.7)
axes[0].plot(val_df["timestamp"], val_df["request_rate"], label="Validation", alpha=0.7)
axes[0].plot(test_df["timestamp"], test_df["request_rate"], label="Test", alpha=0.7)
axes[0].set_title("Request Rate Over Time", fontsize=14, fontweight="bold")
axes[0].set_ylabel("Request Rate (req/s)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot latency p95
axes[1].plot(train_df["timestamp"], train_df["latency_p95"], label="Train", alpha=0.7)
axes[1].plot(val_df["timestamp"], val_df["latency_p95"], label="Validation", alpha=0.7)
axes[1].plot(test_df["timestamp"], test_df["latency_p95"], label="Test", alpha=0.7)
axes[1].set_title("Latency P95 Over Time", fontsize=14, fontweight="bold")
axes[1].set_ylabel("Latency (seconds)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot active jobs
axes[2].plot(train_df["timestamp"], train_df["active_jobs"], label="Train", alpha=0.7)
axes[2].plot(val_df["timestamp"], val_df["active_jobs"], label="Validation", alpha=0.7)
axes[2].plot(test_df["timestamp"], test_df["active_jobs"], label="Test", alpha=0.7)
axes[2].set_title("Active Jobs Over Time", fontsize=14, fontweight="bold")
axes[2].set_ylabel("Active Jobs")
axes[2].set_xlabel("Timestamp")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "timeseries.png", dpi=150, bbox_inches="tight")
print(f"Saved: {output_dir / 'timeseries.png'}")
plt.close()

# 2. Hourly patterns
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

hourly_mean = full_df.groupby("hour")[metrics].mean()

axes[0, 0].plot(hourly_mean.index, hourly_mean["request_rate"], marker="o", linewidth=2)
axes[0, 0].set_title("Average Request Rate by Hour", fontweight="bold")
axes[0, 0].set_xlabel("Hour of Day")
axes[0, 0].set_ylabel("Request Rate")
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(
    hourly_mean.index, hourly_mean["latency_p95"], marker="o", linewidth=2, color="orange"
)
axes[0, 1].set_title("Average Latency P95 by Hour", fontweight="bold")
axes[0, 1].set_xlabel("Hour of Day")
axes[0, 1].set_ylabel("Latency (s)")
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(
    hourly_mean.index, hourly_mean["active_jobs"], marker="o", linewidth=2, color="green"
)
axes[1, 0].set_title("Average Active Jobs by Hour", fontweight="bold")
axes[1, 0].set_xlabel("Hour of Day")
axes[1, 0].set_ylabel("Active Jobs")
axes[1, 0].grid(True, alpha=0.3)

# Distribution
axes[1, 1].hist(full_df["request_rate"], bins=50, edgecolor="black", alpha=0.7)
axes[1, 1].set_title("Request Rate Distribution", fontweight="bold")
axes[1, 1].set_xlabel("Request Rate")
axes[1, 1].set_ylabel("Frequency")
axes[1, 1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(output_dir / "hourly_patterns.png", dpi=150, bbox_inches="tight")
print(f"Saved: {output_dir / 'hourly_patterns.png'}")
plt.close()

# 3. Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=1,
    ax=ax,
)
ax.set_title("Correlation Matrix - Key Metrics", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(output_dir / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
print(f"Saved: {output_dir / 'correlation_heatmap.png'}")
plt.close()

# 4. Day of week comparison
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

daily_mean = full_df.groupby("day_of_week")[metrics].mean()
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

axes[0].bar(range(7), daily_mean["request_rate"], color="steelblue", alpha=0.7, edgecolor="black")
axes[0].set_title("Average Request Rate by Day of Week", fontweight="bold")
axes[0].set_xlabel("Day of Week")
axes[0].set_ylabel("Request Rate")
axes[0].set_xticks(range(7))
axes[0].set_xticklabels(days)
axes[0].grid(True, alpha=0.3, axis="y")

axes[1].bar(range(7), daily_mean["latency_p95"], color="coral", alpha=0.7, edgecolor="black")
axes[1].set_title("Average Latency P95 by Day of Week", fontweight="bold")
axes[1].set_xlabel("Day of Week")
axes[1].set_ylabel("Latency (s)")
axes[1].set_xticks(range(7))
axes[1].set_xticklabels(days)
axes[1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(output_dir / "weekly_patterns.png", dpi=150, bbox_inches="tight")
print(f"Saved: {output_dir / 'weekly_patterns.png'}")
plt.close()

# Save summary statistics to JSON
summary = {
    "dataset_info": {
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "total_samples": len(full_df),
        "date_range": {
            "start": str(full_df["timestamp"].min()),
            "end": str(full_df["timestamp"].max()),
            "duration_hours": float(
                (full_df["timestamp"].max() - full_df["timestamp"].min()).total_seconds() / 3600
            ),
        },
    },
    "statistics": {
        metric: {
            "mean": float(full_df[metric].mean()),
            "std": float(full_df[metric].std()),
            "min": float(full_df[metric].min()),
            "max": float(full_df[metric].max()),
            "median": float(full_df[metric].median()),
        }
        for metric in metrics
    },
    "missing_values": int(full_df.isnull().sum().sum()),
    "correlations": correlation_matrix.to_dict(),
}

with open(output_dir / "eda_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved: {output_dir / 'eda_summary.json'}")

print("\n" + "=" * 80)
print("EDA COMPLETE!")
print("=" * 80)
print(f"\nVisualizations saved to: {output_dir}")
print("\nKey findings:")
print(
    f"1. Request rate: mean={summary['statistics']['request_rate']['mean']:.2f}, std={summary['statistics']['request_rate']['std']:.2f}"
)
print(
    f"2. Latency P95: mean={summary['statistics']['latency_p95']['mean']:.4f}s, std={summary['statistics']['latency_p95']['std']:.4f}s"
)
print(
    f"3. Active jobs: mean={summary['statistics']['active_jobs']['mean']:.2f}, max={summary['statistics']['active_jobs']['max']:.0f}"
)
print(f"4. Data duration: {summary['dataset_info']['date_range']['duration_hours']:.1f} hours")
