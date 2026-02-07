#!/usr/bin/env python3
"""Check if raw data has realistic patterns."""
import glob

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid")

# Load all raw request_rate files
files = sorted(glob.glob("data/raw/request_rate_*.csv"))
print(f"Found {len(files)} request_rate files")

dfs = []
for f in files:
    df = pd.read_csv(f, parse_dates=["timestamp"])
    dfs.append(df)

full_df = pd.concat(dfs, ignore_index=True).sort_values("timestamp")
print(f"\nTotal samples: {len(full_df)}")
print(f"Date range: {full_df['timestamp'].min()} to {full_df['timestamp'].max()}")

# Statistics
print("\n" + "=" * 60)
print("RAW REQUEST RATE STATISTICS:")
print("=" * 60)
print(f"Mean: {full_df['value'].mean():.4f} req/s")
print(f"Std: {full_df['value'].std():.4f}")
print(f"Min: {full_df['value'].min():.4f}")
print(f"Max: {full_df['value'].max():.4f}")
print(f"Coefficient of Variation (CV): {(full_df['value'].std() / full_df['value'].mean()):.2%}")

# Check hourly patterns
full_df["hour"] = full_df["timestamp"].dt.hour
full_df["day_of_week"] = full_df["timestamp"].dt.dayofweek

hourly_stats = full_df.groupby("hour")["value"].agg(["mean", "std", "min", "max"])
print("\n" + "=" * 60)
print("HOURLY PATTERNS:")
print("=" * 60)
print(hourly_stats)

daily_stats = full_df.groupby("day_of_week")["value"].agg(["mean", "std"])
print("\n" + "=" * 60)
print("DAILY PATTERNS (0=Mon, 6=Sun):")
print("=" * 60)
print(daily_stats)

# Visualize
fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# Time series
axes[0].plot(full_df["timestamp"], full_df["value"], alpha=0.7, linewidth=0.5)
axes[0].set_title("Raw Request Rate Over Time", fontsize=14, fontweight="bold")
axes[0].set_ylabel("Request Rate (req/s)")
axes[0].grid(True, alpha=0.3)

# Hourly pattern
axes[1].plot(hourly_stats.index, hourly_stats["mean"], marker="o", linewidth=2)
axes[1].fill_between(
    hourly_stats.index,
    hourly_stats["mean"] - hourly_stats["std"],
    hourly_stats["mean"] + hourly_stats["std"],
    alpha=0.3,
)
axes[1].set_title("Hourly Pattern (Mean ± Std)", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Hour of Day")
axes[1].set_ylabel("Request Rate (req/s)")
axes[1].grid(True, alpha=0.3)

# Distribution
axes[2].hist(full_df["value"], bins=100, edgecolor="black", alpha=0.7)
axes[2].axvline(
    full_df["value"].mean(),
    color="red",
    linestyle="--",
    label=f'Mean: {full_df["value"].mean():.3f}',
)
axes[2].axvline(
    full_df["value"].median(),
    color="green",
    linestyle="--",
    label=f'Median: {full_df["value"].median():.3f}',
)
axes[2].set_title("Request Rate Distribution", fontsize=14, fontweight="bold")
axes[2].set_xlabel("Request Rate (req/s)")
axes[2].set_ylabel("Frequency")
axes[2].legend()
axes[2].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("docs/eda_figures/raw_data_analysis.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: docs/eda_figures/raw_data_analysis.png")

# Check for spikes
threshold = full_df["value"].mean() + 2 * full_df["value"].std()
spikes = full_df[full_df["value"] > threshold]
print(
    f"\nSpikes (> mean + 2*std = {threshold:.3f}): {len(spikes)} ({len(spikes)/len(full_df)*100:.2f}%)"
)

# Check variance over time
full_df["hour_block"] = (
    full_df["timestamp"] - full_df["timestamp"].min()
).dt.total_seconds() // 3600
variance_over_time = full_df.groupby("hour_block")["value"].std()
print(f"\nVariance over time (hourly blocks):")
print(f"Mean variance: {variance_over_time.mean():.4f}")
print(f"Max variance: {variance_over_time.max():.4f}")
print(f"Min variance: {variance_over_time.min():.4f}")

print("\n" + "=" * 60)
print("ASSESSMENT:")
print("=" * 60)
cv = full_df["value"].std() / full_df["value"].mean()
if cv < 0.1:
    print("❌ TOO STABLE - CV < 10% (real prod usually 20-50%)")
elif cv < 0.2:
    print("⚠️  LOW VARIANCE - CV < 20% (borderline synthetic)")
else:
    print("✅ REALISTIC VARIANCE - CV >= 20%")

hourly_range = hourly_stats["mean"].max() - hourly_stats["mean"].min()
if hourly_range < 0.1:
    print("❌ NO HOURLY PATTERNS - range < 0.1 req/s")
elif hourly_range < 0.3:
    print("⚠️  WEAK HOURLY PATTERNS - range < 0.3 req/s")
else:
    print("✅ CLEAR HOURLY PATTERNS - range >= 0.3 req/s")
