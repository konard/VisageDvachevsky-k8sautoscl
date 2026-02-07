#!/usr/bin/env python3
"""
Convert public datasets (Alibaba, Azure) to project format.
Synthesize realistic HTTP request rates and latency metrics from CPU/memory data.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def load_alibaba_data(file_path: Path, sample_containers: int = 100) -> pd.DataFrame:
    """Load and preprocess Alibaba cluster trace data.

    Expected format (container_usage.csv):
    - container_id: unique ID
    - timestamp: unix timestamp
    - cpu_util: CPU utilization (0-100)
    - mem_util: Memory utilization (0-100)
    - disk_io_percent: Disk I/O percentage
    - net_in: Network in (KB/s)
    - net_out: Network out (KB/s)
    """
    print(f"Loading Alibaba data from {file_path}...")

    # Read CSV
    df = pd.read_csv(file_path)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")

    # Sample containers if too many
    if "container_id" in df.columns:
        unique_containers = df["container_id"].nunique()
        if unique_containers > sample_containers:
            print(f"  Sampling {sample_containers} out of {unique_containers} containers...")
            selected = df["container_id"].unique()[:sample_containers]
            df = df[df["container_id"].isin(selected)]

    # Convert timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    print(f"  Final shape: {df.shape}")
    return df


def load_azure_data(file_path: Path, sample_vms: int = 100) -> pd.DataFrame:
    """Load and preprocess Azure VM traces.

    Expected format:
    - vmId: VM identifier
    - timestamp: datetime
    - avg_cpu: Average CPU utilization (0-100)
    - max_cpu: Max CPU utilization
    - min_cpu: Min CPU utilization
    """
    print(f"Loading Azure data from {file_path}...")

    df = pd.read_csv(file_path)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")

    # Sample VMs if needed
    if "vmId" in df.columns:
        unique_vms = df["vmId"].nunique()
        if unique_vms > sample_vms:
            print(f"  Sampling {sample_vms} out of {unique_vms} VMs...")
            selected = df["vmId"].unique()[:sample_vms]
            df = df[df["vmId"].isin(selected)]

    # Parse timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    print(f"  Final shape: {df.shape}")
    return df


def synthesize_request_rate(
    cpu_util: pd.Series, base_rps: float = 100.0, variance: float = 0.3
) -> pd.Series:
    """Synthesize HTTP request rate from CPU utilization.

    Assumption: Higher CPU = more requests being processed.
    Uses non-linear relationship with realistic noise.

    Args:
        cpu_util: CPU utilization (0-100)
        base_rps: Base requests per second at 50% CPU
        variance: Random variance factor (0-1)

    Returns:
        Synthesized request rate (req/s)
    """
    # Non-linear relationship: RPS ~ CPU^1.5 (more realistic than linear)
    normalized_cpu = cpu_util / 100.0
    base_rps_value = base_rps * (normalized_cpu**1.5)

    # Add realistic noise (log-normal distribution)
    noise = np.random.lognormal(mean=0, sigma=variance, size=len(cpu_util))
    request_rate = base_rps_value * noise

    # Add occasional spikes (5% chance)
    spike_mask = np.random.random(len(cpu_util)) < 0.05
    request_rate[spike_mask] *= np.random.uniform(1.5, 3.0, spike_mask.sum())

    # Ensure non-negative
    request_rate = np.maximum(request_rate, 0.01)

    return pd.Series(request_rate, index=cpu_util.index)


def synthesize_latency(request_rate: pd.Series, cpu_util: pd.Series) -> dict[str, pd.Series]:
    """Synthesize latency percentiles from request rate and CPU.

    Assumptions:
    - Higher load = higher latency (queueing theory)
    - CPU saturation increases tail latency

    Returns:
        Dictionary with p50, p95, p99 latency (seconds)
    """
    # Base latency: 50-200ms depending on load
    normalized_rr = (request_rate - request_rate.min()) / (
        request_rate.max() - request_rate.min() + 1e-6
    )
    normalized_cpu = cpu_util / 100.0

    # p50: relatively stable, influenced by load
    base_p50 = 0.05 + 0.15 * normalized_rr
    noise_p50 = np.random.normal(1.0, 0.1, len(request_rate))
    latency_p50 = base_p50 * noise_p50

    # p95: more variance, sensitive to CPU
    base_p95 = 0.15 + 0.35 * normalized_rr + 0.2 * normalized_cpu
    noise_p95 = np.random.lognormal(0, 0.2, len(request_rate))
    latency_p95 = base_p95 * noise_p95

    # p99: high variance, tail latency
    base_p99 = 0.25 + 0.5 * normalized_rr + 0.4 * normalized_cpu
    noise_p99 = np.random.lognormal(0, 0.3, len(request_rate))
    latency_p99 = base_p99 * noise_p99

    # Ensure ordering: p50 < p95 < p99
    latency_p50 = np.maximum(latency_p50, 0.01)
    latency_p95 = np.maximum(latency_p95, latency_p50 * 1.1)
    latency_p99 = np.maximum(latency_p99, latency_p95 * 1.2)

    return {
        "latency_p50": pd.Series(latency_p50, index=request_rate.index),
        "latency_p95": pd.Series(latency_p95, index=request_rate.index),
        "latency_p99": pd.Series(latency_p99, index=request_rate.index),
    }


def convert_to_project_format(
    df: pd.DataFrame,
    source: str,
    cpu_col: str = "cpu_util",
    mem_col: str = "mem_util",
) -> pd.DataFrame:
    """Convert dataset to project format with synthesized metrics."""

    print(f"\nConverting {source} data to project format...")

    # Resample to 1-minute intervals (aggregate multiple containers/VMs)
    df = df.set_index("timestamp")

    # Aggregate metrics (mean across all containers/VMs per minute)
    agg_df = (
        df.resample("1min")
        .agg(
            {
                cpu_col: "mean",
                mem_col: "mean",
            }
        )
        .dropna()
    )

    print(f"  After resampling: {len(agg_df)} samples")

    # Synthesize request rate from CPU
    print("  Synthesizing request_rate from CPU...")
    agg_df["request_rate"] = synthesize_request_rate(
        agg_df[cpu_col],
        base_rps=50.0,  # Moderate base load
        variance=0.4,  # 40% variance
    )

    # Synthesize latency from request rate and CPU
    print("  Synthesizing latency percentiles...")
    latency = synthesize_latency(agg_df["request_rate"], agg_df[cpu_col])
    agg_df["latency_p50"] = latency["latency_p50"]
    agg_df["latency_p95"] = latency["latency_p95"]
    agg_df["latency_p99"] = latency["latency_p99"]

    # Active jobs: synthesize from memory + CPU
    # Assumption: higher mem+cpu = more active jobs
    agg_df["active_jobs"] = np.round(
        (agg_df[cpu_col] / 100.0 * 50) + np.random.randint(-5, 5, len(agg_df))
    ).clip(lower=0)

    # Rename columns to project format
    result = pd.DataFrame(
        {
            "timestamp": agg_df.index,
            "request_rate": agg_df["request_rate"],
            "latency_p50": agg_df["latency_p50"],
            "latency_p95": agg_df["latency_p95"],
            "latency_p99": agg_df["latency_p99"],
            "active_jobs": agg_df["active_jobs"],
            "cpu_util": agg_df[cpu_col],
            "mem_util": agg_df[mem_col],
        }
    )

    # Statistics
    print("\n  Converted data statistics:")
    print(f"    Date range: {result['timestamp'].min()} to {result['timestamp'].max()}")
    print(f"    Duration: {result['timestamp'].max() - result['timestamp'].min()}")
    print(f"    Samples: {len(result)}")
    print(
        f"\n    request_rate: mean={result['request_rate'].mean():.2f}, std={result['request_rate'].std():.2f}, CV={result['request_rate'].std()/result['request_rate'].mean():.2%}"
    )
    print(
        f"    latency_p95: mean={result['latency_p95'].mean():.3f}s, std={result['latency_p95'].std():.3f}s"
    )
    print(
        f"    active_jobs: mean={result['active_jobs'].mean():.1f}, max={result['active_jobs'].max():.0f}"
    )

    return result


def save_in_prometheus_format(df: pd.DataFrame, output_dir: Path, metric_name: str):
    """Save data in Prometheus-like CSV format (compatible with collector)."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by date for daily files
    df["date"] = df["timestamp"].dt.date

    for metric in ["request_rate", "latency_p50", "latency_p95", "latency_p99", "active_jobs"]:
        for date, group in df.groupby("date"):
            output_file = output_dir / f"{metric}_{date.strftime('%Y%m%d')}.csv"

            # Prometheus format
            prometheus_df = pd.DataFrame(
                {
                    "timestamp": group["timestamp"],
                    "metric": metric,
                    "promql": f'rate(demo_service_requests_total{{job="demo-services"}}[1m])',
                    "value": group[metric],
                    "labels": '{"endpoint": "/workload", "job": "demo-services"}',
                }
            )

            prometheus_df.to_csv(output_file, index=False)
            print(f"  Saved: {output_file} ({len(prometheus_df)} samples)")


def main():
    parser = argparse.ArgumentParser(description="Convert public datasets to project format")
    parser.add_argument(
        "--source",
        choices=["alibaba", "azure", "both"],
        default="both",
        help="Dataset source to convert",
    )
    parser.add_argument(
        "--alibaba-file",
        type=Path,
        default=Path("data/external/alibaba_2018_sample.csv"),
        help="Alibaba dataset file",
    )
    parser.add_argument(
        "--azure-file",
        type=Path,
        default=Path("data/external/azure_functions_2019.csv"),
        help="Azure dataset file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw_realistic"),
        help="Output directory for converted data",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of containers/VMs to sample",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("PUBLIC DATASET CONVERTER")
    print("=" * 70)

    datasets = []

    # Load and convert Alibaba
    if args.source in ["alibaba", "both"] and args.alibaba_file.exists():
        try:
            alibaba_df = load_alibaba_data(args.alibaba_file, args.sample_size)
            # Detect column names (different formats)
            cpu_col = next((c for c in alibaba_df.columns if "cpu" in c.lower()), "cpu_util")
            mem_col = next((c for c in alibaba_df.columns if "mem" in c.lower()), "mem_util")

            converted = convert_to_project_format(alibaba_df, "Alibaba", cpu_col, mem_col)
            datasets.append(("alibaba", converted))
        except Exception as e:
            print(f"✗ Failed to convert Alibaba data: {e}")

    # Load and convert Azure
    if args.source in ["azure", "both"] and args.azure_file.exists():
        try:
            azure_df = load_azure_data(args.azure_file, args.sample_size)
            cpu_col = next((c for c in azure_df.columns if "cpu" in c.lower()), "avg_cpu")
            mem_col = "mem_util" if "mem_util" in azure_df.columns else cpu_col  # Fallback

            converted = convert_to_project_format(azure_df, "Azure", cpu_col, mem_col)
            datasets.append(("azure", converted))
        except Exception as e:
            print(f"✗ Failed to convert Azure data: {e}")

    if not datasets:
        print("\n✗ No datasets were successfully converted!")
        print("  Make sure to download datasets first: python scripts/download_datasets.py")
        sys.exit(1)

    # Combine datasets if both available
    if len(datasets) == 2:
        print("\n" + "=" * 70)
        print("COMBINING DATASETS")
        print("=" * 70)
        combined = pd.concat([df for _, df in datasets], ignore_index=True)
        combined = combined.sort_values("timestamp").reset_index(drop=True)
        print(f"  Combined samples: {len(combined)}")
        print(f"  Date range: {combined['timestamp'].min()} to {combined['timestamp'].max()}")

        # Save combined
        save_in_prometheus_format(combined, args.output_dir, "combined")
    else:
        # Save individual dataset
        source_name, df = datasets[0]
        save_in_prometheus_format(df, args.output_dir, source_name)

    print("\n" + "=" * 70)
    print("CONVERSION COMPLETE!")
    print("=" * 70)
    print(f"\n✓ Data saved to: {args.output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Check data: ls -lh {args.output_dir}/")
    print(f"  2. Run preprocessing: python -m k8s_ml_predictive_autoscaling.preprocessor.pipeline")
    print(f"  3. Train models: python -m k8s_ml_predictive_autoscaling.predictor.train")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
