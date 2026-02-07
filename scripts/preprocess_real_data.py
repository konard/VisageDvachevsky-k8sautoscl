#!/usr/bin/env python3
"""
Preprocess real production datasets (Alibaba 2018, Azure v2)
for time series forecasting with Prophet and LSTM models.

Generates:
  - data/processed/train.csv, validation.csv, test.csv
  - data/processed/sequences_{train,validation,test}.npz (for LSTM)
  - data/processed/scaler.pkl
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_alibaba(path: Path) -> pd.DataFrame:
    """Load Alibaba 2018 cluster trace data.

    The dataset has no timestamp column, so we generate a 5-minute
    interval time index typical for cluster monitoring.
    """
    df = pd.read_csv(path)
    # Create a 5-minute interval time index
    start = pd.Timestamp("2018-01-01 00:00:00")
    df["timestamp"] = pd.date_range(start=start, periods=len(df), freq="5min")
    df = df.set_index("timestamp")
    # Rename for consistent naming
    df = df.rename(columns={"cpu_util_percent": "cpu_usage"})
    return df


def load_azure(path: Path) -> pd.DataFrame:
    """Load Azure v2 VM workload data.

    The dataset has no timestamp column, so we generate a 5-minute
    interval time index.
    """
    df = pd.read_csv(path)
    start = pd.Timestamp("2019-01-01 00:00:00")
    df["timestamp"] = pd.date_range(start=start, periods=len(df), freq="5min")
    df = df.set_index("timestamp")
    return df


def merge_and_prepare(alibaba: pd.DataFrame, azure: pd.DataFrame) -> pd.DataFrame:
    """Merge datasets and prepare a unified dataframe.

    Strategy: Use Alibaba's multi-metric richness as the primary dataset.
    We normalize Azure cpu_usage to the same scale as Alibaba cpu_usage
    and concatenate temporally to get more training data.
    """
    # Normalize Azure cpu_usage to percentage (it's in absolute units ~6M)
    azure_norm = azure.copy()
    azure_norm["cpu_usage"] = (
        (azure_norm["cpu_usage"] - azure_norm["cpu_usage"].min())
        / (azure_norm["cpu_usage"].max() - azure_norm["cpu_usage"].min())
        * 100
    )
    azure_norm["assigned_mem"] = (
        (azure_norm["assigned_mem"] - azure_norm["assigned_mem"].min())
        / (azure_norm["assigned_mem"].max() - azure_norm["assigned_mem"].min())
        * 100
    )
    azure_norm = azure_norm.rename(columns={"assigned_mem": "mem_util_percent"})

    # Use Alibaba as our primary dataset (richer features)
    # Add Azure as a secondary dataset for cpu/mem only
    primary = alibaba.copy()

    # Create Azure-based entries with synthetic values for missing columns
    azure_extended = azure_norm.copy()
    azure_extended["net_in"] = azure_extended["cpu_usage"] * 0.4 + np.random.normal(
        0, 2, len(azure_extended)
    )
    azure_extended["net_out"] = azure_extended["cpu_usage"] * 0.3 + np.random.normal(
        0, 1.5, len(azure_extended)
    )
    azure_extended["disk_io_percent"] = azure_extended["cpu_usage"] * 0.15 + np.random.normal(
        0, 1, len(azure_extended)
    )
    azure_extended["disk_io_percent"] = azure_extended["disk_io_percent"].clip(lower=0)

    # Shift Azure timestamps to follow Alibaba
    azure_start = primary.index[-1] + pd.Timedelta(minutes=5)
    azure_extended.index = pd.date_range(
        start=azure_start, periods=len(azure_extended), freq="5min"
    )

    # Concatenate both datasets
    combined = pd.concat([primary, azure_extended], axis=0)
    combined = combined.sort_index()

    return combined


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal features from the datetime index."""
    df = df.copy()
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    df["minute_of_day"] = df.index.hour * 60 + df.index.minute
    return df


def add_lag_features(
    df: pd.DataFrame, columns: list[str], lags: list[int]
) -> pd.DataFrame:
    """Add lag features for specified columns."""
    df = df.copy()
    for col in columns:
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame, columns: list[str], windows: list[int]
) -> pd.DataFrame:
    """Add rolling mean features for specified columns."""
    df = df.copy()
    for col in columns:
        for window in windows:
            df[f"{col}_rolling_mean_{window}"] = df[col].rolling(window=window).mean()
    return df


def build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seq_length: int = 60,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sliding window sequences for LSTM.

    Args:
        df: DataFrame with features and target
        feature_cols: Input feature column names
        target_col: Target column name
        seq_length: Number of timesteps per sequence
        stride: Step between windows

    Returns:
        (sequences, targets, timestamps)
    """
    features = df[feature_cols].values.astype(np.float32)
    targets = df[target_col].values.astype(np.float32)

    seqs, tgts, times = [], [], []
    for i in range(0, len(df) - seq_length, stride):
        seqs.append(features[i : i + seq_length])
        tgts.append(targets[i + seq_length - 1])
        times.append(str(df.index[i + seq_length - 1]))

    return np.array(seqs), np.array(tgts), np.array(times)


def main():
    parser = argparse.ArgumentParser(description="Preprocess real production data")
    parser.add_argument(
        "--alibaba-path",
        type=Path,
        default=Path("data/external/alibaba_2018_real.csv"),
    )
    parser.add_argument(
        "--azure-path",
        type=Path,
        default=Path("data/external/azure_v2_real.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument("--target", default="cpu_usage", help="Target metric to forecast")
    parser.add_argument("--seq-length", type=int, default=24, help="LSTM sequence length")
    parser.add_argument("--stride", type=int, default=1, help="Sliding window stride")
    parser.add_argument("--forecast-horizon", type=int, default=3, help="Steps ahead to predict")
    args = parser.parse_args()

    print("=" * 70)
    print("PREPROCESSING REAL PRODUCTION DATA")
    print("=" * 70)

    # Load datasets
    print("\n[1/7] Loading datasets...")
    alibaba = load_alibaba(args.alibaba_path)
    print(f"  Alibaba: {alibaba.shape} ({alibaba.index[0]} to {alibaba.index[-1]})")

    azure = load_azure(args.azure_path)
    print(f"  Azure:   {azure.shape} ({azure.index[0]} to {azure.index[-1]})")

    # Merge
    print("\n[2/7] Merging datasets...")
    df = merge_and_prepare(alibaba, azure)
    print(f"  Combined: {df.shape} ({df.index[0]} to {df.index[-1]})")

    # Feature engineering
    print("\n[3/7] Engineering features...")
    metrics = ["cpu_usage", "mem_util_percent", "net_in", "net_out", "disk_io_percent"]
    df = add_time_features(df)
    df = add_lag_features(df, metrics, lags=[1, 3, 6, 12])
    df = add_rolling_features(df, metrics, windows=[3, 6, 12])

    # Build targets
    print("\n[4/7] Building forecast targets...")
    for horizon in [1, 3, 6]:
        df[f"target_{args.target}_t+{horizon}"] = df[args.target].shift(-horizon)

    # Drop NaN rows from lag/rolling/target creation
    df = df.dropna()
    print(f"  After dropna: {df.shape}")

    # Normalize
    print("\n[5/7] Normalizing features...")
    scaler = StandardScaler()
    # Only scale numeric feature columns, not time features or targets
    target_cols = [c for c in df.columns if c.startswith("target_")]
    time_feature_cols = ["hour", "day_of_week", "is_weekend", "minute_of_day"]
    feature_cols = [
        c for c in df.columns if c not in target_cols and c not in time_feature_cols
    ]

    # Normalize time features to 0-1 range
    df["hour"] = df["hour"] / 23.0
    df["day_of_week"] = df["day_of_week"] / 6.0
    df["minute_of_day"] = df["minute_of_day"] / 1439.0
    # is_weekend is already 0/1

    # Fit scaler on feature columns
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Also normalize targets using the same scaler's cpu_usage params
    cpu_idx = feature_cols.index(args.target)
    target_mean = scaler.mean_[cpu_idx]
    target_scale = scaler.scale_[cpu_idx]
    for tc in target_cols:
        df[tc] = (df[tc] - target_mean) / target_scale

    # Train/val/test split (70/15/15)
    print("\n[6/7] Splitting data...")
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    print(f"  Train:      {train_df.shape}")
    print(f"  Validation: {val_df.shape}")
    print(f"  Test:       {test_df.shape}")

    # Save
    print("\n[7/7] Saving processed data...")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(args.output_dir / "train.csv", index_label="timestamp")
    val_df.to_csv(args.output_dir / "validation.csv", index_label="timestamp")
    test_df.to_csv(args.output_dir / "test.csv", index_label="timestamp")

    # Save scaler
    joblib.dump(scaler, args.output_dir / "scaler.pkl")

    # Build sequences for LSTM
    all_feature_cols = feature_cols + time_feature_cols
    main_target = f"target_{args.target}_t+{args.forecast_horizon}"

    for name, split_df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        seqs, tgts, times = build_sequences(
            split_df, all_feature_cols, main_target, args.seq_length, args.stride
        )
        path = args.output_dir / f"sequences_{name}.npz"
        np.savez_compressed(
            path,
            sequences=seqs,
            targets=tgts,
            timestamps=times,
            feature_columns=np.array(all_feature_cols),
            target_column=main_target,
        )
        print(f"  {name} sequences: {seqs.shape}")

    # Save metadata
    metadata = {
        "source_datasets": ["alibaba_2018", "azure_v2"],
        "target_metric": args.target,
        "forecast_horizon": args.forecast_horizon,
        "sequence_length": args.seq_length,
        "stride": args.stride,
        "feature_columns": all_feature_cols,
        "target_columns": target_cols,
        "scaler_feature_columns": feature_cols,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
    }
    with open(args.output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nAll outputs saved to {args.output_dir}/")
    print("=" * 70)
    print("PREPROCESSING COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
