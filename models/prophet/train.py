#!/usr/bin/env python3
"""
Train Prophet model for time series forecasting.
Prophet is good for capturing seasonality and trends.
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from prophet import Prophet


def load_data(data_path: Path) -> pd.DataFrame:
    """Load preprocessed training data."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    print(f"Loaded {len(df):,} samples")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df


def prepare_prophet_data(df: pd.DataFrame, target_col: str = "request_rate") -> pd.DataFrame:
    """Convert to Prophet format (ds, y)."""
    # Remove timezone for Prophet compatibility
    timestamps = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

    prophet_df = pd.DataFrame({"ds": timestamps, "y": df[target_col]})
    return prophet_df


def train_prophet_model(
    df: pd.DataFrame,
    seasonality_mode: str = "multiplicative",
    changepoint_prior_scale: float = 0.05,
    seasonality_prior_scale: float = 10.0,
) -> Prophet:
    """Train Prophet model with custom hyperparameters.

    Args:
        df: DataFrame with 'ds' (datetime) and 'y' (target) columns
        seasonality_mode: 'additive' or 'multiplicative'
        changepoint_prior_scale: Flexibility of trend (0.001-0.5)
        seasonality_prior_scale: Strength of seasonality (0.01-10)
    """
    print("\n" + "=" * 70)
    print("TRAINING PROPHET MODEL")
    print("=" * 70)
    print(f"Seasonality mode: {seasonality_mode}")
    print(f"Changepoint prior scale: {changepoint_prior_scale}")
    print(f"Seasonality prior scale: {seasonality_prior_scale}")

    model = Prophet(
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,  # Not enough data for yearly
        interval_width=0.95,
    )

    # Add custom seasonalities if needed
    # model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

    print("\nFitting model...")
    model.fit(df)
    print("[OK] Model trained successfully")

    return model


def evaluate_on_validation(
    model: Prophet, val_path: Path, target_col: str = "request_rate"
) -> dict:
    """Quick evaluation on validation set."""
    print("\n" + "=" * 70)
    print("VALIDATION EVALUATION")
    print("=" * 70)

    val_df = pd.read_csv(val_path, parse_dates=["timestamp"])
    prophet_val = prepare_prophet_data(val_df, target_col)

    # Predict
    forecast = model.predict(prophet_val)

    # Calculate metrics
    y_true = prophet_val["y"].values
    y_pred = forecast["yhat"].values

    mse = ((y_true - y_pred) ** 2).mean()
    rmse = mse**0.5
    mae = abs(y_true - y_pred).mean()
    mape = (abs((y_true - y_pred) / (y_true + 1e-8))).mean() * 100

    metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "samples": len(y_true),
    }

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train Prophet model")
    parser.add_argument(
        "--train-data",
        type=Path,
        default=Path("data/processed/train.csv"),
        help="Path to training data",
    )
    parser.add_argument(
        "--val-data",
        type=Path,
        default=Path("data/processed/validation.csv"),
        help="Path to validation data",
    )
    parser.add_argument(
        "--target",
        default="request_rate",
        help="Target column to forecast",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/prophet/artifacts"),
        help="Directory to save trained model",
    )
    parser.add_argument(
        "--seasonality-mode",
        choices=["additive", "multiplicative"],
        default="multiplicative",
        help="Seasonality mode",
    )
    parser.add_argument(
        "--changepoint-prior",
        type=float,
        default=0.05,
        help="Changepoint prior scale (0.001-0.5)",
    )
    parser.add_argument(
        "--seasonality-prior",
        type=float,
        default=10.0,
        help="Seasonality prior scale (0.01-10)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("PROPHET MODEL TRAINING")
    print("=" * 70)

    # Load data
    train_df = load_data(args.train_data)
    prophet_train = prepare_prophet_data(train_df, args.target)

    # Train model
    model = train_prophet_model(
        prophet_train,
        seasonality_mode=args.seasonality_mode,
        changepoint_prior_scale=args.changepoint_prior,
        seasonality_prior_scale=args.seasonality_prior,
    )

    # Evaluate on validation
    val_metrics = evaluate_on_validation(model, args.val_data, args.target)

    # Save model
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "prophet_model.pkl"

    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)
    joblib.dump(model, model_path)
    print(f"[OK] Model saved to: {model_path}")

    # Save metrics
    metrics_path = args.output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "validation_metrics": val_metrics,
                "hyperparameters": {
                    "seasonality_mode": args.seasonality_mode,
                    "changepoint_prior_scale": args.changepoint_prior,
                    "seasonality_prior_scale": args.seasonality_prior,
                },
                "target_column": args.target,
            },
            f,
            indent=2,
        )
    print(f"[OK] Metrics saved to: {metrics_path}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  1. Evaluate on test: python models/prophet/evaluate.py")
    print(f"  2. View results: cat {metrics_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
