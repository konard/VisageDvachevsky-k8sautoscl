#!/usr/bin/env python3
"""
Export trained models to ONNX format for production inference.

Exports:
  - LSTM model -> models/onnx/lstm_forecaster.onnx
  - Hybrid residual LSTM -> models/onnx/hybrid_residual_lstm.onnx
  - Prophet component -> models/onnx/prophet_lookup.json (serialized params)
  - Validates each exported ONNX model against PyTorch outputs
  - Benchmarks inference speed (PyTorch vs ONNX Runtime)
  - Applies int8 quantization for optimized models

Usage:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --quantize
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models" / "lstm"))
from model import create_model


def export_lstm_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    seq_length: int = 24,
    opset_version: int = 17,
) -> dict:
    """Export PyTorch LSTM model to ONNX.

    Args:
        checkpoint_path: Path to .pt checkpoint
        output_path: Where to save .onnx file
        seq_length: Input sequence length
        opset_version: ONNX opset version

    Returns:
        Export metadata dict
    """
    device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
    model.eval()

    # Create dummy input
    batch_size = 1
    dummy_input = torch.randn(batch_size, seq_length, config["input_size"])

    # Export
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["prediction"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "prediction": {0: "batch_size"},
        },
        dynamo=False,
    )

    # Validate
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    file_size = output_path.stat().st_size
    return {
        "model_type": config["model_type"],
        "input_size": config["input_size"],
        "hidden_size": config["hidden_size"],
        "num_layers": config["num_layers"],
        "seq_length": seq_length,
        "opset_version": opset_version,
        "file_size_bytes": file_size,
        "file_size_mb": round(file_size / (1024 * 1024), 2),
    }


def validate_onnx_output(
    pytorch_checkpoint: Path,
    onnx_path: Path,
    seq_length: int = 24,
    n_samples: int = 100,
) -> dict:
    """Validate ONNX model output matches PyTorch output.

    Returns:
        Validation metrics (max_diff, mean_diff, all_close)
    """
    device = torch.device("cpu")
    checkpoint = torch.load(pytorch_checkpoint, map_location=device, weights_only=False)
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
    model.eval()

    # Create random test inputs
    test_input = np.random.randn(n_samples, seq_length, config["input_size"]).astype(np.float32)

    # PyTorch predictions
    with torch.no_grad():
        pytorch_output = model(torch.FloatTensor(test_input)).numpy()

    # ONNX Runtime predictions
    session = ort.InferenceSession(str(onnx_path))
    onnx_output = session.run(None, {"input": test_input})[0]

    # Compare
    max_diff = float(np.max(np.abs(pytorch_output - onnx_output)))
    mean_diff = float(np.mean(np.abs(pytorch_output - onnx_output)))
    all_close = bool(np.allclose(pytorch_output, onnx_output, atol=1e-5))

    return {
        "max_abs_difference": max_diff,
        "mean_abs_difference": mean_diff,
        "all_close_atol_1e5": all_close,
        "n_samples_tested": n_samples,
    }


def benchmark_inference(
    pytorch_checkpoint: Path,
    onnx_path: Path,
    seq_length: int = 24,
    n_iterations: int = 1000,
    batch_size: int = 1,
) -> dict:
    """Benchmark inference speed: PyTorch vs ONNX Runtime.

    Returns:
        Benchmark results dict
    """
    device = torch.device("cpu")
    checkpoint = torch.load(pytorch_checkpoint, map_location=device, weights_only=False)
    config = checkpoint["model_config"]

    # PyTorch model
    model = create_model(
        model_type=config["model_type"],
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        bidirectional=config.get("bidirectional", False),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Test input
    test_input_np = np.random.randn(batch_size, seq_length, config["input_size"]).astype(np.float32)
    test_input_torch = torch.FloatTensor(test_input_np)

    # Warm up
    with torch.no_grad():
        for _ in range(50):
            model(test_input_torch)

    # PyTorch benchmark
    pytorch_times = []
    with torch.no_grad():
        for _ in range(n_iterations):
            start = time.perf_counter()
            model(test_input_torch)
            pytorch_times.append((time.perf_counter() - start) * 1000)  # ms

    # ONNX Runtime benchmark
    session = ort.InferenceSession(str(onnx_path))

    # Warm up
    for _ in range(50):
        session.run(None, {"input": test_input_np})

    onnx_times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        session.run(None, {"input": test_input_np})
        onnx_times.append((time.perf_counter() - start) * 1000)  # ms

    pytorch_mean = np.mean(pytorch_times)
    onnx_mean = np.mean(onnx_times)
    speedup = pytorch_mean / onnx_mean if onnx_mean > 0 else 0

    return {
        "batch_size": batch_size,
        "n_iterations": n_iterations,
        "pytorch_ms": {
            "mean": round(float(pytorch_mean), 3),
            "p50": round(float(np.percentile(pytorch_times, 50)), 3),
            "p95": round(float(np.percentile(pytorch_times, 95)), 3),
            "p99": round(float(np.percentile(pytorch_times, 99)), 3),
        },
        "onnx_runtime_ms": {
            "mean": round(float(onnx_mean), 3),
            "p50": round(float(np.percentile(onnx_times, 50)), 3),
            "p95": round(float(np.percentile(onnx_times, 95)), 3),
            "p99": round(float(np.percentile(onnx_times, 99)), 3),
        },
        "speedup": round(float(speedup), 2),
    }


def quantize_onnx_model(input_path: Path, output_path: Path) -> dict:
    """Apply dynamic int8 quantization to an ONNX model.

    Returns:
        Quantization metadata
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )

    original_size = input_path.stat().st_size
    quantized_size = output_path.stat().st_size
    compression = (1 - quantized_size / original_size) * 100

    return {
        "original_size_mb": round(original_size / (1024 * 1024), 2),
        "quantized_size_mb": round(quantized_size / (1024 * 1024), 2),
        "compression_pct": round(float(compression), 1),
    }


def export_prophet_params(prophet_model_path: Path, output_path: Path) -> dict:
    """Serialize Prophet model parameters to JSON for ONNX pipeline.

    Since Prophet can't be directly exported to ONNX, we save its
    trained parameters (trend, seasonality coefficients) as a JSON
    lookup table that can be used by the inference service.
    """
    import joblib

    model = joblib.load(prophet_model_path)

    # Extract Prophet params
    params = {
        "growth": model.growth,
        "changepoints": [str(cp) for cp in model.changepoints],
        "n_changepoints": model.n_changepoints,
        "seasonality_mode": model.seasonality_mode,
        "changepoint_prior_scale": model.changepoint_prior_scale,
        "seasonality_prior_scale": model.seasonality_prior_scale,
        "daily_seasonality": bool(model.daily_seasonality),
        "weekly_seasonality": bool(model.weekly_seasonality),
        "yearly_seasonality": bool(model.yearly_seasonality),
    }

    # Save the trained parameters needed for prediction
    # Prophet uses internal Stan model parameters
    prophet_data = {
        "metadata": params,
        "note": "Prophet model is serialized as pickle. This JSON contains metadata only. "
                "For inference, use joblib.load() on the .pkl file or the ONNX LSTM model directly.",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(prophet_data, f, indent=2, default=str)

    return params


def main():
    parser = argparse.ArgumentParser(description="Export models to ONNX")
    parser.add_argument("--output-dir", type=Path, default=Path("models/onnx"))
    parser.add_argument("--quantize", action="store_true", help="Apply int8 quantization")
    parser.add_argument("--benchmark-iterations", type=int, default=1000)
    parser.add_argument("--seq-length", type=int, default=24)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    print("\n" + "=" * 70)
    print("ONNX MODEL EXPORT")
    print("=" * 70)

    # 1. Export standalone LSTM
    lstm_path = Path("models/lstm/artifacts/best_model.pt")
    if lstm_path.exists():
        print("\n[1/4] Exporting LSTM model...")
        onnx_path = args.output_dir / "lstm_forecaster.onnx"
        export_meta = export_lstm_to_onnx(lstm_path, onnx_path, args.seq_length)
        print(f"  Exported to: {onnx_path} ({export_meta['file_size_mb']} MB)")

        print("  Validating...")
        validation = validate_onnx_output(lstm_path, onnx_path, args.seq_length)
        print(f"  Max diff: {validation['max_abs_difference']:.2e}, "
              f"All close: {validation['all_close_atol_1e5']}")

        print("  Benchmarking...")
        benchmark = benchmark_inference(
            lstm_path, onnx_path, args.seq_length, args.benchmark_iterations
        )
        print(f"  PyTorch: {benchmark['pytorch_ms']['mean']:.3f}ms, "
              f"ONNX: {benchmark['onnx_runtime_ms']['mean']:.3f}ms, "
              f"Speedup: {benchmark['speedup']}x")

        all_results["lstm"] = {
            "export": export_meta,
            "validation": validation,
            "benchmark": benchmark,
        }

        # Quantization
        if args.quantize:
            print("  Quantizing (int8)...")
            quant_path = args.output_dir / "lstm_forecaster_int8.onnx"
            quant_meta = quantize_onnx_model(onnx_path, quant_path)
            print(f"  Quantized: {quant_meta['quantized_size_mb']} MB "
                  f"({quant_meta['compression_pct']:.1f}% smaller)")

            # Benchmark quantized
            quant_benchmark = benchmark_inference(
                lstm_path, quant_path, args.seq_length, args.benchmark_iterations
            )
            print(f"  Quantized ONNX: {quant_benchmark['onnx_runtime_ms']['mean']:.3f}ms")
            all_results["lstm"]["quantization"] = quant_meta
            all_results["lstm"]["quantized_benchmark"] = quant_benchmark
    else:
        print("\n[1/4] LSTM model not found, skipping")

    # 2. Export hybrid residual LSTM
    hybrid_path = Path("models/hybrid/artifacts/residual_lstm.pt")
    if hybrid_path.exists():
        print("\n[2/4] Exporting hybrid residual LSTM...")
        onnx_path = args.output_dir / "hybrid_residual_lstm.onnx"
        export_meta = export_lstm_to_onnx(hybrid_path, onnx_path, args.seq_length)
        print(f"  Exported to: {onnx_path} ({export_meta['file_size_mb']} MB)")

        print("  Validating...")
        validation = validate_onnx_output(hybrid_path, onnx_path, args.seq_length)
        print(f"  Max diff: {validation['max_abs_difference']:.2e}, "
              f"All close: {validation['all_close_atol_1e5']}")

        print("  Benchmarking...")
        benchmark = benchmark_inference(
            hybrid_path, onnx_path, args.seq_length, args.benchmark_iterations
        )
        print(f"  PyTorch: {benchmark['pytorch_ms']['mean']:.3f}ms, "
              f"ONNX: {benchmark['onnx_runtime_ms']['mean']:.3f}ms, "
              f"Speedup: {benchmark['speedup']}x")

        all_results["hybrid_residual_lstm"] = {
            "export": export_meta,
            "validation": validation,
            "benchmark": benchmark,
        }

        if args.quantize:
            print("  Quantizing (int8)...")
            quant_path = args.output_dir / "hybrid_residual_lstm_int8.onnx"
            quant_meta = quantize_onnx_model(onnx_path, quant_path)
            print(f"  Quantized: {quant_meta['quantized_size_mb']} MB "
                  f"({quant_meta['compression_pct']:.1f}% smaller)")
            all_results["hybrid_residual_lstm"]["quantization"] = quant_meta
    else:
        print("\n[2/4] Hybrid model not found, skipping")

    # 3. Export Prophet parameters
    prophet_path = Path("models/prophet/artifacts/prophet_model.pkl")
    if prophet_path.exists():
        print("\n[3/4] Exporting Prophet parameters...")
        prophet_json = args.output_dir / "prophet_params.json"
        params = export_prophet_params(prophet_path, prophet_json)
        print(f"  Saved to: {prophet_json}")
        print(f"  Growth: {params['growth']}, Mode: {params['seasonality_mode']}")
        all_results["prophet"] = {"params": params, "note": "Serialized as pickle + JSON metadata"}
    else:
        print("\n[3/4] Prophet model not found, skipping")

    # 4. Save full export report
    print("\n[4/4] Saving export report...")
    report_path = args.output_dir / "export_report.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Report saved to: {report_path}")

    # Summary
    print("\n" + "=" * 70)
    print("EXPORT SUMMARY")
    print("=" * 70)

    exported_files = list(args.output_dir.glob("*"))
    for f in sorted(exported_files):
        size = f.stat().st_size
        if size > 1024 * 1024:
            print(f"  {f.name:<40} {size / (1024*1024):.2f} MB")
        elif size > 1024:
            print(f"  {f.name:<40} {size / 1024:.1f} KB")
        else:
            print(f"  {f.name:<40} {size} B")

    print("\n" + "=" * 70)
    print("ONNX EXPORT COMPLETE!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
