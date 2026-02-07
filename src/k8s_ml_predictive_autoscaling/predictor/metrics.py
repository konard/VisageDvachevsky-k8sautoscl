"""Prometheus metrics for the predictor service."""

from prometheus_client import Counter, Gauge, Histogram, Summary

PREDICTION_LATENCY = Histogram(
    "predictor_inference_latency_seconds",
    "End-to-end prediction latency including preprocessing.",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

PREDICTION_COUNTER = Counter(
    "predictor_predictions_total",
    "Total predictions generated.",
    labelnames=("status",),
)

INGEST_COUNTER = Counter(
    "predictor_observations_ingested_total",
    "Total metric observations ingested.",
)

BUFFER_SIZE = Gauge(
    "predictor_buffer_size",
    "Current number of observations in the sliding window buffer.",
)

MODEL_LOAD_TIME = Gauge(
    "predictor_model_load_seconds",
    "Time taken to load the ONNX model.",
)

PREDICTION_VALUE = Gauge(
    "predictor_latest_prediction",
    "Most recent predicted value (original scale).",
)

ONNX_INFERENCE_LATENCY = Summary(
    "predictor_onnx_inference_seconds",
    "Pure ONNX Runtime inference latency (no preprocessing).",
)


__all__ = [
    "PREDICTION_LATENCY",
    "PREDICTION_COUNTER",
    "INGEST_COUNTER",
    "BUFFER_SIZE",
    "MODEL_LOAD_TIME",
    "PREDICTION_VALUE",
    "ONNX_INFERENCE_LATENCY",
]
