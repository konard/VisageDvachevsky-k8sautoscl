"""CLI to collect historical metrics from Prometheus."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Protocol

from ..logging import get_logger
from .config import CollectorConfig, MetricConfig, load_config
from .prometheus_client import PrometheusClient

LOGGER = get_logger(__name__)


class PrometheusClientProtocol(Protocol):
    def query_range(
        self,
        query: str,
        *,
        start: datetime,
        end: datetime,
        step: timedelta,
    ) -> list[dict[str, Any]]:
        """Subset of PrometheusClient used by the collector."""


@dataclass(slots=True)
class HistoricalSample:
    timestamp: datetime
    metric: str
    promql: str
    value: float
    labels: dict[str, Any]

    def serialize(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "metric": self.metric,
            "promql": self.promql,
            "value": self.value,
            "labels": json.dumps(self.labels, sort_keys=True),
        }


class HistoricalCollector:
    """Collects Prometheus metrics according to config and stores CSV exports."""

    def __init__(self, config: CollectorConfig, client: "PrometheusClientProtocol") -> None:
        self.config = config
        self.client = client

    def collect(self) -> list[Path]:
        outputs: list[Path] = []
        for metric in self.config.metrics:
            samples = self._collect_metric(metric)
            outputs.extend(self._persist(metric, samples))
        return outputs

    def _collect_metric(self, metric: MetricConfig) -> list[HistoricalSample]:
        lookback = self.config.collection.lookback_delta
        chunk_delta = self.config.collection.chunk_delta
        end_time = datetime.now(tz=UTC)
        start_time = end_time - lookback
        cursor = start_time
        samples: list[HistoricalSample] = []
        step = metric.resolve_step(self.config.collection.default_step)

        while cursor < end_time:
            chunk_end = min(cursor + chunk_delta, end_time)
            result = self.client.query_range(
                metric.promql,
                start=cursor,
                end=chunk_end,
                step=step,
            )
            samples.extend(self._transform_results(metric, result))
            cursor = chunk_end

        LOGGER.info(
            "Collected %s samples for %s",
            len(samples),
            metric.name,
        )
        return samples

    def _persist(self, metric: MetricConfig, samples: list[HistoricalSample]) -> list[Path]:
        if not samples:
            LOGGER.warning("No samples for metric %s", metric.name)
            return []

        output_dir = self.config.collection.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for sample in samples:
            day = sample.timestamp.strftime("%Y%m%d")
            grouped[day].append(sample.serialize())

        written: list[Path] = []
        for day, rows in grouped.items():
            path = output_dir / f"{metric.resolved_prefix()}_{day}.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["timestamp", "metric", "promql", "value", "labels"],
                )
                writer.writeheader()
                writer.writerows(rows)
            LOGGER.info("Wrote %s samples to %s", len(rows), path)
            written.append(path)
        return written

    @staticmethod
    def _transform_results(
        metric: MetricConfig, result: Iterable[dict[str, Any]]
    ) -> list[HistoricalSample]:
        samples: list[HistoricalSample] = []
        for series in result:
            labels = series.get("metric", {})
            for timestamp, value in series.get("values", []):
                ts = datetime.fromtimestamp(float(timestamp), tz=UTC)
                samples.append(
                    HistoricalSample(
                        timestamp=ts,
                        metric=metric.name,
                        promql=metric.promql,
                        value=float(value),
                        labels=labels,
                    )
                )
        return samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to collector YAML config (defaults to package config).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Override Prometheus base URL from config.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.base_url:
        config.prometheus = config.prometheus.model_copy(update={"base_url": args.base_url})

    client = PrometheusClient(
        config.prometheus.base_url,
        timeout_seconds=config.prometheus.timeout_seconds,
        verify_ssl=config.prometheus.verify_ssl,
    )
    try:
        collector = HistoricalCollector(config, client)
        outputs = collector.collect()
        LOGGER.info("Export complete: %s files", len(outputs))
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
