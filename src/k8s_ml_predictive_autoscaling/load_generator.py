"""Async synthetic load generator for demo services."""

from __future__ import annotations

import argparse
import asyncio
import itertools
import os
import signal
from collections.abc import AsyncIterator

import httpx

from .logging import get_logger, log_structured
from .synthetic import PatternConfig, generate_profile

LOGGER = get_logger(__name__)
DEFAULT_API_KEY_HEADER = "X-API-Key"


async def payload_stream(profile: list[float]) -> AsyncIterator[dict[str, float]]:
    """Yield payloads based on a deterministic profile.

    Args:
        profile: Normalized load profile values between 0 and 1.

    Yields:
        Request payloads with payload_size / cpu_hint fields.
    """

    for value in itertools.cycle(profile):
        yield {
            "payload_size": 64,
            "cpu_hint": 0.02 + value * 0.05,
        }


async def hit_targets(
    targets: list[str],
    interval: float,
    profile: list[float],
    stop_event: asyncio.Event,
    *,
    api_key: str | None,
    api_key_header: str,
    retries: int,
    retry_backoff: float,
    client: httpx.AsyncClient | None = None,
) -> None:
    """Continuously POST synthetic workload payloads to given targets.

    Args:
        targets: Base URLs for demo services.
        interval: Delay between batches in seconds.
        profile: Normalized load profile.
        stop_event: Stop flag controlled by signal handlers.
        api_key: Token that authorizes POST /workload calls.
        api_key_header: Header used to transport the token.
    """

    headers = {api_key_header: api_key} if api_key else None
    owns_client = client is None
    http_client = client or httpx.AsyncClient(timeout=5.0, headers=headers)

    try:
        async for body in payload_stream(profile):
            if stop_event.is_set():
                break
            for target in targets:
                await _post_with_retry(
                    http_client,
                    f"{target}/workload",
                    body,
                    retries=retries,
                    retry_backoff=retry_backoff,
                )
            await asyncio.sleep(interval)
    finally:
        if owns_client:
            await http_client.aclose()


async def _post_with_retry(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, float],
    *,
    retries: int,
    retry_backoff: float,
) -> httpx.Response | None:
    """Send POST request with bounded retries and exponential backoff."""

    attempts = retries + 1
    for attempt in range(1, attempts + 1):
        try:
            response = await client.post(url, json=payload)
            log_structured(
                LOGGER,
                "sent synthetic payload",
                target=url,
                status=response.status_code,
                attempt=attempt,
            )
            return response
        except httpx.HTTPError as exc:
            LOGGER.warning(
                "Load generator request failed (attempt %s/%s): %s",
                attempt,
                attempts,
                exc,
            )
            if attempt == attempts:
                LOGGER.error("Giving up sending payload to %s", url)
                return None
            delay = retry_backoff * (2 ** (attempt - 1))
            if delay > 0:
                await asyncio.sleep(delay)
    return None


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for the load generator."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets",
        nargs="+",
        required=True,
        help="List of service base URLs to hit (e.g. http://demo-service-a:8000)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Delay between batches in seconds",
    )
    parser.add_argument(
        "--minutes",
        type=int,
        default=60,
        help="Profile length in minutes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic profile",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API token for authenticating workload requests. "
        "Defaults to AUTOSCALER_API_TOKEN env variable.",
    )
    parser.add_argument(
        "--api-key-header",
        type=str,
        default=DEFAULT_API_KEY_HEADER,
        help=f"Header used for API token authentication (default: {DEFAULT_API_KEY_HEADER})",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Number of retry attempts per target (default: 2).",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=0.5,
        help="Base delay (in seconds) for exponential backoff between retries.",
    )
    return parser


async def _run_async(
    targets: list[str],
    interval: float,
    minutes: int,
    seed: int,
    api_key: str,
    api_key_header: str,
    retries: int,
    retry_backoff: float,
) -> None:
    """Entry point wiring profile generation and async loop.

    Args:
        targets: List of service base URLs.
        interval: Delay between payload batches.
        minutes: Profile length before repeating.
        seed: RNG seed to keep experiments reproducible.
        api_key: Token that authenticates against the demo service.
        api_key_header: Header used to transport the token.
    """
    pattern = generate_profile(PatternConfig(minutes=minutes, seed=seed))
    stop_event = asyncio.Event()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    await hit_targets(
        targets,
        interval,
        pattern,
        stop_event,
        api_key=api_key,
        api_key_header=api_key_header,
        retries=retries,
        retry_backoff=retry_backoff,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    api_key = args.api_key or os.getenv("AUTOSCALER_API_TOKEN")
    if not api_key:
        raise SystemExit("API key is required. Provide --api-key or export AUTOSCALER_API_TOKEN.")
    LOGGER.info(
        "Starting load generator",
        extra={"targets": args.targets},
    )
    asyncio.run(
        _run_async(
            args.targets,
            args.interval,
            args.minutes,
            args.seed,
            api_key,
            args.api_key_header,
            args.retries,
            args.retry_backoff,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
