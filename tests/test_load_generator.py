"""Tests for the async load generator helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx
import pytest

from k8s_ml_predictive_autoscaling.load_generator import _post_with_retry, payload_stream
from k8s_ml_predictive_autoscaling.synthetic import PatternConfig, generate_profile


class DummyAsyncClient(httpx.AsyncClient):
    """Minimal AsyncClient stub for retry unit tests."""

    def __init__(self, responses: Sequence[httpx.Response | Exception]) -> None:
        super().__init__()
        self._responses = list(responses)
        self.calls = 0

    async def post(  # type: ignore[override]
        self,
        url: httpx.URL | str,
        *,
        content: Any = None,
        data: Any = None,
        files: Any = None,
        json: Any = None,
        params: Any = None,
        headers: Any = None,
        cookies: Any = None,
        auth: Any = None,
        follow_redirects: Any = None,
        timeout: Any = None,
        extensions: Any = None,
    ) -> httpx.Response:
        self.calls += 1
        result = self._responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_payload_stream_cycles_values() -> None:
    profile = generate_profile(PatternConfig(minutes=2, seed=1))
    stream = payload_stream(profile)
    first = await stream.__anext__()
    second = await stream.__anext__()
    third = await stream.__anext__()
    assert first != second  # profile changes
    assert third == first  # cycle restarts after exhausting profile


@pytest.mark.asyncio
async def test_post_with_retry_eventually_succeeds() -> None:
    responses: Sequence[httpx.Response | Exception] = [
        httpx.HTTPError("boom"),
        httpx.Response(status_code=202),
    ]
    client = DummyAsyncClient(responses)
    result = await _post_with_retry(
        client,
        "http://example.com/workload",
        {"payload_size": 64, "cpu_hint": 0.1},
        retries=1,
        retry_backoff=0.0,
    )
    assert isinstance(result, httpx.Response)
    assert result.status_code == 202
    assert client.calls == 2


@pytest.mark.asyncio
async def test_post_with_retry_returns_none_after_exhaustion() -> None:
    responses: Sequence[httpx.Response | Exception] = [
        httpx.HTTPError("boom"),
        httpx.HTTPError("boom-again"),
    ]
    client = DummyAsyncClient(responses)
    result = await _post_with_retry(
        client,
        "http://example.com/workload",
        {"payload_size": 64, "cpu_hint": 0.1},
        retries=1,
        retry_backoff=0.0,
    )
    assert result is None
    assert client.calls == 2
