"""Tests for the MockExecutor."""

from __future__ import annotations

import pytest

from k8s_ml_predictive_autoscaling.executor.base import PodStatus, ScalingExecutor
from k8s_ml_predictive_autoscaling.executor.mock_executor import MockExecutor


class TestMockExecutor:

    @pytest.mark.asyncio
    async def test_initial_replicas(self) -> None:
        ex = MockExecutor(initial_replicas=3)
        assert await ex.get_current_replicas() == 3

    @pytest.mark.asyncio
    async def test_scale_changes_replicas(self) -> None:
        ex = MockExecutor(initial_replicas=2)
        ok = await ex.scale(5, reason="test scale-up")
        assert ok is True
        assert await ex.get_current_replicas() == 5
        assert len(ex.history) == 1
        assert ex.history[0].previous_replicas == 2
        assert ex.history[0].desired_replicas == 5
        assert ex.history[0].reason == "test scale-up"

    @pytest.mark.asyncio
    async def test_pod_statuses_match_replicas(self) -> None:
        ex = MockExecutor(initial_replicas=4)
        statuses = await ex.get_pod_statuses()
        assert len(statuses) == 4
        assert all(isinstance(s, PodStatus) for s in statuses)
        assert all(s.ready for s in statuses)
        assert all(s.phase == "Running" for s in statuses)

    @pytest.mark.asyncio
    async def test_statuses_update_after_scale(self) -> None:
        ex = MockExecutor(initial_replicas=2)
        await ex.scale(6, reason="grow")
        statuses = await ex.get_pod_statuses()
        assert len(statuses) == 6

    @pytest.mark.asyncio
    async def test_satisfies_protocol(self) -> None:
        """MockExecutor is a structural subtype of ScalingExecutor."""
        ex = MockExecutor()
        assert isinstance(ex, ScalingExecutor)

    @pytest.mark.asyncio
    async def test_multiple_scale_events_recorded(self) -> None:
        ex = MockExecutor(initial_replicas=2)
        await ex.scale(4, reason="up-1")
        await ex.scale(3, reason="down-1")
        await ex.scale(5, reason="up-2")
        assert len(ex.history) == 3
        assert [e.desired_replicas for e in ex.history] == [4, 3, 5]
