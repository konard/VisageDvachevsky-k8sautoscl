"""Tests for K8sExecutor using mocked kubernetes client.

These tests mock the kubernetes Python client to avoid needing a real cluster.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# Skip entire module if kubernetes is not installed
pytest.importorskip("kubernetes", reason="kubernetes package not installed")

from k8s_ml_predictive_autoscaling.executor.k8s_executor import K8sExecutor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_deployment(replicas: int = 3, labels: dict | None = None):
    """Create a mock Deployment object."""
    dep = MagicMock()
    dep.spec.replicas = replicas
    dep.spec.selector.match_labels = labels or {"app": "demo-service"}
    return dep


def _mock_pod(name: str, phase: str = "Running", ready: bool = True, restarts: int = 0):
    """Create a mock Pod object."""
    pod = MagicMock()
    pod.metadata.name = name
    pod.status.phase = phase
    cs = MagicMock()
    cs.ready = ready
    cs.restart_count = restarts
    pod.status.container_statuses = [cs]
    return pod


@pytest.fixture
def executor():
    """Create a K8sExecutor with mocked kubernetes config and API clients."""
    with (
        patch("k8s_ml_predictive_autoscaling.executor.k8s_executor.k8s_config") as mock_config,
        patch("k8s_ml_predictive_autoscaling.executor.k8s_executor.k8s_client") as mock_client,
    ):
        # Simulate in-cluster config loading
        mock_config.load_incluster_config.return_value = None
        mock_config.ConfigException = Exception

        # Build mock API instances
        mock_apps = MagicMock()
        mock_core = MagicMock()
        mock_client.AppsV1Api.return_value = mock_apps
        mock_client.CoreV1Api.return_value = mock_core

        ex = K8sExecutor(
            deployment_name="demo-service",
            namespace="predictive-autoscaling",
        )
        # Expose mocks for test assertions
        ex._mock_apps = mock_apps
        ex._mock_core = mock_core
        yield ex


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestK8sExecutor:

    @pytest.mark.asyncio
    async def test_get_current_replicas(self, executor: K8sExecutor) -> None:
        executor._mock_apps.read_namespaced_deployment.return_value = _mock_deployment(5)

        replicas = await executor.get_current_replicas()
        assert replicas == 5
        executor._mock_apps.read_namespaced_deployment.assert_called_once_with(
            "demo-service", "predictive-autoscaling"
        )

    @pytest.mark.asyncio
    async def test_scale_patches_deployment(self, executor: K8sExecutor) -> None:
        ok = await executor.scale(6, reason="test scale")
        assert ok is True
        executor._mock_apps.patch_namespaced_deployment_scale.assert_called_once_with(
            "demo-service",
            "predictive-autoscaling",
            {"spec": {"replicas": 6}},
        )

    @pytest.mark.asyncio
    async def test_scale_returns_false_on_error(self, executor: K8sExecutor) -> None:
        executor._mock_apps.patch_namespaced_deployment_scale.side_effect = Exception("API error")
        ok = await executor.scale(4, reason="fail test")
        assert ok is False

    @pytest.mark.asyncio
    async def test_get_pod_statuses(self, executor: K8sExecutor) -> None:
        executor._mock_apps.read_namespaced_deployment.return_value = _mock_deployment(3)

        pod_list = MagicMock()
        pod_list.items = [
            _mock_pod("pod-1", "Running", True, 0),
            _mock_pod("pod-2", "Running", True, 2),
            _mock_pod("pod-3", "Pending", False, 0),
        ]
        executor._mock_core.list_namespaced_pod.return_value = pod_list

        statuses = await executor.get_pod_statuses()
        assert len(statuses) == 3
        assert statuses[0].name == "pod-1"
        assert statuses[0].ready is True
        assert statuses[1].restart_count == 2
        assert statuses[2].phase == "Pending"
        assert statuses[2].ready is False
