"""Real Kubernetes executor using the official Python client.

Patches the ``spec.replicas`` field on a Deployment via the Apps/v1 API.
Requires:
  - ``kubernetes`` package installed
  - Proper RBAC: get/patch on deployments, list pods
  - In-cluster config (ServiceAccount) or local kubeconfig
"""

from __future__ import annotations

import asyncio
import logging
from functools import partial

from .base import PodStatus

logger = logging.getLogger(__name__)

# Lazy import -- the kubernetes package is optional (not needed in Docker Compose mode)
try:
    from kubernetes import client as k8s_client
    from kubernetes import config as k8s_config

    HAS_K8S = True
except ImportError:
    HAS_K8S = False


class K8sExecutorError(RuntimeError):
    """Raised when a Kubernetes API call fails."""


class K8sExecutor:
    """Executor that patches Deployment replicas via the Kubernetes API.

    Automatically detects in-cluster vs local kubeconfig.

    Args:
        deployment_name: Target Deployment name.
        namespace: Target namespace.
    """

    def __init__(
        self,
        deployment_name: str,
        namespace: str = "predictive-autoscaling",
    ) -> None:
        if not HAS_K8S:
            raise ImportError(
                "The 'kubernetes' package is required for K8sExecutor. "
                "Install it with: pip install kubernetes"
            )

        self._deployment = deployment_name
        self._namespace = namespace

        # Load config (in-cluster or local kubeconfig)
        try:
            k8s_config.load_incluster_config()
            logger.info("Using in-cluster K8s config")
        except k8s_config.ConfigException:
            k8s_config.load_kube_config()
            logger.info("Using local kubeconfig")

        self._apps_v1 = k8s_client.AppsV1Api()
        self._core_v1 = k8s_client.CoreV1Api()

    async def get_current_replicas(self) -> int:
        """Get current replica count from the Deployment spec."""
        try:
            deployment = await asyncio.get_event_loop().run_in_executor(
                None,
                partial(
                    self._apps_v1.read_namespaced_deployment,
                    self._deployment,
                    self._namespace,
                ),
            )
            return deployment.spec.replicas or 0
        except Exception as exc:
            raise K8sExecutorError(
                f"Failed to get replicas for {self._namespace}/{self._deployment}: {exc}"
            ) from exc

    async def scale(self, desired_replicas: int, *, reason: str = "") -> bool:
        """Patch the Deployment's spec.replicas.

        Args:
            desired_replicas: Target replica count.
            reason: Scaling reason (logged only).

        Returns:
            True if the patch succeeded.
        """
        try:
            body = {"spec": {"replicas": desired_replicas}}
            await asyncio.get_event_loop().run_in_executor(
                None,
                partial(
                    self._apps_v1.patch_namespaced_deployment_scale,
                    self._deployment,
                    self._namespace,
                    body,
                ),
            )
            logger.info(
                "K8s scale %s/%s to %d (%s)",
                self._namespace,
                self._deployment,
                desired_replicas,
                reason,
            )
            return True
        except Exception as exc:
            logger.error(
                "K8s scale failed for %s/%s: %s",
                self._namespace,
                self._deployment,
                exc,
            )
            return False

    async def get_pod_statuses(self) -> list[PodStatus]:
        """List pods matching the deployment's label selector."""
        try:
            deployment = await asyncio.get_event_loop().run_in_executor(
                None,
                partial(
                    self._apps_v1.read_namespaced_deployment,
                    self._deployment,
                    self._namespace,
                ),
            )
            # Build label selector from deployment spec
            match_labels = deployment.spec.selector.match_labels or {}
            selector = ",".join(f"{k}={v}" for k, v in match_labels.items())

            pods = await asyncio.get_event_loop().run_in_executor(
                None,
                partial(
                    self._core_v1.list_namespaced_pod,
                    self._namespace,
                    label_selector=selector,
                ),
            )

            statuses: list[PodStatus] = []
            for pod in pods.items:
                restart_count = 0
                if pod.status.container_statuses:
                    restart_count = sum(
                        cs.restart_count for cs in pod.status.container_statuses
                    )
                statuses.append(
                    PodStatus(
                        name=pod.metadata.name,
                        ready=all(
                            cs.ready
                            for cs in (pod.status.container_statuses or [])
                        ),
                        phase=pod.status.phase or "Unknown",
                        restart_count=restart_count,
                    )
                )
            return statuses
        except Exception as exc:
            logger.error("Failed to list pods: %s", exc)
            return []
