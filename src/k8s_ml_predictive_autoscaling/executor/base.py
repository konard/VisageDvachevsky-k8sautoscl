"""Executor protocol -- structural typing for scaling backends.

Using ``typing.Protocol`` instead of ABC to keep coupling loose:
any object implementing ``get_current_replicas``, ``scale``, and
``get_pod_statuses`` satisfies the contract without inheriting from a base.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class PodStatus:
    """Lightweight status of a single pod."""

    name: str
    ready: bool
    phase: str  # Running, Pending, Failed, ...
    restart_count: int = 0


@runtime_checkable
class ScalingExecutor(Protocol):
    """Interface that any executor backend must satisfy.

    Implementations:
      - ``MockExecutor``  -- in-memory, for Docker Compose & tests
      - ``K8sExecutor``   -- real Kubernetes API calls
    """

    async def get_current_replicas(self) -> int:
        """Return current replica count of the target deployment."""
        ...

    async def scale(self, desired_replicas: int, *, reason: str = "") -> bool:
        """Apply a scaling action.

        Args:
            desired_replicas: Target replica count.
            reason: Human-readable reason (for logging / audit).

        Returns:
            True if the scaling was applied successfully.
        """
        ...

    async def get_pod_statuses(self) -> list[PodStatus]:
        """Return lightweight status of all pods in the target deployment."""
        ...
