"""In-memory mock executor for Docker Compose and unit tests.

Records every scaling decision in a log for debugging/assertions
without calling any real Kubernetes API.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from .base import PodStatus

logger = logging.getLogger(__name__)


@dataclass
class ScaleEvent:
    """One recorded scaling action."""

    timestamp: float
    previous_replicas: int
    desired_replicas: int
    reason: str
    success: bool


class MockExecutor:
    """Satisfies :class:`ScalingExecutor` protocol entirely in-memory.

    Attributes:
        replicas: Current (simulated) replica count.
        history: Ordered list of scaling events.
    """

    def __init__(self, initial_replicas: int = 2) -> None:
        self.replicas: int = initial_replicas
        self.history: list[ScaleEvent] = []

    # -- Protocol methods ------------------------------------------------------

    async def get_current_replicas(self) -> int:
        return self.replicas

    async def scale(self, desired_replicas: int, *, reason: str = "") -> bool:
        previous = self.replicas
        self.replicas = desired_replicas
        event = ScaleEvent(
            timestamp=time.monotonic(),
            previous_replicas=previous,
            desired_replicas=desired_replicas,
            reason=reason,
            success=True,
        )
        self.history.append(event)
        logger.info(
            "MockExecutor: %d -> %d (%s)",
            previous,
            desired_replicas,
            reason,
        )
        return True

    async def get_pod_statuses(self) -> list[PodStatus]:
        """Simulate N healthy pods matching current replica count."""
        return [
            PodStatus(
                name=f"mock-pod-{i}",
                ready=True,
                phase="Running",
                restart_count=0,
            )
            for i in range(self.replicas)
        ]
