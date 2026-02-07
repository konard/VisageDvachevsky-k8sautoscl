"""Locust workload definition hitting the demo FastAPI service."""

from __future__ import annotations

from itertools import cycle

from locust import HttpUser, between, task

from tools.load_generator.synthetic_patterns import PatternConfig, generate_profile

PROFILE = generate_profile(PatternConfig(minutes=120, seed=7))
PROFILE_ITER = cycle(PROFILE)


class DemoServiceUser(HttpUser):
    wait_time = between(0.1, 1.0)

    def _payload(self) -> dict[str, float]:
        next_value = next(PROFILE_ITER)
        return {
            "payload_size": 64,
            "cpu_hint": 0.02 + next_value * 0.05,
        }

    @task(3)
    def submit_workload(self) -> None:
        self.client.post("/workload", json=self._payload())

    @task
    def check_health(self) -> None:
        self.client.get("/health")
