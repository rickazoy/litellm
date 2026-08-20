"""Per-connection circuit breaker for delegated evaluation (§2.8).

Five failures inside thirty seconds trips a connection. While tripped, the
guardrail stops calling that vendor and applies the configured fail mode
immediately, which is the difference between a degraded vendor costing 800 ms
per request and a degraded vendor costing 800 ms per request forever.

The clock is injected. A breaker whose behaviour can only be tested by sleeping
is a breaker nobody tests.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Final


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True, slots=True)
class BreakerConfig:
    failure_threshold: int = 5
    window_seconds: float = 30.0
    cooldown_seconds: float = 30.0


@dataclass(frozen=True, slots=True)
class _ConnectionState:
    failure_times: tuple[float, ...] = ()
    opened_at: float | None = None


class CircuitBreaker:
    def __init__(self, config: BreakerConfig | None = None, clock: Callable[[], float] = time.monotonic) -> None:
        self._config: Final = config if config is not None else BreakerConfig()
        self._clock: Final = clock
        # mutable-ok: per-connection state map, keyed by connection id and never handed out
        self._states: Final[dict[str, _ConnectionState]] = {}  # mutable-ok: per-connection state map, never handed out

    def state(self, connection_id: str) -> BreakerState:
        current: Final = self._states.get(connection_id, _ConnectionState())
        if current.opened_at is None:
            return BreakerState.CLOSED
        if self._clock() - current.opened_at >= self._config.cooldown_seconds:
            return BreakerState.HALF_OPEN
        return BreakerState.OPEN

    def allows(self, connection_id: str) -> bool:
        return self.state(connection_id) is not BreakerState.OPEN

    def record_success(self, connection_id: str) -> None:
        self._states[connection_id] = _ConnectionState()

    def record_failure(self, connection_id: str) -> None:
        now: Final = self._clock()
        if self.state(connection_id) is BreakerState.HALF_OPEN:
            self._states[connection_id] = _ConnectionState(failure_times=(), opened_at=now)
            return
        current: Final = self._states.get(connection_id, _ConnectionState())
        recent: Final = tuple(
            moment for moment in (*current.failure_times, now) if now - moment < self._config.window_seconds
        )
        tripped: Final = len(recent) >= self._config.failure_threshold
        self._states[connection_id] = _ConnectionState(
            failure_times=() if tripped else recent,
            opened_at=now if tripped else current.opened_at,
        )
