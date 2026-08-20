"""Test doubles for the Phase 2 FinOps suite.

Phase 2 is arithmetic and API shaping, so unlike Phase 1 (whose subject was SQL
semantics and therefore needed a real Postgres) almost everything here is
exercised against pure functions. The executor below exists only for the handful
of components whose job is to build a statement or shape a row, and it records
what was asked so a test can assert on the statement rather than on a database's
reaction to it.
"""

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from types import MappingProxyType

from litellm.proxy.witos.finops.history import DailyUsage, ScopeRef, UsageHistory, build_history
from litellm.proxy.witos.finops.pricing import build_price_book

Row = Mapping[str, object]


class RecordingExecutor:
    """``SqlExecutor`` that answers from canned rows and remembers every call."""

    def __init__(
        self,
        responses: Sequence[tuple[str, Sequence[Row]]] = (),
        *,
        row_count: int = 1,
    ) -> None:
        self.responses = list(responses)
        self.row_count = row_count
        self.queries: list[tuple[str, tuple[object, ...]]] = []
        self.executions: list[tuple[str, tuple[object, ...]]] = []

    def _match(self, sql: str) -> Sequence[Row]:
        for fragment, rows in self.responses:
            if fragment in sql:
                return rows
        return ()

    async def query(self, sql: str, *args: object) -> Sequence[Row]:
        self.queries.append((sql, args))
        return self._match(sql)

    async def execute(self, sql: str, *args: object) -> int:
        self.executions.append((sql, args))
        return self.row_count

    def statements(self) -> tuple[str, ...]:
        return tuple(sql for sql, _ in self.queries + self.executions)


def day_usage(
    day: date,
    *,
    requests: float = 100.0,
    prompt: float = 100_000.0,
    completion: float = 25_000.0,
    cache: float = 0.0,
    spend: str = "1.0",
    mix: Mapping[str, float] | None = None,
) -> DailyUsage:
    return DailyUsage(
        day=day,
        request_count=requests,
        prompt_tokens=prompt,
        completion_tokens=completion,
        cache_read_tokens=cache,
        spend_usd=Decimal(spend),
        mix_tokens=MappingProxyType(dict(mix) if mix else {"gpt-4o-mini": prompt + completion}),
    )


def flat_history(
    days: int,
    *,
    start: date = date(2026, 5, 1),
    scope: ScopeRef | None = None,
    prompt: float = 100_000.0,
    spend_per_day: str = "1.0",
) -> UsageHistory:
    """A history with no trend and no seasonality, for boundary and state tests."""
    built = build_history(
        scope or ScopeRef("team", "t1"),
        tuple(day_usage(start + timedelta(days=offset), prompt=prompt, spend=spend_per_day) for offset in range(days)),
        as_of=start + timedelta(days=days - 1),
    )
    assert built is not None
    return built


def priced_day(
    day: date,
    *,
    prompt: float,
    completion: float,
    cache: float,
    model: str = "gpt-4o-mini",
    requests: float | None = None,
) -> DailyUsage:
    """A day whose spend is what LiteLLM's own calculator would have charged for it.

    Fixtures price themselves through the real cost calculator so the engine's
    fidelity check (§1.3) sees a scope whose invoice its price book can explain,
    which is the case the token-based strategy is for.
    """
    blended = build_price_book((model,)).blend({model: 1.0})
    return day_usage(
        day,
        requests=prompt / 2000 if requests is None else requests,
        prompt=prompt,
        completion=completion,
        cache=cache,
        spend=str(
            blended.cost_of(
                prompt_tokens=Decimal(str(prompt)),
                completion_tokens=Decimal(str(completion)),
                cache_read_tokens=Decimal(str(cache)),
            )
        ),
        mix={model: prompt + completion},
    )


def seasonal_history(
    days: int,
    *,
    start: date = date(2026, 5, 1),
    base: float = 1_000_000.0,
    growth: float = 0.0,
    weekend_factor: float = 0.4,
    scope: ScopeRef | None = None,
) -> UsageHistory:
    """Weekday/weekend seasonality with optional growth, priced through the cost calculator."""
    rows = tuple(
        priced_day(
            start + timedelta(days=offset),
            prompt=_seasonal_level(base, growth, weekend_factor, start, offset),
            completion=_seasonal_level(base, growth, weekend_factor, start, offset) * 0.25,
            cache=_seasonal_level(base, growth, weekend_factor, start, offset) * 0.1,
        )
        for offset in range(days)
    )
    built = build_history(scope or ScopeRef("team", "t1"), rows, as_of=start + timedelta(days=days - 1))
    assert built is not None
    return built


def _seasonal_level(base: float, growth: float, weekend_factor: float, start: date, offset: int) -> float:
    weekend = (start + timedelta(days=offset)).weekday() >= 5
    return base * (1 + growth) ** offset * (weekend_factor if weekend else 1.0)


def utc(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)
