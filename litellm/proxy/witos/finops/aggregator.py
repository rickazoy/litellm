"""Incremental FinOps aggregation on a watermark plus lateness window (blueprint §1.5).

Spend rows are written asynchronously by ``DBSpendUpdateWriter``, so a row for
10:59 can reach the table after a cycle that already read past 11:00. A plain
"everything since the watermark" sweep would miss it forever. Each cycle
therefore rewinds to ``watermark - lateness_window`` and recomputes every bucket
in that overlap from source. Correctness rests on the recompute being a
replacement rather than an increment (see ``aggregation_sql``): re-sweeping a
bucket that has not changed rewrites the same numbers, and re-sweeping one that
gained a late row rewrites it including that row.

The watermark itself only ever advances to the top of the current hour, never
into it, so the partial hour is guaranteed to be swept again while it is still
filling.

Nothing in this module runs on the inference request path.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Final, Literal, TypeAlias, TypedDict

from pydantic import TypeAdapter
from typing_extensions import ReadOnly

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy.witos.finops.aggregation_sql import (
    DAILY_UPSERT_SQL,
    PRICING_COVERAGE_SQL,
    SCOPE_SPECS,
    ScopeType,
    hourly_upsert_sql,
)
from litellm.proxy.witos.finops.config import HOURLY_WATERMARK_KEY, FinOpsConfig
from litellm.proxy.witos.finops.metrics import FinOpsMetrics
from litellm.proxy.witos.shared.scheduler import JobLeaseManager, LeaseHeld
from litellm.proxy.witos.shared.sql import SqlExecutor

AGGREGATION_JOB_NAME: Final = "witos_finops_aggregation"

# Sweeping less than a full hour of overlap would let the partial current hour
# fall out of the window between cycles, so the floor is a hard invariant rather
# than a default.
MIN_LATENESS_WINDOW: Final = timedelta(hours=1)
DEFAULT_LATENESS_WINDOW: Final = timedelta(hours=2)

# On a database with no watermark yet, only the overlap is swept. History is the
# backfill CLI's job; silently scanning every spend log at first boot is not.
DEFAULT_COLD_START_LOOKBACK: Final = timedelta(hours=2)

Clock: TypeAlias = Callable[[], datetime]  # mutable-ok: Callable's parameter list, not a list literal


class _PricingCoverageRow(TypedDict):
    model: ReadOnly[str | None]
    request_count: ReadOnly[int]
    zero_priced_requests: ReadOnly[int]


_PRICING_COVERAGE_ROWS: Final = TypeAdapter(tuple[_PricingCoverageRow, ...])


@dataclass(frozen=True, slots=True)
class AggregationWindow:
    """The half-open span of spend-log time one cycle recomputes."""

    start: datetime
    end: datetime
    next_watermark: datetime

    @property
    def daily_start(self) -> datetime:
        return _floor_day(self.start)

    @property
    def daily_end(self) -> datetime:
        return _floor_day(self.end) + timedelta(days=1)


@dataclass(frozen=True, slots=True)
class PricingCoverage:
    """How much of the swept window LiteLLM's cost map could actually price."""

    missing_price_requests: int
    unknown_model_requests: int


@dataclass(frozen=True, slots=True)
class CycleCompleted:
    window: AggregationWindow
    hourly_rows: int
    daily_rows: int
    coverage: PricingCoverage
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class CycleSkipped:
    reason: Literal["lease_held"]
    held_by: str


CycleOutcome: TypeAlias = CycleCompleted | CycleSkipped


def _floor_hour(moment: datetime) -> datetime:
    return moment.replace(minute=0, second=0, microsecond=0)


def _floor_day(moment: datetime) -> datetime:
    return moment.replace(hour=0, minute=0, second=0, microsecond=0)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def plan_window(
    *,
    watermark: datetime | None,
    now: datetime,
    lateness_window: timedelta = DEFAULT_LATENESS_WINDOW,
    cold_start_lookback: timedelta = DEFAULT_COLD_START_LOOKBACK,
) -> AggregationWindow:
    """The span to recompute, given how far aggregation previously got.

    ``end`` is the caller's now rather than the last closed hour, so the
    in-flight hour shows up for intraday burn; ``next_watermark`` deliberately
    stops at the top of that hour so the next cycle sweeps it again.
    """
    overlap: Final = max(lateness_window, MIN_LATENESS_WINDOW)
    start: Final = _floor_hour(now - cold_start_lookback if watermark is None else watermark - overlap)
    return AggregationWindow(start=start, end=now, next_watermark=_floor_hour(now))


def _priced_model_names() -> frozenset[str]:
    """The model names LiteLLM's cost map can price.

    That map is the only pricing source WIT OS may use (blueprint §0.1); it is a
    plain dict assembled at import time and carries no type information.
    """
    return frozenset(litellm.model_cost)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType, reportUnknownVariableType]  # untyped price map  # fmt: skip


def _hourly_arguments(
    scope_type: ScopeType, window: AggregationWindow, approved_dimensions: Sequence[str]
) -> tuple[object, ...]:
    if not SCOPE_SPECS[scope_type].takes_tag_dimensions:
        return (window.start, window.end)
    dimensions: Final = list(approved_dimensions)  # mutable-ok: an array bind needs a list, not a tuple
    return (window.start, window.end, dimensions)


class FinOpsAggregator:
    """Recomputes the hourly and daily fact tables from ``LiteLLM_SpendLogs``."""

    def __init__(
        self,
        executor: SqlExecutor,
        *,
        lateness_window: timedelta = DEFAULT_LATENESS_WINDOW,
        scope_types: Sequence[ScopeType] = tuple(SCOPE_SPECS),
        lease_manager: JobLeaseManager | None = None,
        clock: Clock = _utc_now,
    ) -> None:
        unknown: Final = tuple(scope for scope in scope_types if scope not in SCOPE_SPECS)
        if unknown:
            raise ValueError(f"Unknown FinOps scope types: {unknown}")
        self._executor: Final = executor
        self._config: Final = FinOpsConfig(executor)
        self._lateness_window: Final = max(lateness_window, MIN_LATENESS_WINDOW)
        self._scope_types: Final = tuple(scope_types)
        self._lease_manager: Final = lease_manager or JobLeaseManager(executor)
        self._clock: Final = clock

    async def run_cycle(self) -> CycleOutcome:
        """Sweep the overlap window under the job lease and advance the watermark."""
        started: Final = time.monotonic()
        async with self._lease_manager.hold(AGGREGATION_JOB_NAME) as lease:
            if isinstance(lease, LeaseHeld):
                FinOpsMetrics.record_cycle("skipped_locked")
                return CycleSkipped(reason="lease_held", held_by=lease.held_by)

            now: Final = self._clock()
            window: Final = plan_window(
                watermark=await self._config.watermark(HOURLY_WATERMARK_KEY),
                now=now,
                lateness_window=self._lateness_window,
            )
            hourly_rows: Final = await self.aggregate_hourly(window)
            daily_rows: Final = await self.roll_up_daily(window)
            coverage: Final = await self.measure_pricing_coverage(window)
            await self._config.set_watermark(HOURLY_WATERMARK_KEY, window.next_watermark)

            duration: Final = time.monotonic() - started
            FinOpsMetrics.record_cycle("completed", duration)
            FinOpsMetrics.record_rows("hourly", hourly_rows)
            FinOpsMetrics.record_rows("daily", daily_rows)
            FinOpsMetrics.record_pricing_coverage(coverage.missing_price_requests, coverage.unknown_model_requests)
            FinOpsMetrics.set_last_success(now.timestamp())
            FinOpsMetrics.set_aggregation_lag((now - window.next_watermark).total_seconds())
            verbose_proxy_logger.info(
                "WIT OS FinOps aggregated %s..%s: %d hourly rows, %d daily rows in %.2fs",
                window.start.isoformat(),
                window.end.isoformat(),
                hourly_rows,
                daily_rows,
                duration,
            )
            return CycleCompleted(
                window=window,
                hourly_rows=hourly_rows,
                daily_rows=daily_rows,
                coverage=coverage,
                duration_seconds=duration,
            )

    async def aggregate_hourly(self, window: AggregationWindow) -> int:
        approved_dimensions: Final = await self._config.approved_tag_dimensions()
        counts: Final = tuple(
            [
                await self._executor.execute(
                    hourly_upsert_sql(scope_type), *_hourly_arguments(scope_type, window, approved_dimensions)
                )
                for scope_type in self._scope_types
            ]
        )
        return sum(counts)

    async def roll_up_daily(self, window: AggregationWindow) -> int:
        """Rebuild every calendar day the hourly sweep touched.

        Whole days only: rolling up a partial day would write an undercount that
        no later cycle is obliged to revisit.
        """
        return await self._executor.execute(DAILY_UPSERT_SQL, window.daily_start, window.daily_end)

    async def measure_pricing_coverage(self, window: AggregationWindow) -> PricingCoverage:
        """Count requests the cost map could not price, per blueprint §1.14.

        Zero spend against non-zero tokens means no price was applied, which is
        indistinguishable at this layer from a genuinely free model, so it is
        reported as coverage rather than treated as an error.
        """
        rows: Final = _PRICING_COVERAGE_ROWS.validate_python(
            await self._executor.query(PRICING_COVERAGE_SQL, window.start, window.end)
        )
        priced_models: Final = _priced_model_names()
        return PricingCoverage(
            missing_price_requests=sum(row["zero_priced_requests"] for row in rows),
            unknown_model_requests=sum(
                row["request_count"] for row in rows if row["model"] is not None and row["model"] not in priced_models
            ),
        )
