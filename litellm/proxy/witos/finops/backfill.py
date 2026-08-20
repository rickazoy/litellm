"""Rebuild FinOps facts for a historical range.

    python -m litellm.proxy.witos.finops.backfill --days 90

The steady-state aggregator only ever sweeps its overlap window, so a fork that
turns FinOps on has no history until this runs. It reuses the same recomputing
upserts, which is what makes it idempotent: running it twice, or over a range the
scheduled job already covered, produces identical buckets.

Coverage is bounded by spend-log retention. ``maximum_spend_logs_retention_period``
purges ``LiteLLM_SpendLogs``, so ``--days 90`` against a 30-day retention yields
30 days of facts and 60 days of nothing, not 90 days of zeroes.

The watermark is never moved backwards here. Rewinding it would make the next
scheduled cycle re-sweep the whole backfilled range at the 5-minute cadence.
"""

import argparse
import asyncio
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Final, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict

from litellm._logging import verbose_proxy_logger
from litellm.proxy.witos.finops.aggregation_sql import SCOPE_SPECS, ScopeType
from litellm.proxy.witos.finops.aggregator import AggregationWindow, FinOpsAggregator
from litellm.proxy.witos.shared.scheduler import JobLeaseManager, LeaseHeld
from litellm.proxy.witos.shared.sql import PrismaSqlExecutor, SqlExecutor

BACKFILL_JOB_NAME: Final = "witos_finops_backfill"
DEFAULT_CHUNK: Final = timedelta(days=1)


@dataclass(frozen=True, slots=True)
class BackfillCompleted:
    start: datetime
    end: datetime
    chunks: int
    hourly_rows: int
    daily_rows: int


@dataclass(frozen=True, slots=True)
class BackfillSkipped:
    held_by: str


BackfillOutcome: TypeAlias = BackfillCompleted | BackfillSkipped


def chunk_windows(start: datetime, end: datetime, chunk: timedelta) -> tuple[AggregationWindow, ...]:
    """Split a range into ascending windows.

    Ascending order matters: a calendar day split across two chunks is rolled up
    once per chunk, and only the last of those sees every hour of that day.
    """
    if chunk <= timedelta(0):
        raise ValueError("backfill chunk must be positive")
    span: Final = max((end - start).total_seconds(), 0.0)
    count: Final = -(-int(span) // int(chunk.total_seconds()))
    return tuple(
        AggregationWindow(
            start=start + chunk * index,
            end=min(start + chunk * (index + 1), end),
            next_watermark=start + chunk * index,
        )
        for index in range(count)
    )


async def backfill(
    executor: SqlExecutor,
    *,
    days: int,
    end: datetime | None = None,
    chunk: timedelta = DEFAULT_CHUNK,
    scope_types: Sequence[ScopeType] = tuple(SCOPE_SPECS),
    lease_manager: JobLeaseManager | None = None,
) -> BackfillOutcome:
    if days <= 0:
        raise ValueError("--days must be positive")
    finish: Final = end or datetime.now(timezone.utc)
    begin: Final = finish - timedelta(days=days)
    aggregator: Final = FinOpsAggregator(executor, scope_types=scope_types)
    leases: Final = lease_manager or JobLeaseManager(executor)

    async with leases.hold(BACKFILL_JOB_NAME, ttl=timedelta(hours=6)) as lease:
        if isinstance(lease, LeaseHeld):
            return BackfillSkipped(held_by=lease.held_by)
        windows: Final = chunk_windows(begin, finish, chunk)
        results: Final = tuple(
            [(await aggregator.aggregate_hourly(window), await aggregator.roll_up_daily(window)) for window in windows]
        )
        for window, (hourly, daily) in zip(windows, results):
            verbose_proxy_logger.info(
                "WIT OS FinOps backfilled %s: %d hourly rows, %d daily rows",
                window.start.date().isoformat(),
                hourly,
                daily,
            )
        return BackfillCompleted(
            start=begin,
            end=finish,
            chunks=len(windows),
            hourly_rows=sum(hourly for hourly, _ in results),
            daily_rows=sum(daily for _, daily in results),
        )


class _Arguments(BaseModel):
    """The CLI surface, validated out of argparse's untyped namespace."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    days: int
    end: datetime | None = None
    chunk_days: int = 1
    scope_type: tuple[ScopeType, ...] | None = None

    @property
    def end_utc(self) -> datetime | None:
        if self.end is None:
            return None
        return self.end if self.end.tzinfo is not None else self.end.replace(tzinfo=timezone.utc)

    @property
    def scopes(self) -> tuple[ScopeType, ...]:
        return self.scope_type or tuple(SCOPE_SPECS)


class _PrismaDatabase(Protocol):
    """The slice of the Prisma client the CLI drives."""

    async def connect(self) -> None: ...

    async def disconnect(self) -> None: ...

    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]: ...

    async def execute_raw(self, query: str, *args: object) -> int: ...


def parse_args(argv: Sequence[str] | None = None) -> _Arguments:
    parser: Final = argparse.ArgumentParser(
        prog="python -m litellm.proxy.witos.finops.backfill",
        description="Rebuild WIT OS FinOps hourly and daily facts from LiteLLM_SpendLogs.",
    )
    parser.add_argument("--days", type=int, required=True, help="How many days back from --end to rebuild.")
    parser.add_argument(
        "--end",
        type=datetime.fromisoformat,
        default=None,
        help="ISO-8601 UTC end of the range (exclusive). Defaults to now.",
    )
    parser.add_argument("--chunk-days", type=int, default=1, help="Days rebuilt per statement batch.")
    parser.add_argument(
        "--scope-type",
        action="append",
        choices=sorted(SCOPE_SPECS),
        default=None,
        help="Restrict to one scope type. Repeatable. Defaults to all.",
    )
    return _Arguments.model_validate(vars(parser.parse_args(argv)))


def _prisma_database() -> _PrismaDatabase:
    """Narrow the generated Prisma client, which ships no type information, exactly once."""
    from prisma import (
        Prisma,  # pyright: ignore[reportAttributeAccessIssue, reportUnknownVariableType]  # generated client ships no stubs  # fmt: skip
    )

    return Prisma()  # pyright: ignore[reportUnknownVariableType]  # narrowed by this function's return type  # fmt: skip


async def _run(argv: Sequence[str] | None = None) -> int:
    arguments: Final = parse_args(argv)
    if not os.getenv("DATABASE_URL"):
        verbose_proxy_logger.error("DATABASE_URL is not set; the backfill has no database to read.")
        return 2

    database: Final = _prisma_database()
    await database.connect()
    try:
        outcome: Final = await backfill(
            PrismaSqlExecutor(database),
            days=arguments.days,
            end=arguments.end_utc,
            chunk=timedelta(days=arguments.chunk_days),
            scope_types=arguments.scopes,
        )
    finally:
        await database.disconnect()

    if isinstance(outcome, BackfillSkipped):
        verbose_proxy_logger.error("Another backfill is already running (lease held by %s)", outcome.held_by)
        return 1
    verbose_proxy_logger.info(
        "Backfilled %s..%s in %d chunks: %d hourly rows, %d daily rows",
        outcome.start.isoformat(),
        outcome.end.isoformat(),
        outcome.chunks,
        outcome.hourly_rows,
        outcome.daily_rows,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(_run(argv))


if __name__ == "__main__":
    raise SystemExit(main())
