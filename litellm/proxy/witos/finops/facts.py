"""Reading the Phase 1 fact tables: history in, nothing computed here.

Everything the forecast engine sees comes through this module, and it only ever
reads ``WITOS_FinOpsUsageDaily`` and ``WITOS_FinOpsUsageHourly``. Raw
``LiteLLM_SpendLogs`` is deliberately never touched: retention deletes it
(``spend_log_cleanup.py``), so a forecast built from it would silently change
shape the day the purge ran.

Rows are validated into concrete types at this boundary with pydantic, because
the two drivers this SQL runs under disagree about scalars: Prisma hands back a
``Decimal`` column one way and psycopg another, and a ``BigInt`` can arrive as a
string. Downstream code should never have to know that.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from types import MappingProxyType
from typing import Final, TypedDict

from pydantic import TypeAdapter
from typing_extensions import ReadOnly

from litellm.proxy.witos.finops.history import DailyUsage, ScopeRef, UsageHistory, build_history
from litellm.proxy.witos.shared.sql import SqlExecutor

# Scope types a nightly forecast is produced for. Model and tag scopes are
# aggregated for drill-down but nobody holds a budget against them, so
# forecasting all of them would multiply the job's cost for no reader.
DEFAULT_FORECAST_SCOPE_TYPES: Final = ("global", "organization", "team", "user", "key", "end_user")

MAX_HISTORY_DAYS: Final = 365


class _DailyRow(TypedDict):
    bucket_date: ReadOnly[datetime]
    request_count: ReadOnly[int]
    prompt_tokens: ReadOnly[int]
    completion_tokens: ReadOnly[int]
    cache_read_tokens: ReadOnly[int]
    spend_usd: ReadOnly[Decimal]
    breakdown: ReadOnly[Mapping[str, Mapping[str, Decimal]] | None]


class _ScopeRow(TypedDict):
    scope_type: ReadOnly[str]
    scope_id: ReadOnly[str]
    scope_label: ReadOnly[str | None]
    spend_usd: ReadOnly[Decimal]


class _TotalsRow(TypedDict):
    spend_usd: ReadOnly[Decimal]
    total_tokens: ReadOnly[int]
    prompt_tokens: ReadOnly[int]
    completion_tokens: ReadOnly[int]
    request_count: ReadOnly[int]


class _SeriesRow(TypedDict):
    bucket: ReadOnly[datetime]
    request_count: ReadOnly[int]
    prompt_tokens: ReadOnly[int]
    completion_tokens: ReadOnly[int]
    total_tokens: ReadOnly[int]
    cache_read_tokens: ReadOnly[int]
    spend_usd: ReadOnly[Decimal]


_DAILY_ROWS: Final = TypeAdapter(tuple[_DailyRow, ...])
_SCOPE_ROWS: Final = TypeAdapter(tuple[_ScopeRow, ...])
_TOTALS_ROWS: Final = TypeAdapter(tuple[_TotalsRow, ...])
_SERIES_ROWS: Final = TypeAdapter(tuple[_SeriesRow, ...])

_HISTORY_SQL: Final = """
SELECT bucket_date, request_count, prompt_tokens, completion_tokens,
       cache_read_tokens, spend_usd, breakdown
FROM "WITOS_FinOpsUsageDaily"
WHERE scope_type = $1 AND scope_id = $2
  AND bucket_date >= ($3::timestamptz AT TIME ZONE 'UTC')
  AND bucket_date <= ($4::timestamptz AT TIME ZONE 'UTC')
ORDER BY bucket_date
"""

# Ordered by spend so a run that has to stop early has already done the scopes
# whose numbers somebody is waiting on.
_ACTIVE_SCOPES_SQL: Final = """
SELECT scope_type, scope_id,
       MAX(scope_label) AS scope_label,
       COALESCE(SUM(spend_usd), 0)::numeric(18, 8) AS spend_usd
FROM "WITOS_FinOpsUsageDaily"
WHERE bucket_date >= ($1::timestamptz AT TIME ZONE 'UTC')
  AND scope_type = ANY($2::text[])
GROUP BY scope_type, scope_id
ORDER BY spend_usd DESC
"""

_TOTALS_SQL: Final = """
SELECT COALESCE(SUM(spend_usd), 0)::numeric(18, 8) AS spend_usd,
       COALESCE(SUM(total_tokens), 0)::bigint AS total_tokens,
       COALESCE(SUM(prompt_tokens), 0)::bigint AS prompt_tokens,
       COALESCE(SUM(completion_tokens), 0)::bigint AS completion_tokens,
       COALESCE(SUM(request_count), 0)::bigint AS request_count
FROM "WITOS_FinOpsUsageDaily"
WHERE scope_type = $1 AND scope_id = $2
  AND bucket_date >= ($3::timestamptz AT TIME ZONE 'UTC')
  AND bucket_date <= ($4::timestamptz AT TIME ZONE 'UTC')
"""

_TOTALS_BY_SCOPE_SQL: Final = """
SELECT scope_type, scope_id,
       COALESCE(SUM(spend_usd), 0)::numeric(18, 8) AS spend_usd,
       MAX(scope_label) AS scope_label
FROM "WITOS_FinOpsUsageDaily"
WHERE bucket_date >= ($1::timestamptz AT TIME ZONE 'UTC')
  AND bucket_date <= ($2::timestamptz AT TIME ZONE 'UTC')
  AND scope_type = ANY($3::text[])
GROUP BY scope_type, scope_id
"""

_DAILY_SERIES_SQL: Final = """
SELECT bucket_date AS bucket, request_count, prompt_tokens, completion_tokens,
       total_tokens, cache_read_tokens, spend_usd
FROM "WITOS_FinOpsUsageDaily"
WHERE scope_type = $1 AND scope_id = $2
  AND bucket_date >= ($3::timestamptz AT TIME ZONE 'UTC')
  AND bucket_date <= ($4::timestamptz AT TIME ZONE 'UTC')
ORDER BY bucket_date
"""

_HOURLY_SERIES_SQL: Final = """
SELECT bucket_start_utc AS bucket, request_count, prompt_tokens, completion_tokens,
       total_tokens, cache_read_tokens, spend_usd
FROM "WITOS_FinOpsUsageHourly"
WHERE scope_type = $1 AND scope_id = $2
  AND bucket_start_utc >= ($3::timestamptz AT TIME ZONE 'UTC')
  AND bucket_start_utc <= ($4::timestamptz AT TIME ZONE 'UTC')
ORDER BY bucket_start_utc
"""


@dataclass(frozen=True, slots=True)
class ActiveScope:
    scope: ScopeRef
    label: str | None
    recent_spend: Decimal


@dataclass(frozen=True, slots=True)
class ScopeTotals:
    """Consumption of every quota metric over one window, for runway's ``consumed``."""

    spend_usd: Decimal
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    request_count: int

    def of(self, metric: str) -> Decimal:
        match metric:
            case "usd" | "spend_usd":
                return self.spend_usd
            case "total_tokens":
                return Decimal(self.total_tokens)
            case "prompt_tokens":
                return Decimal(self.prompt_tokens)
            case "completion_tokens":
                return Decimal(self.completion_tokens)
            case "requests" | "request_count":
                return Decimal(self.request_count)
            case _:
                return Decimal(0)


@dataclass(frozen=True, slots=True)
class SeriesPoint:
    bucket: datetime
    request_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cache_read_tokens: int
    spend_usd: Decimal


def _mix_from(breakdown: Mapping[str, Mapping[str, Decimal]] | None) -> Mapping[str, float]:
    if not breakdown:
        return MappingProxyType({})
    return MappingProxyType({model: float(values.get("tokens", Decimal(0))) for model, values in breakdown.items()})


class FactReader:
    """Read-only access to the aggregated fact tables."""

    def __init__(self, executor: SqlExecutor) -> None:
        self._executor: Final = executor

    async def history(self, scope: ScopeRef, *, as_of: date, days: int = MAX_HISTORY_DAYS) -> UsageHistory | None:
        rows: Final = _DAILY_ROWS.validate_python(
            await self._executor.query(
                _HISTORY_SQL, scope.scope_type, scope.scope_id, as_of - timedelta(days=days - 1), as_of
            )
        )
        return build_history(scope, tuple(_as_daily(row) for row in rows), as_of=as_of)

    async def active_scopes(
        self, *, since: date, scope_types: Sequence[str] = DEFAULT_FORECAST_SCOPE_TYPES
    ) -> tuple[ActiveScope, ...]:
        scopes: Final = list(scope_types)  # mutable-ok: an array bind needs a list, not a tuple
        rows: Final = _SCOPE_ROWS.validate_python(await self._executor.query(_ACTIVE_SCOPES_SQL, since, scopes))
        return tuple(
            ActiveScope(
                scope=ScopeRef(scope_type=row["scope_type"], scope_id=row["scope_id"]),
                label=row["scope_label"],
                recent_spend=row["spend_usd"],
            )
            for row in rows
        )

    async def totals(self, scope: ScopeRef, *, start: date, end: date) -> ScopeTotals:
        rows: Final = _TOTALS_ROWS.validate_python(
            await self._executor.query(_TOTALS_SQL, scope.scope_type, scope.scope_id, start, end)
        )
        if not rows:
            return ScopeTotals(
                spend_usd=Decimal(0), total_tokens=0, prompt_tokens=0, completion_tokens=0, request_count=0
            )
        return ScopeTotals(
            spend_usd=rows[0]["spend_usd"],
            total_tokens=rows[0]["total_tokens"],
            prompt_tokens=rows[0]["prompt_tokens"],
            completion_tokens=rows[0]["completion_tokens"],
            request_count=rows[0]["request_count"],
        )

    async def spend_by_scope(
        self, *, start: date, end: date, scope_types: Sequence[str]
    ) -> Mapping[tuple[str, str], Decimal]:
        """Month-to-date actuals for a whole scope type in one statement, for the CFO export."""
        scopes: Final = list(scope_types)  # mutable-ok: an array bind needs a list, not a tuple
        rows: Final = _SCOPE_ROWS.validate_python(await self._executor.query(_TOTALS_BY_SCOPE_SQL, start, end, scopes))
        return MappingProxyType({(row["scope_type"], row["scope_id"]): row["spend_usd"] for row in rows})

    async def series(
        self, scope: ScopeRef, *, start: datetime, end: datetime, grain: str = "day"
    ) -> tuple[SeriesPoint, ...]:
        rows: Final = _SERIES_ROWS.validate_python(
            await self._executor.query(
                _DAILY_SERIES_SQL if grain == "day" else _HOURLY_SERIES_SQL,
                scope.scope_type,
                scope.scope_id,
                start,
                end,
            )
        )
        return tuple(
            SeriesPoint(
                bucket=row["bucket"],
                request_count=row["request_count"],
                prompt_tokens=row["prompt_tokens"],
                completion_tokens=row["completion_tokens"],
                total_tokens=row["total_tokens"],
                cache_read_tokens=row["cache_read_tokens"],
                spend_usd=row["spend_usd"],
            )
            for row in rows
        )


def _as_daily(row: _DailyRow) -> DailyUsage:
    return DailyUsage(
        day=row["bucket_date"].date(),
        request_count=float(row["request_count"]),
        prompt_tokens=float(row["prompt_tokens"]),
        completion_tokens=float(row["completion_tokens"]),
        cache_read_tokens=float(row["cache_read_tokens"]),
        spend_usd=row["spend_usd"],
        mix_tokens=_mix_from(row["breakdown"]),
    )
