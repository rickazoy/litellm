"""Quota CRUD (blueprint §1.11), with the mirrored rows held read-only.

A quota whose ``source_type`` is ``litellm_budget`` is a projection of a LiteLLM
budget that Phase 1's mirror job rewrites on a schedule. Editing one here would
survive until the next mirror run and then silently revert, so writes against
those rows are refused rather than accepted and lost. The budget is edited in
LiteLLM and arrives here on the next pass.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Final, Literal, TypeAlias, TypedDict

from pydantic import TypeAdapter
from typing_extensions import ReadOnly

from litellm.proxy.witos.finops.history import ScopeRef
from litellm.proxy.witos.finops.quotas import LITELLM_BUDGET_SOURCE
from litellm.proxy.witos.finops.runway import Quota
from litellm.proxy.witos.shared.sql import SqlExecutor

QUOTA_METRICS: Final = ("usd", "total_tokens", "prompt_tokens", "completion_tokens", "requests")
QUOTA_PERIODS: Final = ("day", "week", "month", "quarter", "year", "contract", "custom")

WriteRefusal: TypeAlias = Literal["mirrored_quota_is_read_only", "not_found"]


class _QuotaRow(TypedDict):
    quota_id: ReadOnly[str]
    scope_type: ReadOnly[str]
    scope_id: ReadOnly[str]
    metric: ReadOnly[str]
    limit_value: ReadOnly[Decimal]
    period: ReadOnly[str]
    period_start: ReadOnly[datetime]
    period_end: ReadOnly[datetime | None]
    reset_at: ReadOnly[datetime | None]
    source_type: ReadOnly[str]
    soft_threshold_pct: ReadOnly[Decimal | None]


_QUOTA_ROWS: Final = TypeAdapter(tuple[_QuotaRow, ...])

_COLUMNS: Final = (
    "quota_id, scope_type, scope_id, metric, limit_value, period, "
    "period_start, period_end, reset_at, source_type, soft_threshold_pct"
)

_BY_SCOPE_SQL: Final = f"""
SELECT {_COLUMNS} FROM "WITOS_FinOpsQuota"
WHERE scope_type = $1 AND scope_id = $2
ORDER BY metric, period
"""

_ALL_SQL: Final = f'SELECT {_COLUMNS} FROM "WITOS_FinOpsQuota" ORDER BY scope_type, scope_id, metric'

_BY_ID_SQL: Final = f'SELECT {_COLUMNS} FROM "WITOS_FinOpsQuota" WHERE quota_id = $1'

_INSERT_SQL: Final = """
INSERT INTO "WITOS_FinOpsQuota" (
    quota_id, scope_type, scope_id, metric, limit_value, period, period_start, period_end,
    reset_at, source_type, soft_threshold_pct, created_at, updated_at
)
VALUES (
    gen_random_uuid()::text, $1, $2, $3, $4::numeric, $5,
    ($6::timestamptz AT TIME ZONE 'UTC'), ($7::timestamptz AT TIME ZONE 'UTC'),
    ($8::timestamptz AT TIME ZONE 'UTC'), $9, $10::numeric,
    (now() AT TIME ZONE 'UTC'), (now() AT TIME ZONE 'UTC')
)
RETURNING quota_id, scope_type, scope_id, metric, limit_value, period,
          period_start, period_end, reset_at, source_type, soft_threshold_pct
"""

_UPDATE_SQL: Final = """
UPDATE "WITOS_FinOpsQuota"
SET limit_value = COALESCE($2::numeric, limit_value),
    period = COALESCE($3, period),
    period_end = COALESCE(($4::timestamptz AT TIME ZONE 'UTC'), period_end),
    reset_at = COALESCE(($5::timestamptz AT TIME ZONE 'UTC'), reset_at),
    soft_threshold_pct = COALESCE($6::numeric, soft_threshold_pct),
    updated_at = (now() AT TIME ZONE 'UTC')
WHERE quota_id = $1 AND source_type <> '{source}'
RETURNING quota_id, scope_type, scope_id, metric, limit_value, period,
          period_start, period_end, reset_at, source_type, soft_threshold_pct
""".replace("{source}", LITELLM_BUDGET_SOURCE)

_DELETE_SQL: Final = f"""
DELETE FROM "WITOS_FinOpsQuota" WHERE quota_id = $1 AND source_type <> '{LITELLM_BUDGET_SOURCE}'
"""


@dataclass(frozen=True, slots=True)
class QuotaDraft:
    """A caller's proposed quota, already validated by the API layer."""

    scope: ScopeRef
    metric: str
    limit_value: Decimal
    period: str
    period_start: datetime
    period_end: datetime | None = None
    reset_at: datetime | None = None
    soft_threshold_pct: Decimal | None = None


def _as_quota(row: _QuotaRow) -> Quota:
    return Quota(
        quota_id=row["quota_id"],
        scope=ScopeRef(scope_type=row["scope_type"], scope_id=row["scope_id"]),
        metric=row["metric"],
        limit_value=row["limit_value"],
        period=row["period"],
        period_start=row["period_start"],
        period_end=row["period_end"],
        reset_at=row["reset_at"],
        source_type=row["source_type"],
        soft_threshold_pct=row["soft_threshold_pct"],
    )


class QuotaStore:
    """Reads every quota; writes only the ones LiteLLM does not own."""

    def __init__(self, executor: SqlExecutor) -> None:
        self._executor: Final = executor

    async def for_scope(self, scope: ScopeRef) -> tuple[Quota, ...]:
        rows: Final = _QUOTA_ROWS.validate_python(
            await self._executor.query(_BY_SCOPE_SQL, scope.scope_type, scope.scope_id)
        )
        return tuple(_as_quota(row) for row in rows)

    async def all(self) -> tuple[Quota, ...]:
        rows: Final = _QUOTA_ROWS.validate_python(await self._executor.query(_ALL_SQL))
        return tuple(_as_quota(row) for row in rows)

    async def by_id(self, quota_id: str) -> Quota | None:
        rows: Final = _QUOTA_ROWS.validate_python(await self._executor.query(_BY_ID_SQL, quota_id))
        return _as_quota(rows[0]) if rows else None

    async def create(self, draft: QuotaDraft) -> Quota:
        rows: Final = _QUOTA_ROWS.validate_python(
            await self._executor.query(
                _INSERT_SQL,
                draft.scope.scope_type,
                draft.scope.scope_id,
                draft.metric,
                draft.limit_value,
                draft.period,
                draft.period_start,
                draft.period_end,
                draft.reset_at,
                "manual",
                draft.soft_threshold_pct,
            )
        )
        return _as_quota(rows[0])

    async def update(
        self,
        quota_id: str,
        *,
        limit_value: Decimal | None = None,
        period: str | None = None,
        period_end: datetime | None = None,
        reset_at: datetime | None = None,
        soft_threshold_pct: Decimal | None = None,
    ) -> Quota | WriteRefusal:
        rows: Final = _QUOTA_ROWS.validate_python(
            await self._executor.query(
                _UPDATE_SQL, quota_id, limit_value, period, period_end, reset_at, soft_threshold_pct
            )
        )
        if rows:
            return _as_quota(rows[0])
        return "mirrored_quota_is_read_only" if await self.by_id(quota_id) else "not_found"

    async def delete(self, quota_id: str) -> Literal["deleted"] | WriteRefusal:
        deleted: Final = await self._executor.execute(_DELETE_SQL, quota_id)
        if deleted:
            return "deleted"
        return "mirrored_quota_is_read_only" if await self.by_id(quota_id) else "not_found"


def metrics_for(quotas: Sequence[Quota]) -> frozenset[str]:
    return frozenset(quota.metric for quota in quotas)
