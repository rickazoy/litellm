"""Mirror LiteLLM budgets into ``WITOS_FinOpsQuota`` as entitlements.

Blueprint §1.4 draws the line this module exists to hold: a quota is an
*entitlement*, an amount that can be consumed and then is gone. LiteLLM's
``tpm_limit`` and ``rpm_limit`` are *rate limits*: a 100M TPM key has no
100M-token allowance, it has a throttle. Mirroring them here would make every
downstream runway sentence ("exhausted on Aug 27") a fabrication, so they are
deliberately not mirrored, and neither is ``max_parallel_requests``. Only
``max_budget`` crosses over, as ``metric = 'usd'``.

Budgets reach an entity two ways in this fork, and both are mirrored:

  * through ``LiteLLM_BudgetTable`` via a ``budget_id`` foreign key
    (organization, key, end user, tag)
  * inline on the entity row itself (team, user, and keys with no ``budget_id``)

Teams and users only ever carry inline budgets, so a mirror that read
``LiteLLM_BudgetTable`` alone would silently omit the two scopes a CFO cares
about most. ``source_type`` stays ``litellm_budget`` for both, because both are
LiteLLM-owned and read-only downstream: an operator edits the budget in LiteLLM
and the next mirror run carries it across.

Rows are keyed by ``(source_type, scope_type, scope_id, metric, period)``, so a
mirror run is an idempotent upsert and manual or contract quotas on the same
scope are never touched.
"""

from dataclasses import dataclass
from typing import Final, TypeAlias

from litellm._logging import verbose_proxy_logger
from litellm.proxy.witos.shared.scheduler import JobLeaseManager, LeaseHeld
from litellm.proxy.witos.shared.sql import SqlExecutor

QUOTA_MIRROR_JOB_NAME: Final = "witos_finops_quota_mirror"
LITELLM_BUDGET_SOURCE: Final = "litellm_budget"

# LiteLLM stores a budget's length as a duration string and the *end* of the
# current cycle as `budget_reset_at`. The period label and start are derived from
# those; anything this CASE does not recognise stays 'custom' with a NULL start
# rather than inventing a cycle.
_PERIOD_LABEL: Final = """
CASE lower(coalesce(src.budget_duration, ''))
    WHEN '' THEN 'custom'
    WHEN 'hourly' THEN 'day'
    WHEN 'daily' THEN 'day'
    WHEN '24h' THEN 'day'
    WHEN '1d' THEN 'day'
    WHEN 'weekly' THEN 'week'
    WHEN '7d' THEN 'week'
    WHEN 'monthly' THEN 'month'
    WHEN '30d' THEN 'month'
    WHEN '1mo' THEN 'month'
    WHEN '31d' THEN 'month'
    WHEN '3mo' THEN 'quarter'
    WHEN '90d' THEN 'quarter'
    WHEN '12mo' THEN 'year'
    WHEN '365d' THEN 'year'
    WHEN '1y' THEN 'year'
    ELSE 'custom'
END
"""

_PERIOD_LENGTH: Final = """
CASE lower(coalesce(src.budget_duration, ''))
    WHEN 'hourly' THEN interval '1 hour'
    WHEN 'daily' THEN interval '1 day'
    WHEN '24h' THEN interval '1 day'
    WHEN '1d' THEN interval '1 day'
    WHEN 'weekly' THEN interval '7 days'
    WHEN '7d' THEN interval '7 days'
    WHEN 'monthly' THEN interval '30 days'
    WHEN '30d' THEN interval '30 days'
    WHEN '1mo' THEN interval '1 month'
    WHEN '31d' THEN interval '31 days'
    WHEN '3mo' THEN interval '3 months'
    WHEN '90d' THEN interval '90 days'
    WHEN '12mo' THEN interval '1 year'
    WHEN '365d' THEN interval '365 days'
    WHEN '1y' THEN interval '1 year'
    ELSE NULL
END
"""

# `soft_budget` is an advisory threshold on the same allowance, so it is carried
# as a percentage of the hard limit rather than as a second quota row.
_SOFT_THRESHOLD: Final = """
CASE
    WHEN src.soft_budget IS NULL OR src.max_budget IS NULL OR src.max_budget <= 0 THEN NULL
    ELSE LEAST(100, GREATEST(0, (src.soft_budget / src.max_budget) * 100))::numeric(5, 2)
END
"""

_SOURCES: Final = """
SELECT 'organization' AS scope_type, o.organization_id AS scope_id, b.budget_id AS source_id,
       b.max_budget, b.soft_budget, b.budget_duration, b.budget_reset_at
FROM "LiteLLM_OrganizationTable" o
JOIN "LiteLLM_BudgetTable" b ON b.budget_id = o.budget_id

UNION ALL
SELECT 'end_user', e.user_id, b.budget_id, b.max_budget, b.soft_budget, b.budget_duration, b.budget_reset_at
FROM "LiteLLM_EndUserTable" e
JOIN "LiteLLM_BudgetTable" b ON b.budget_id = e.budget_id

UNION ALL
SELECT 'tag', t.tag_name, b.budget_id, b.max_budget, b.soft_budget, b.budget_duration, b.budget_reset_at
FROM "LiteLLM_TagTable" t
JOIN "LiteLLM_BudgetTable" b ON b.budget_id = t.budget_id

UNION ALL
SELECT 'key', v.token, b.budget_id, b.max_budget, b.soft_budget, b.budget_duration, b.budget_reset_at
FROM "LiteLLM_VerificationToken" v
JOIN "LiteLLM_BudgetTable" b ON b.budget_id = v.budget_id

UNION ALL
SELECT 'key', v.token, NULL, v.max_budget, NULL, v.budget_duration, v.budget_reset_at
FROM "LiteLLM_VerificationToken" v
WHERE v.budget_id IS NULL

UNION ALL
SELECT 'team', t.team_id, NULL, t.max_budget, t.soft_budget, t.budget_duration, t.budget_reset_at
FROM "LiteLLM_TeamTable" t

UNION ALL
SELECT 'user', u.user_id, NULL, u.max_budget, NULL, u.budget_duration, u.budget_reset_at
FROM "LiteLLM_UserTable" u
"""

MIRROR_SQL: Final = f"""
WITH src AS (
    SELECT * FROM ({_SOURCES}) AS budgets
    WHERE budgets.max_budget IS NOT NULL AND budgets.max_budget > 0
),
resolved AS (
    SELECT
        src.scope_type,
        src.scope_id,
        src.source_id,
        src.max_budget::numeric(24, 4) AS limit_value,
        ({_PERIOD_LABEL}) AS period,
        src.budget_reset_at AS reset_at,
        ({_PERIOD_LENGTH}) AS period_length,
        ({_SOFT_THRESHOLD}) AS soft_threshold_pct
    FROM src
)
INSERT INTO "WITOS_FinOpsQuota" (
    quota_id, scope_type, scope_id, metric, limit_value, period,
    period_start, period_end, reset_at, source_type, source_id,
    soft_threshold_pct, created_at, updated_at
)
SELECT
    gen_random_uuid()::text,
    resolved.scope_type,
    resolved.scope_id,
    'usd',
    resolved.limit_value,
    resolved.period,
    COALESCE(resolved.reset_at - resolved.period_length, resolved.reset_at, (now() AT TIME ZONE 'UTC')),
    resolved.reset_at,
    resolved.reset_at,
    '{LITELLM_BUDGET_SOURCE}',
    resolved.source_id,
    resolved.soft_threshold_pct,
    (now() AT TIME ZONE 'UTC'),
    (now() AT TIME ZONE 'UTC')
FROM resolved
ON CONFLICT (source_type, scope_type, scope_id, metric, period) DO UPDATE
SET limit_value = EXCLUDED.limit_value,
    period_start = EXCLUDED.period_start,
    period_end = EXCLUDED.period_end,
    reset_at = EXCLUDED.reset_at,
    source_id = EXCLUDED.source_id,
    soft_threshold_pct = EXCLUDED.soft_threshold_pct,
    updated_at = (now() AT TIME ZONE 'UTC')
"""

# A budget that was deleted or zeroed in LiteLLM must stop being an entitlement.
# Only mirrored rows are ever removed; manual and contract quotas are untouched.
PRUNE_SQL: Final = f"""
DELETE FROM "WITOS_FinOpsQuota" q
WHERE q.source_type = '{LITELLM_BUDGET_SOURCE}'
  AND NOT EXISTS (
      SELECT 1 FROM ({_SOURCES}) AS src
      WHERE src.scope_type = q.scope_type
        AND src.scope_id = q.scope_id
        AND src.max_budget IS NOT NULL
        AND src.max_budget > 0
        AND ({_PERIOD_LABEL}) = q.period
  )
"""


@dataclass(frozen=True, slots=True)
class MirrorCompleted:
    mirrored: int
    pruned: int


@dataclass(frozen=True, slots=True)
class MirrorSkipped:
    held_by: str


MirrorOutcome: TypeAlias = MirrorCompleted | MirrorSkipped


class QuotaMirror:
    """Keeps ``WITOS_FinOpsQuota`` in step with the budgets LiteLLM owns."""

    def __init__(self, executor: SqlExecutor, *, lease_manager: JobLeaseManager | None = None) -> None:
        self._executor: Final = executor
        self._lease_manager: Final = lease_manager or JobLeaseManager(executor)

    async def run(self) -> MirrorOutcome:
        async with self._lease_manager.hold(QUOTA_MIRROR_JOB_NAME) as lease:
            if isinstance(lease, LeaseHeld):
                return MirrorSkipped(held_by=lease.held_by)
            mirrored: Final = await self._executor.execute(MIRROR_SQL)
            pruned: Final = await self._executor.execute(PRUNE_SQL)
            verbose_proxy_logger.info(
                "WIT OS FinOps mirrored %d LiteLLM budgets as quotas, pruned %d", mirrored, pruned
            )
            return MirrorCompleted(mirrored=mirrored, pruned=pruned)
