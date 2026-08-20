"""The SQL the FinOps aggregator runs. Every statement here is a recompute, not an increment.

Blueprint §1.5 requires the spend writer's late rows to be corrected rather than
double counted. Both fact tables are therefore filled with
``INSERT ... SELECT ... ON CONFLICT DO UPDATE SET col = EXCLUDED.col``: a bucket
is recomputed from source and replaced wholesale, so running a cycle twice over
the same window produces byte-identical buckets and a row that lands after its
bucket was first written is simply included the next time that bucket is swept.
An increment could not offer either property.

Two grain-independent facts about the source shape drive the odd-looking bits:

``LiteLLM_SpendLogs.startTime`` is ``TIMESTAMP(3)`` holding UTC with no zone, so
every window bound is bound as ``timestamptz`` and converted with
``AT TIME ZONE 'UTC'`` the way the spend endpoints already do. Bucketing is
therefore pure UTC arithmetic and no local DST transition can shorten, lengthen
or duplicate a bucket.

Cache token counts are not columns. The spend writer nests them under
``metadata -> additional_usage_values``, so they are read back out of JSONB and
only when the stored value really is a number.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, TypeAlias

# Blueprint §1.4. `global` is the fleet roll-up; the rest mirror the entity
# hierarchy the proxy already attributes spend to.
ScopeType: TypeAlias = str

_HOURLY_COLUMNS: Final = (
    "request_count",
    "successful_requests",
    "failed_requests",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
    "spend_usd",
    "avg_latency_ms",
    "p95_latency_ms",
    "scope_label",
    "organization_id",
    "team_id",
    "model",
    "model_group",
    "provider",
    "breakdown",
)

_DAILY_COLUMNS: Final = (
    "request_count",
    "successful_requests",
    "failed_requests",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
    "spend_usd",
    "scope_label",
    "organization_id",
    "team_id",
    "model_group",
    "provider",
    "breakdown",
)


def _replace_all(columns: Sequence[str]) -> str:
    return ", ".join(f'"{column}" = EXCLUDED."{column}"' for column in columns)


def _unambiguous(expression: str) -> str:
    """Denormalize a dimension onto the bucket row only when the bucket agrees on it.

    A key belongs to one team, but a user can spend under several. Picking an
    arbitrary member with a bare MIN would print a confident wrong team on the
    row; collapsing to NULL when the bucket disagrees keeps the column honest and
    keeps recomputes deterministic.
    """
    return f"CASE WHEN COUNT(DISTINCT {expression}) = 1 THEN MIN({expression}) END"


def _cache_tokens(field: str) -> str:
    return (
        f"CASE WHEN jsonb_typeof(sl.metadata #> '{{additional_usage_values,{field}}}') = 'number' "
        f"THEN (sl.metadata #>> '{{additional_usage_values,{field}}}')::numeric::bigint ELSE 0 END"
    )


@dataclass(frozen=True, slots=True)
class ScopeSpec:
    """How one scope type is projected out of the spend log.

    Every field is a SQL fragment defined in this module and never derived from
    caller input; scope selection happens by looking a name up in ``SCOPE_SPECS``.
    """

    scope_type: ScopeType
    scope_id_sql: str
    extra_from_sql: str = ""
    extra_where_sql: str = ""
    label_sql: str = "NULL::text"
    model_sql: str = "NULL::text"
    model_group_sql: str = "NULL::text"
    provider_sql: str = "NULL::text"
    takes_tag_dimensions: bool = False


_TAG_ELEMENTS: Final = (
    "CROSS JOIN LATERAL jsonb_array_elements_text("
    "CASE WHEN jsonb_typeof(sl.request_tags) = 'array' THEN sl.request_tags ELSE '[]'::jsonb END"
    ") AS witos_tag(value)"
)

SCOPE_SPECS: Final[Mapping[ScopeType, ScopeSpec]] = MappingProxyType(
    {
        spec.scope_type: spec
        for spec in (
            ScopeSpec(scope_type="global", scope_id_sql="'global'"),
            ScopeSpec(
                scope_type="organization",
                scope_id_sql="NULLIF(sl.organization_id, '')",
                extra_where_sql="AND NULLIF(sl.organization_id, '') IS NOT NULL",
                label_sql=(
                    'SELECT o.organization_alias FROM "LiteLLM_OrganizationTable" o '
                    "WHERE o.organization_id = agg.scope_id"
                ),
            ),
            ScopeSpec(
                scope_type="team",
                scope_id_sql="NULLIF(sl.team_id, '')",
                extra_where_sql="AND NULLIF(sl.team_id, '') IS NOT NULL",
                label_sql='SELECT t.team_alias FROM "LiteLLM_TeamTable" t WHERE t.team_id = agg.scope_id',
            ),
            ScopeSpec(
                scope_type="user",
                scope_id_sql="NULLIF(sl.\"user\", '')",
                extra_where_sql="AND NULLIF(sl.\"user\", '') IS NOT NULL",
                label_sql='SELECT u.user_alias FROM "LiteLLM_UserTable" u WHERE u.user_id = agg.scope_id',
            ),
            ScopeSpec(
                scope_type="key",
                scope_id_sql="NULLIF(sl.api_key, '')",
                extra_where_sql="AND NULLIF(sl.api_key, '') IS NOT NULL",
                label_sql='SELECT v.key_alias FROM "LiteLLM_VerificationToken" v WHERE v.token = agg.scope_id',
            ),
            ScopeSpec(
                scope_type="end_user",
                scope_id_sql="NULLIF(sl.end_user, '')",
                extra_where_sql="AND NULLIF(sl.end_user, '') IS NOT NULL",
                label_sql='SELECT e.alias FROM "LiteLLM_EndUserTable" e WHERE e.user_id = agg.scope_id',
            ),
            ScopeSpec(
                scope_type="model",
                scope_id_sql="NULLIF(sl.model, '')",
                extra_where_sql="AND NULLIF(sl.model, '') IS NOT NULL",
                model_sql="MIN(scope_id)",
                provider_sql=_unambiguous("provider"),
            ),
            ScopeSpec(
                scope_type="model_group",
                scope_id_sql="NULLIF(sl.model_group, '')",
                extra_where_sql="AND NULLIF(sl.model_group, '') IS NOT NULL",
                model_group_sql="MIN(scope_id)",
                provider_sql=_unambiguous("provider"),
            ),
            ScopeSpec(
                scope_type="tag",
                scope_id_sql="witos_tag.value",
                extra_from_sql=_TAG_ELEMENTS,
                # Only `dimension:value` tags matching an admin-approved dimension become
                # facts. Aggregating every tag a caller can invent is the cardinality bomb
                # blueprint §1.4 forbids.
                extra_where_sql=(
                    "AND position(':' in witos_tag.value) > 0 AND split_part(witos_tag.value, ':', 1) = ANY($3::text[])"
                ),
                label_sql="SELECT split_part(agg.scope_id, ':', 2)",
                takes_tag_dimensions=True,
            ),
        )
    }
)


def hourly_upsert_sql(scope_type: ScopeType) -> str:
    """Recompute every hourly bucket for one scope type over the bound window."""
    spec: Final = SCOPE_SPECS[scope_type]
    return f"""
WITH src AS (
    SELECT
        date_trunc('hour', sl."startTime") AS bucket_start_utc,
        {spec.scope_id_sql} AS scope_id,
        NULLIF(sl.model, '') AS model,
        NULLIF(sl.model_group, '') AS model_group,
        NULLIF(sl.custom_llm_provider, '') AS provider,
        NULLIF(sl.organization_id, '') AS organization_id,
        NULLIF(sl.team_id, '') AS team_id,
        sl.status AS status,
        sl.spend AS spend,
        sl.prompt_tokens AS prompt_tokens,
        sl.completion_tokens AS completion_tokens,
        sl.total_tokens AS total_tokens,
        sl.request_duration_ms AS request_duration_ms,
        {_cache_tokens("cache_read_input_tokens")} AS cache_read_tokens,
        {_cache_tokens("cache_creation_input_tokens")} AS cache_creation_tokens
    FROM "LiteLLM_SpendLogs" sl
    {spec.extra_from_sql}
    WHERE sl."startTime" >= ($1::timestamptz AT TIME ZONE 'UTC')
      AND sl."startTime" <  ($2::timestamptz AT TIME ZONE 'UTC')
      {spec.extra_where_sql}
),
mix AS (
    SELECT
        bucket_start_utc,
        scope_id,
        COALESCE(model_group, model, 'unknown') AS mix_key,
        SUM(total_tokens)::bigint AS tokens,
        SUM(spend)::numeric(18, 8) AS spend,
        COUNT(*)::bigint AS requests
    FROM src
    WHERE scope_id IS NOT NULL
    GROUP BY 1, 2, 3
),
mix_json AS (
    SELECT
        bucket_start_utc,
        scope_id,
        jsonb_object_agg(
            mix_key,
            jsonb_build_object('tokens', tokens, 'spend', spend, 'requests', requests)
        ) AS breakdown
    FROM mix
    GROUP BY 1, 2
),
agg AS (
    SELECT
        bucket_start_utc,
        scope_id,
        COUNT(*)::int AS request_count,
        COUNT(*) FILTER (WHERE status = 'success')::int AS successful_requests,
        COUNT(*) FILTER (WHERE status = 'failure')::int AS failed_requests,
        COALESCE(SUM(prompt_tokens), 0)::bigint AS prompt_tokens,
        COALESCE(SUM(completion_tokens), 0)::bigint AS completion_tokens,
        COALESCE(SUM(total_tokens), 0)::bigint AS total_tokens,
        COALESCE(SUM(cache_read_tokens), 0)::bigint AS cache_read_tokens,
        COALESCE(SUM(cache_creation_tokens), 0)::bigint AS cache_creation_tokens,
        COALESCE(SUM(spend), 0)::numeric(18, 8) AS spend_usd,
        AVG(request_duration_ms)::int AS avg_latency_ms,
        percentile_cont(0.95) WITHIN GROUP (ORDER BY request_duration_ms)::int AS p95_latency_ms,
        {_unambiguous("organization_id")} AS organization_id,
        {_unambiguous("team_id")} AS team_id,
        {spec.model_sql} AS model,
        {spec.model_group_sql} AS model_group,
        {spec.provider_sql} AS provider
    FROM src
    WHERE scope_id IS NOT NULL
    GROUP BY 1, 2
)
INSERT INTO "WITOS_FinOpsUsageHourly" (
    id, bucket_start_utc, scope_type, scope_id, scope_label, organization_id, team_id,
    model, model_group, provider, call_type,
    request_count, successful_requests, failed_requests,
    prompt_tokens, completion_tokens, total_tokens,
    cache_read_tokens, cache_creation_tokens, spend_usd,
    avg_latency_ms, p95_latency_ms, breakdown, created_at, updated_at
)
SELECT
    gen_random_uuid()::text,
    agg.bucket_start_utc,
    '{spec.scope_type}',
    agg.scope_id,
    ({spec.label_sql}),
    agg.organization_id,
    agg.team_id,
    agg.model,
    agg.model_group,
    agg.provider,
    NULL::text,
    agg.request_count,
    agg.successful_requests,
    agg.failed_requests,
    agg.prompt_tokens,
    agg.completion_tokens,
    agg.total_tokens,
    agg.cache_read_tokens,
    agg.cache_creation_tokens,
    agg.spend_usd,
    agg.avg_latency_ms,
    agg.p95_latency_ms,
    mix_json.breakdown,
    (now() AT TIME ZONE 'UTC'),
    (now() AT TIME ZONE 'UTC')
FROM agg
LEFT JOIN mix_json USING (bucket_start_utc, scope_id)
ON CONFLICT (bucket_start_utc, scope_type, scope_id) DO UPDATE
SET {_replace_all(_HOURLY_COLUMNS)},
    updated_at = (now() AT TIME ZONE 'UTC')
"""


DAILY_UPSERT_SQL: Final = f"""
WITH hourly AS (
    SELECT *
    FROM "WITOS_FinOpsUsageHourly"
    WHERE bucket_start_utc >= ($1::timestamptz AT TIME ZONE 'UTC')
      AND bucket_start_utc <  ($2::timestamptz AT TIME ZONE 'UTC')
),
mix AS (
    SELECT
        date_trunc('day', h.bucket_start_utc) AS bucket_date,
        h.scope_type,
        h.scope_id,
        kv.key AS mix_key,
        SUM((kv.value ->> 'tokens')::bigint) AS tokens,
        SUM((kv.value ->> 'spend')::numeric) AS spend,
        SUM((kv.value ->> 'requests')::bigint) AS requests
    FROM hourly h
    CROSS JOIN LATERAL jsonb_each(COALESCE(h.breakdown, '{{}}'::jsonb)) AS kv
    GROUP BY 1, 2, 3, 4
),
mix_json AS (
    SELECT
        bucket_date,
        scope_type,
        scope_id,
        jsonb_object_agg(
            mix_key,
            jsonb_build_object('tokens', tokens, 'spend', spend, 'requests', requests)
        ) AS breakdown
    FROM mix
    GROUP BY 1, 2, 3
),
agg AS (
    SELECT
        date_trunc('day', h.bucket_start_utc) AS bucket_date,
        h.scope_type,
        h.scope_id,
        SUM(h.request_count)::int AS request_count,
        SUM(h.successful_requests)::int AS successful_requests,
        SUM(h.failed_requests)::int AS failed_requests,
        SUM(h.prompt_tokens)::bigint AS prompt_tokens,
        SUM(h.completion_tokens)::bigint AS completion_tokens,
        SUM(h.total_tokens)::bigint AS total_tokens,
        SUM(h.cache_read_tokens)::bigint AS cache_read_tokens,
        SUM(h.cache_creation_tokens)::bigint AS cache_creation_tokens,
        SUM(h.spend_usd)::numeric(18, 8) AS spend_usd,
        {_unambiguous("h.scope_label")} AS scope_label,
        {_unambiguous("h.organization_id")} AS organization_id,
        {_unambiguous("h.team_id")} AS team_id,
        {_unambiguous("h.model_group")} AS model_group,
        {_unambiguous("h.provider")} AS provider
    FROM hourly h
    GROUP BY 1, 2, 3
)
INSERT INTO "WITOS_FinOpsUsageDaily" (
    id, bucket_date, scope_type, scope_id, scope_label, organization_id, team_id,
    model_group, provider,
    request_count, successful_requests, failed_requests,
    prompt_tokens, completion_tokens, total_tokens,
    cache_read_tokens, cache_creation_tokens, spend_usd,
    breakdown, created_at, updated_at
)
SELECT
    gen_random_uuid()::text,
    agg.bucket_date,
    agg.scope_type,
    agg.scope_id,
    agg.scope_label,
    agg.organization_id,
    agg.team_id,
    agg.model_group,
    agg.provider,
    agg.request_count,
    agg.successful_requests,
    agg.failed_requests,
    agg.prompt_tokens,
    agg.completion_tokens,
    agg.total_tokens,
    agg.cache_read_tokens,
    agg.cache_creation_tokens,
    agg.spend_usd,
    mix_json.breakdown,
    (now() AT TIME ZONE 'UTC'),
    (now() AT TIME ZONE 'UTC')
FROM agg
LEFT JOIN mix_json USING (bucket_date, scope_type, scope_id)
ON CONFLICT (bucket_date, scope_type, scope_id) DO UPDATE
SET {_replace_all(_DAILY_COLUMNS)},
    updated_at = (now() AT TIME ZONE 'UTC')
"""


# One aggregate row per model in the window. Pricing coverage is a property of the
# model, so this stays a handful of rows however many requests the window holds:
# the aggregator never loops over spend rows in Python.
PRICING_COVERAGE_SQL: Final = """
SELECT
    NULLIF(sl.model, '') AS model,
    COUNT(*)::bigint AS request_count,
    COUNT(*) FILTER (WHERE sl.total_tokens > 0 AND sl.spend = 0)::bigint AS zero_priced_requests
FROM "LiteLLM_SpendLogs" sl
WHERE sl."startTime" >= ($1::timestamptz AT TIME ZONE 'UTC')
  AND sl."startTime" <  ($2::timestamptz AT TIME ZONE 'UTC')
GROUP BY 1
"""
