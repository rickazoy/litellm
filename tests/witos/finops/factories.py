"""Row builders for the WIT OS suite.

Spend logs are written with the same column set ``DBSpendUpdateWriter`` uses,
including the cache-token nesting under ``metadata.additional_usage_values``,
so a test that passes here is exercising the shape production actually stores.
"""

import json
from datetime import datetime, timezone
from typing import Any

import psycopg

INSERT_SPEND_LOG = """
INSERT INTO "LiteLLM_SpendLogs" (
    request_id, call_type, api_key, spend, total_tokens, prompt_tokens, completion_tokens,
    "startTime", "endTime", request_duration_ms, model, model_group, custom_llm_provider,
    "user", metadata, request_tags, team_id, organization_id, end_user, status
)
VALUES (%(request_id)s, %(call_type)s, %(api_key)s, %(spend)s, %(total_tokens)s, %(prompt_tokens)s,
        %(completion_tokens)s, %(start_time)s, %(end_time)s, %(request_duration_ms)s, %(model)s,
        %(model_group)s, %(provider)s, %(user)s, %(metadata)s, %(request_tags)s, %(team_id)s,
        %(organization_id)s, %(end_user)s, %(status)s)
"""


def _naive_utc(moment: datetime | None) -> datetime | None:
    """LiteLLM stores naive UTC in its TIMESTAMP(3) columns; write what production writes."""
    return None if moment is None else moment.astimezone(timezone.utc).replace(tzinfo=None)


async def insert_spend_log(
    connection: psycopg.AsyncConnection,
    *,
    request_id: str,
    start_time: datetime,
    spend: float = 1.0,
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    total_tokens: int | None = None,
    cache_read_tokens: int | None = None,
    cache_creation_tokens: int | None = None,
    model: str = "gpt-4o",
    model_group: str = "gpt-4o",
    provider: str = "openai",
    api_key: str = "hash-key-1",
    user: str = "user-1",
    team_id: str | None = "team-1",
    organization_id: str | None = "org-1",
    end_user: str | None = "customer-1",
    request_tags: list[str] | None = None,
    status: str | None = "success",
    call_type: str = "acompletion",
    request_duration_ms: int | None = 250,
) -> None:
    usage: dict[str, Any] = {}
    if cache_read_tokens is not None:
        usage["cache_read_input_tokens"] = cache_read_tokens
    if cache_creation_tokens is not None:
        usage["cache_creation_input_tokens"] = cache_creation_tokens
    await connection.execute(
        INSERT_SPEND_LOG,
        {
            "request_id": request_id,
            "call_type": call_type,
            "api_key": api_key,
            "spend": spend,
            "total_tokens": total_tokens if total_tokens is not None else prompt_tokens + completion_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "start_time": _naive_utc(start_time),
            "end_time": _naive_utc(start_time),
            "request_duration_ms": request_duration_ms,
            "model": model,
            "model_group": model_group,
            "provider": provider,
            "user": user,
            "metadata": json.dumps({"additional_usage_values": usage}),
            "request_tags": json.dumps(request_tags or []),
            "team_id": team_id,
            "organization_id": organization_id,
            "end_user": end_user,
            "status": status,
        },
    )


async def insert_team(
    connection: psycopg.AsyncConnection,
    *,
    team_id: str,
    alias: str | None = None,
    max_budget: float | None = None,
    soft_budget: float | None = None,
    budget_duration: str | None = None,
    budget_reset_at: datetime | None = None,
    tpm_limit: int | None = None,
) -> None:
    await connection.execute(
        """
        INSERT INTO "LiteLLM_TeamTable" (team_id, team_alias, max_budget, soft_budget,
                                         budget_duration, budget_reset_at, tpm_limit, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, now())
        """,
        (team_id, alias, max_budget, soft_budget, budget_duration, _naive_utc(budget_reset_at), tpm_limit),
    )


async def insert_user(
    connection: psycopg.AsyncConnection,
    *,
    user_id: str,
    alias: str | None = None,
    max_budget: float | None = None,
    tpm_limit: int | None = None,
) -> None:
    await connection.execute(
        """
        INSERT INTO "LiteLLM_UserTable" (user_id, user_alias, max_budget, tpm_limit, updated_at)
        VALUES (%s, %s, %s, %s, now())
        """,
        (user_id, alias, max_budget, tpm_limit),
    )


async def insert_budget(
    connection: psycopg.AsyncConnection,
    *,
    budget_id: str,
    max_budget: float | None = None,
    soft_budget: float | None = None,
    budget_duration: str | None = None,
    budget_reset_at: datetime | None = None,
    tpm_limit: int | None = None,
) -> None:
    await connection.execute(
        """
        INSERT INTO "LiteLLM_BudgetTable" (budget_id, max_budget, soft_budget, budget_duration,
                                           budget_reset_at, tpm_limit, created_by, updated_by, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, 'tests', 'tests', now())
        """,
        (budget_id, max_budget, soft_budget, budget_duration, _naive_utc(budget_reset_at), tpm_limit),
    )


async def insert_organization(
    connection: psycopg.AsyncConnection, *, organization_id: str, budget_id: str, alias: str = "Org"
) -> None:
    await connection.execute(
        """
        INSERT INTO "LiteLLM_OrganizationTable" (organization_id, organization_alias, budget_id,
                                                 created_by, updated_by, updated_at)
        VALUES (%s, %s, %s, 'tests', 'tests', now())
        """,
        (organization_id, alias, budget_id),
    )


async def fetch_hourly(connection: psycopg.AsyncConnection, scope_type: str, scope_id: str) -> list[dict]:
    from psycopg.rows import dict_row

    async with connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(
            """
            SELECT * FROM "WITOS_FinOpsUsageHourly"
            WHERE scope_type = %s AND scope_id = %s ORDER BY bucket_start_utc
            """,
            (scope_type, scope_id),
        )
        return await cursor.fetchall()


async def fetch_daily(connection: psycopg.AsyncConnection, scope_type: str, scope_id: str) -> list[dict]:
    from psycopg.rows import dict_row

    async with connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(
            """
            SELECT * FROM "WITOS_FinOpsUsageDaily"
            WHERE scope_type = %s AND scope_id = %s ORDER BY bucket_date
            """,
            (scope_type, scope_id),
        )
        return await cursor.fetchall()


async def fetch_quotas(connection: psycopg.AsyncConnection) -> list[dict]:
    from psycopg.rows import dict_row

    async with connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute('SELECT * FROM "WITOS_FinOpsQuota" ORDER BY scope_type, scope_id, period')
        return await cursor.fetchall()
