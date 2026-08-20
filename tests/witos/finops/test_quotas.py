"""Mirroring LiteLLM budgets as entitlements, and the rate limits that must not cross."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from litellm.proxy.witos.finops.quotas import QuotaMirror
from tests.witos.finops.factories import (
    fetch_quotas,
    insert_budget,
    insert_organization,
    insert_team,
    insert_user,
)

RESET_AT = datetime(2026, 9, 1, tzinfo=timezone.utc)


async def test_a_tpm_limit_never_becomes_a_quota(connection, executor):
    """A rate limit is not an allowance. Mirroring one would make every runway date fiction."""
    await insert_team(connection, team_id="throttled", tpm_limit=100_000_000, max_budget=None)
    await insert_user(connection, user_id="also-throttled", tpm_limit=50_000, max_budget=None)
    await insert_budget(connection, budget_id="b1", max_budget=None, tpm_limit=9_000_000)
    await insert_organization(connection, organization_id="org-1", budget_id="b1")

    await QuotaMirror(executor).run()

    assert await fetch_quotas(connection) == []


async def test_a_team_budget_is_mirrored_as_a_usd_entitlement(connection, executor):
    await insert_team(
        connection,
        team_id="team-1",
        max_budget=1000.0,
        soft_budget=800.0,
        budget_duration="30d",
        budget_reset_at=RESET_AT,
    )

    await QuotaMirror(executor).run()

    quota = (await fetch_quotas(connection))[0]
    assert quota["scope_type"] == "team"
    assert quota["metric"] == "usd"
    assert quota["limit_value"] == Decimal("1000.0000")
    assert quota["period"] == "month"
    assert quota["source_type"] == "litellm_budget"
    assert quota["reset_at"] == RESET_AT.replace(tzinfo=None)
    assert quota["period_start"] == (RESET_AT - timedelta(days=30)).replace(tzinfo=None)
    assert quota["soft_threshold_pct"] == Decimal("80.00")


async def test_budgets_attached_through_the_budget_table_are_mirrored(connection, executor):
    await insert_budget(connection, budget_id="b1", max_budget=5000.0, budget_duration="1mo", budget_reset_at=RESET_AT)
    await insert_organization(connection, organization_id="org-1", budget_id="b1")

    await QuotaMirror(executor).run()

    quota = (await fetch_quotas(connection))[0]
    assert (quota["scope_type"], quota["scope_id"]) == ("organization", "org-1")
    assert quota["source_id"] == "b1"
    assert quota["limit_value"] == Decimal("5000.0000")


async def test_inline_and_attached_budgets_coexist(connection, executor):
    """Teams and users only ever carry inline budgets; reading the budget table alone would drop them."""
    await insert_team(connection, team_id="team-1", max_budget=100.0, budget_duration="7d", budget_reset_at=RESET_AT)
    await insert_user(connection, user_id="u1", max_budget=25.0)
    await insert_budget(connection, budget_id="b1", max_budget=900.0, budget_duration="1d", budget_reset_at=RESET_AT)
    await insert_organization(connection, organization_id="org-1", budget_id="b1")

    await QuotaMirror(executor).run()

    quotas = {(row["scope_type"], row["scope_id"]): row for row in await fetch_quotas(connection)}
    assert set(quotas) == {("team", "team-1"), ("user", "u1"), ("organization", "org-1")}
    assert quotas[("team", "team-1")]["period"] == "week"
    assert quotas[("user", "u1")]["period"] == "custom"
    assert quotas[("organization", "org-1")]["period"] == "day"


async def test_a_budget_with_no_duration_has_no_reset(connection, executor):
    await insert_team(connection, team_id="team-1", max_budget=42.0)

    await QuotaMirror(executor).run()

    quota = (await fetch_quotas(connection))[0]
    assert quota["period"] == "custom"
    assert quota["reset_at"] is None
    assert quota["period_end"] is None


async def test_mirroring_twice_updates_in_place(connection, executor):
    await insert_team(connection, team_id="team-1", max_budget=100.0, budget_duration="30d", budget_reset_at=RESET_AT)
    mirror = QuotaMirror(executor)
    await mirror.run()
    original = (await fetch_quotas(connection))[0]

    await connection.execute('UPDATE "LiteLLM_TeamTable" SET max_budget = 250 WHERE team_id = %s', ("team-1",))
    await mirror.run()

    quotas = await fetch_quotas(connection)
    assert len(quotas) == 1
    assert quotas[0]["quota_id"] == original["quota_id"]
    assert quotas[0]["limit_value"] == Decimal("250.0000")


async def test_a_removed_budget_stops_being_an_entitlement(connection, executor):
    await insert_team(connection, team_id="team-1", max_budget=100.0)
    mirror = QuotaMirror(executor)
    await mirror.run()
    assert len(await fetch_quotas(connection)) == 1

    await connection.execute('UPDATE "LiteLLM_TeamTable" SET max_budget = NULL WHERE team_id = %s', ("team-1",))
    await mirror.run()

    assert await fetch_quotas(connection) == []


async def test_changing_a_budget_period_does_not_leave_a_stale_quota(connection, executor):
    await insert_team(connection, team_id="team-1", max_budget=100.0, budget_duration="30d", budget_reset_at=RESET_AT)
    mirror = QuotaMirror(executor)
    await mirror.run()

    await connection.execute('UPDATE "LiteLLM_TeamTable" SET budget_duration = %s WHERE team_id = %s', ("7d", "team-1"))
    await mirror.run()

    quotas = await fetch_quotas(connection)
    assert [row["period"] for row in quotas] == ["week"]


async def test_manual_quotas_are_never_touched_by_the_mirror(connection, executor):
    await connection.execute(
        """
        INSERT INTO "WITOS_FinOpsQuota" (quota_id, scope_type, scope_id, metric, limit_value, period,
                                          period_start, source_type, updated_at)
        VALUES ('manual-1', 'team', 'team-1', 'total_tokens', 1000000, 'contract', now(), 'customer_contract', now())
        """
    )
    await insert_team(connection, team_id="team-1", max_budget=100.0)

    await QuotaMirror(executor).run()

    quotas = {row["quota_id"]: row for row in await fetch_quotas(connection)}
    assert "manual-1" in quotas
    assert quotas["manual-1"]["limit_value"] == Decimal("1000000.0000")
    assert len(quotas) == 2


async def test_a_second_pod_skips_the_mirror(connection, migrated_database):
    import psycopg

    from litellm.proxy.witos.finops.quotas import QUOTA_MIRROR_JOB_NAME, MirrorSkipped
    from litellm.proxy.witos.shared.scheduler import JobLeaseManager
    from tests.witos.finops.conftest import PsycopgExecutor

    await insert_team(connection, team_id="team-1", max_budget=100.0)
    async with await psycopg.AsyncConnection.connect(migrated_database, autocommit=True) as holder_conn:
        await JobLeaseManager(PsycopgExecutor(holder_conn), owner_id="pod-a").acquire(
            QUOTA_MIRROR_JOB_NAME, ttl=timedelta(hours=1)
        )
        outcome = await QuotaMirror(PsycopgExecutor(connection)).run()

    assert isinstance(outcome, MirrorSkipped)
    assert await fetch_quotas(connection) == []
