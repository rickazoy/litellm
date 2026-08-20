"""Incremental aggregation on a watermark plus lateness window (blueprint §1.5)."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from litellm.proxy.witos.finops.aggregator import (
    DEFAULT_LATENESS_WINDOW,
    MIN_LATENESS_WINDOW,
    AggregationWindow,
    FinOpsAggregator,
    plan_window,
)
from litellm.proxy.witos.finops.config import HOURLY_WATERMARK_KEY, FinOpsConfig
from tests.witos.finops.factories import fetch_daily, fetch_hourly, insert_spend_log, insert_team

NOON = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)


def window_around(*, start: datetime, end: datetime) -> AggregationWindow:
    return AggregationWindow(start=start, end=end, next_watermark=end)


def wide_window() -> AggregationWindow:
    return window_around(start=NOON - timedelta(days=30), end=NOON + timedelta(days=30))


class TestWindowPlanning:
    def test_cold_start_sweeps_only_the_overlap(self):
        window = plan_window(watermark=None, now=NOON + timedelta(minutes=37))
        assert window.start == NOON - timedelta(hours=2)
        assert window.end == NOON + timedelta(minutes=37)

    def test_watermark_rewinds_by_the_lateness_window(self):
        window = plan_window(watermark=NOON, now=NOON + timedelta(minutes=5))
        assert window.start == NOON - DEFAULT_LATENESS_WINDOW

    def test_watermark_never_advances_into_the_partial_hour(self):
        """The in-flight hour is still filling, so the next cycle has to sweep it again."""
        now = NOON + timedelta(minutes=41, seconds=12)
        window = plan_window(watermark=NOON, now=now)
        assert window.end == now
        assert window.next_watermark == NOON
        assert plan_window(watermark=window.next_watermark, now=now).start <= NOON

    def test_lateness_window_cannot_drop_below_one_hour(self):
        """Below an hour of overlap the partial bucket could fall out between cycles."""
        window = plan_window(watermark=NOON, now=NOON + timedelta(minutes=5), lateness_window=timedelta(minutes=1))
        assert window.start == NOON - MIN_LATENESS_WINDOW

    def test_daily_span_covers_whole_days(self):
        window = plan_window(watermark=NOON, now=NOON + timedelta(minutes=5))
        assert window.daily_start == datetime(2026, 8, 12, tzinfo=timezone.utc)
        assert window.daily_end == datetime(2026, 8, 13, tzinfo=timezone.utc)


class TestHourlyAggregation:
    async def test_buckets_by_utc_hour_and_scope(self, connection, executor):
        await insert_spend_log(connection, request_id="r1", start_time=NOON + timedelta(minutes=5), spend=2.5)
        await insert_spend_log(connection, request_id="r2", start_time=NOON + timedelta(minutes=45), spend=1.5)
        await insert_spend_log(connection, request_id="r3", start_time=NOON + timedelta(hours=1), spend=4.0)

        await FinOpsAggregator(executor).aggregate_hourly(wide_window())

        buckets = await fetch_hourly(connection, "team", "team-1")
        assert [row["bucket_start_utc"].hour for row in buckets] == [12, 13]
        assert buckets[0]["request_count"] == 2
        assert buckets[0]["spend_usd"] == Decimal("4.00000000")
        assert buckets[1]["spend_usd"] == Decimal("4.00000000")

    async def test_recomputing_a_window_is_idempotent(self, connection, executor):
        """A cycle re-sweeping the same window must not double anything."""
        for index in range(5):
            await insert_spend_log(connection, request_id=f"r{index}", start_time=NOON + timedelta(minutes=index))
        aggregator = FinOpsAggregator(executor)

        await aggregator.aggregate_hourly(wide_window())
        first = await fetch_hourly(connection, "team", "team-1")
        await aggregator.aggregate_hourly(wide_window())
        await aggregator.aggregate_hourly(wide_window())
        third = await fetch_hourly(connection, "team", "team-1")

        assert len(third) == 1
        assert [row["id"] for row in first] == [row["id"] for row in third]
        for column in ("request_count", "prompt_tokens", "completion_tokens", "spend_usd", "breakdown"):
            assert first[0][column] == third[0][column]

    async def test_a_row_arriving_after_its_bucket_was_written_is_picked_up(self, connection, executor):
        """The whole reason the sweep overlaps: the spend writer is asynchronous."""
        aggregator = FinOpsAggregator(executor)
        await insert_spend_log(connection, request_id="on-time", start_time=NOON + timedelta(minutes=5), spend=1.0)
        await aggregator.aggregate_hourly(wide_window())
        assert (await fetch_hourly(connection, "team", "team-1"))[0]["spend_usd"] == Decimal("1.00000000")

        await insert_spend_log(connection, request_id="late", start_time=NOON + timedelta(minutes=6), spend=3.0)
        await aggregator.aggregate_hourly(wide_window())

        corrected = await fetch_hourly(connection, "team", "team-1")
        assert len(corrected) == 1
        assert corrected[0]["request_count"] == 2
        assert corrected[0]["spend_usd"] == Decimal("4.00000000")

    async def test_a_row_older_than_the_lateness_window_is_not_swept(self, connection, executor):
        """Documents the bound: correction is only guaranteed inside the overlap."""
        aggregator = FinOpsAggregator(executor)
        await insert_spend_log(connection, request_id="ancient", start_time=NOON - timedelta(hours=6))

        await aggregator.aggregate_hourly(plan_window(watermark=NOON, now=NOON))

        assert await fetch_hourly(connection, "team", "team-1") == []

    async def test_success_and_failure_are_counted_separately(self, connection, executor):
        await insert_spend_log(connection, request_id="ok", start_time=NOON, status="success")
        await insert_spend_log(connection, request_id="bad", start_time=NOON, status="failure")
        await insert_spend_log(connection, request_id="unknown", start_time=NOON, status=None)

        await FinOpsAggregator(executor).aggregate_hourly(wide_window())

        bucket = (await fetch_hourly(connection, "team", "team-1"))[0]
        assert (bucket["request_count"], bucket["successful_requests"], bucket["failed_requests"]) == (3, 1, 1)

    async def test_cache_tokens_are_read_out_of_the_spend_log_metadata(self, connection, executor):
        await insert_spend_log(
            connection, request_id="c1", start_time=NOON, cache_read_tokens=900, cache_creation_tokens=120
        )
        await insert_spend_log(connection, request_id="c2", start_time=NOON, cache_read_tokens=100)

        await FinOpsAggregator(executor).aggregate_hourly(wide_window())

        bucket = (await fetch_hourly(connection, "team", "team-1"))[0]
        assert bucket["cache_read_tokens"] == 1000
        assert bucket["cache_creation_tokens"] == 120

    async def test_a_non_numeric_cache_token_value_counts_as_zero(self, connection, executor):
        """Metadata is untrusted JSON; a bad value must not abort the whole cycle."""
        await insert_spend_log(connection, request_id="c1", start_time=NOON)
        await connection.execute(
            """UPDATE "LiteLLM_SpendLogs" SET metadata = '{"additional_usage_values":
               {"cache_read_input_tokens": "lots"}}'::jsonb"""
        )

        await FinOpsAggregator(executor).aggregate_hourly(wide_window())

        assert (await fetch_hourly(connection, "team", "team-1"))[0]["cache_read_tokens"] == 0

    async def test_a_zero_cost_model_still_produces_a_bucket(self, connection, executor):
        await insert_spend_log(connection, request_id="free", start_time=NOON, spend=0.0, model="local-llama")

        await FinOpsAggregator(executor).aggregate_hourly(wide_window())

        bucket = (await fetch_hourly(connection, "team", "team-1"))[0]
        assert bucket["request_count"] == 1
        assert bucket["spend_usd"] == Decimal("0.00000000")
        assert bucket["total_tokens"] == 150

    async def test_unpriced_and_unknown_models_are_reported_as_coverage(self, connection, executor):
        await insert_spend_log(connection, request_id="priced", start_time=NOON, model="gpt-4o", spend=1.0)
        await insert_spend_log(connection, request_id="unpriced", start_time=NOON, model="gpt-4o", spend=0.0)
        await insert_spend_log(connection, request_id="unknown", start_time=NOON, model="witone-dslm-x", spend=0.0)

        coverage = await FinOpsAggregator(executor).measure_pricing_coverage(wide_window())

        assert coverage.missing_price_requests == 2
        assert coverage.unknown_model_requests == 1

    async def test_scope_label_survives_a_deleted_team(self, connection, executor):
        await insert_team(connection, team_id="team-1", alias="Platform")
        await insert_spend_log(connection, request_id="r1", start_time=NOON)
        aggregator = FinOpsAggregator(executor)
        await aggregator.aggregate_hourly(wide_window())
        assert (await fetch_hourly(connection, "team", "team-1"))[0]["scope_label"] == "Platform"

        await connection.execute('DELETE FROM "LiteLLM_TeamTable" WHERE team_id = %s', ("team-1",))
        await aggregator.aggregate_hourly(wide_window())

        surviving = await fetch_hourly(connection, "team", "team-1")
        assert len(surviving) == 1
        assert surviving[0]["scope_label"] is None
        assert surviving[0]["request_count"] == 1

    async def test_the_team_column_is_left_null_when_a_user_spans_teams(self, connection, executor):
        """Denormalizing an arbitrary member would print a confident wrong answer."""
        await insert_spend_log(connection, request_id="a", start_time=NOON, user="u1", team_id="team-1")
        await insert_spend_log(connection, request_id="b", start_time=NOON, user="u1", team_id="team-2")

        await FinOpsAggregator(executor).aggregate_hourly(wide_window())

        assert (await fetch_hourly(connection, "user", "u1"))[0]["team_id"] is None

    async def test_model_mix_breakdown_is_written_per_model_group(self, connection, executor):
        await insert_spend_log(connection, request_id="a", start_time=NOON, model_group="gpt-4o", spend=1.0)
        await insert_spend_log(connection, request_id="b", start_time=NOON, model_group="gpt-4o", spend=2.0)
        await insert_spend_log(connection, request_id="c", start_time=NOON, model_group="haiku", spend=0.25)

        await FinOpsAggregator(executor).aggregate_hourly(wide_window())

        breakdown = (await fetch_hourly(connection, "team", "team-1"))[0]["breakdown"]
        assert set(breakdown) == {"gpt-4o", "haiku"}
        assert breakdown["gpt-4o"]["requests"] == 2
        assert Decimal(str(breakdown["gpt-4o"]["spend"])) == Decimal("3")


class TestTagScopes:
    async def test_only_approved_tag_dimensions_become_facts(self, connection, executor):
        await insert_spend_log(
            connection,
            request_id="tagged",
            start_time=NOON,
            request_tags=["environment:prod", "cost_center:CC-1", "adhoc:whatever", "no-dimension"],
        )

        await FinOpsAggregator(executor).aggregate_hourly(wide_window())

        rows = await connection.execute(
            """SELECT scope_id FROM "WITOS_FinOpsUsageHourly" WHERE scope_type = 'tag' ORDER BY scope_id"""
        )
        assert [row[0] for row in await rows.fetchall()] == ["cost_center:CC-1", "environment:prod"]

    async def test_changing_the_approved_dimensions_changes_what_is_aggregated(self, connection, executor):
        await FinOpsConfig(executor).set_approved_tag_dimensions(("team_name",))
        await insert_spend_log(
            connection, request_id="tagged", start_time=NOON, request_tags=["environment:prod", "team_name:core"]
        )

        await FinOpsAggregator(executor).aggregate_hourly(wide_window())

        assert (await fetch_hourly(connection, "tag", "team_name:core"))[0]["scope_label"] == "core"
        assert await fetch_hourly(connection, "tag", "environment:prod") == []


class TestDailyRollup:
    async def test_hours_roll_into_one_day(self, connection, executor):
        for hour in (0, 6, 23):
            await insert_spend_log(
                connection, request_id=f"h{hour}", start_time=NOON.replace(hour=hour), spend=1.0, prompt_tokens=10
            )
        aggregator = FinOpsAggregator(executor)

        await aggregator.aggregate_hourly(wide_window())
        await aggregator.roll_up_daily(wide_window())

        days = await fetch_daily(connection, "team", "team-1")
        assert len(days) == 1
        assert days[0]["request_count"] == 3
        assert days[0]["spend_usd"] == Decimal("3.00000000")
        assert days[0]["prompt_tokens"] == 30

    async def test_a_late_row_corrects_the_day_it_belongs_to(self, connection, executor):
        aggregator = FinOpsAggregator(executor)
        await insert_spend_log(connection, request_id="early", start_time=NOON, spend=1.0)
        await aggregator.aggregate_hourly(wide_window())
        await aggregator.roll_up_daily(wide_window())

        await insert_spend_log(connection, request_id="late", start_time=NOON.replace(hour=3), spend=5.0)
        await aggregator.aggregate_hourly(wide_window())
        await aggregator.roll_up_daily(wide_window())

        days = await fetch_daily(connection, "team", "team-1")
        assert len(days) == 1
        assert days[0]["spend_usd"] == Decimal("6.00000000")

    async def test_a_month_boundary_splits_into_two_days(self, connection, executor):
        aggregator = FinOpsAggregator(executor)
        await insert_spend_log(
            connection, request_id="jul", start_time=datetime(2026, 7, 31, 23, 30, tzinfo=timezone.utc), spend=1.0
        )
        await insert_spend_log(
            connection, request_id="aug", start_time=datetime(2026, 8, 1, 0, 30, tzinfo=timezone.utc), spend=2.0
        )

        window = window_around(
            start=datetime(2026, 7, 31, tzinfo=timezone.utc), end=datetime(2026, 8, 2, tzinfo=timezone.utc)
        )
        await aggregator.aggregate_hourly(window)
        await aggregator.roll_up_daily(window)

        days = await fetch_daily(connection, "team", "team-1")
        assert [(day["bucket_date"].month, day["bucket_date"].day) for day in days] == [(7, 31), (8, 1)]
        assert [day["spend_usd"] for day in days] == [Decimal("1.00000000"), Decimal("2.00000000")]

    async def test_a_local_dst_transition_still_yields_24_hourly_buckets(self, connection, executor):
        """Bucketing is pure UTC, so no wall-clock jump can shorten or duplicate a day.

        2026-03-08 is the US spring-forward date: local time skips 02:00-03:00.
        """
        day = datetime(2026, 3, 8, tzinfo=timezone.utc)
        for hour in range(24):
            await insert_spend_log(
                connection, request_id=f"dst{hour}", start_time=day + timedelta(hours=hour), spend=1.0
            )
        aggregator = FinOpsAggregator(executor)
        window = window_around(start=day - timedelta(days=1), end=day + timedelta(days=2))

        await aggregator.aggregate_hourly(window)
        await aggregator.roll_up_daily(window)

        assert len(await fetch_hourly(connection, "team", "team-1")) == 24
        days = await fetch_daily(connection, "team", "team-1")
        assert len(days) == 1
        assert days[0]["request_count"] == 24
        assert days[0]["spend_usd"] == Decimal("24.00000000")

    async def test_rollup_is_idempotent(self, connection, executor):
        await insert_spend_log(connection, request_id="a", start_time=NOON, spend=7.0)
        aggregator = FinOpsAggregator(executor)
        await aggregator.aggregate_hourly(wide_window())

        await aggregator.roll_up_daily(wide_window())
        first = await fetch_daily(connection, "team", "team-1")
        await aggregator.roll_up_daily(wide_window())
        second = await fetch_daily(connection, "team", "team-1")

        assert [row["id"] for row in first] == [row["id"] for row in second]
        assert first[0]["spend_usd"] == second[0]["spend_usd"] == Decimal("7.00000000")


class TestCycle:
    async def test_a_cycle_advances_the_watermark_to_the_top_of_the_hour(self, connection, executor):
        now = NOON + timedelta(minutes=23)
        await insert_spend_log(connection, request_id="a", start_time=NOON + timedelta(minutes=10))

        outcome = await FinOpsAggregator(executor, clock=lambda: now).run_cycle()

        assert outcome.window.next_watermark == NOON
        assert await FinOpsConfig(executor).watermark(HOURLY_WATERMARK_KEY) == NOON
        assert (await fetch_hourly(connection, "team", "team-1"))[0]["request_count"] == 1

    async def test_a_crashed_cycle_leaves_the_watermark_alone_and_the_next_cycle_resumes(self, connection, executor):
        """The watermark is only advanced after the sweep, so a crash mid-cycle costs nothing."""
        now = NOON + timedelta(minutes=23)
        await insert_spend_log(connection, request_id="a", start_time=NOON + timedelta(minutes=10))

        class ExplodingExecutor:
            def __init__(self, inner):
                self._inner = inner
                self.calls = 0

            async def query(self, sql, *args):
                return await self._inner.query(sql, *args)

            async def execute(self, sql, *args):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("pod died mid-cycle")
                return await self._inner.execute(sql, *args)

        exploding = ExplodingExecutor(executor)
        with pytest.raises(RuntimeError):
            await FinOpsAggregator(exploding, clock=lambda: now).run_cycle()
        assert await FinOpsConfig(executor).watermark(HOURLY_WATERMARK_KEY) is None

        outcome = await FinOpsAggregator(executor, clock=lambda: now).run_cycle()

        assert await FinOpsConfig(executor).watermark(HOURLY_WATERMARK_KEY) == NOON
        assert outcome.hourly_rows > 0
        assert (await fetch_hourly(connection, "team", "team-1"))[0]["request_count"] == 1

    async def test_a_second_pod_skips_the_cycle_while_the_first_holds_the_lease(self, connection, migrated_database):
        import psycopg

        from litellm.proxy.witos.finops.aggregator import AGGREGATION_JOB_NAME, CycleSkipped
        from litellm.proxy.witos.shared.scheduler import JobLeaseManager
        from tests.witos.finops.conftest import PsycopgExecutor

        async with await psycopg.AsyncConnection.connect(migrated_database, autocommit=True) as holder_conn:
            holder = JobLeaseManager(PsycopgExecutor(holder_conn), owner_id="pod-a")
            await holder.acquire(AGGREGATION_JOB_NAME, ttl=timedelta(hours=1))

            async with await psycopg.AsyncConnection.connect(migrated_database, autocommit=True) as other_conn:
                await insert_spend_log(other_conn, request_id="a", start_time=NOON)
                outcome = await FinOpsAggregator(
                    PsycopgExecutor(other_conn), clock=lambda: NOON + timedelta(minutes=5)
                ).run_cycle()

                assert isinstance(outcome, CycleSkipped)
                assert outcome.held_by == "pod-a"
                assert await fetch_hourly(other_conn, "team", "team-1") == []

    async def test_unknown_scope_types_are_rejected_at_construction(self, executor):
        with pytest.raises(ValueError, match="Unknown FinOps scope types"):
            FinOpsAggregator(executor, scope_types=("team", "solar_system"))
