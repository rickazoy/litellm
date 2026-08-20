"""Historical rebuild: chunking, idempotency, and leaving the live watermark alone."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from litellm.proxy.witos.finops.backfill import BackfillCompleted, BackfillSkipped, backfill, chunk_windows
from litellm.proxy.witos.finops.config import HOURLY_WATERMARK_KEY, FinOpsConfig
from tests.witos.finops.factories import fetch_daily, fetch_hourly, insert_spend_log

END = datetime(2026, 8, 12, 0, 0, tzinfo=timezone.utc)


def measured(rows: list[dict]) -> list[dict]:
    """Everything a recompute must reproduce exactly. ``updated_at`` legitimately moves."""
    return [{key: value for key, value in row.items() if key != "updated_at"} for row in rows]


class TestChunking:
    def test_a_range_splits_into_ascending_whole_chunks(self):
        windows = chunk_windows(END - timedelta(days=3), END, timedelta(days=1))
        assert [window.start for window in windows] == [
            END - timedelta(days=3),
            END - timedelta(days=2),
            END - timedelta(days=1),
        ]
        assert windows[-1].end == END

    def test_a_trailing_partial_chunk_is_clamped_to_the_end(self):
        windows = chunk_windows(END - timedelta(hours=30), END, timedelta(days=1))
        assert len(windows) == 2
        assert windows[-1].end == END
        assert windows[-1].end - windows[-1].start == timedelta(hours=6)

    def test_an_empty_range_produces_no_work(self):
        assert chunk_windows(END, END, timedelta(days=1)) == ()

    def test_a_non_positive_chunk_is_rejected(self):
        with pytest.raises(ValueError, match="chunk must be positive"):
            chunk_windows(END - timedelta(days=1), END, timedelta(0))


class TestBackfill:
    async def test_history_is_rebuilt_across_days(self, connection, executor):
        for day in range(1, 4):
            await insert_spend_log(
                connection, request_id=f"d{day}", start_time=END - timedelta(days=day, hours=-5), spend=2.0
            )

        outcome = await backfill(executor, days=5, end=END)

        assert isinstance(outcome, BackfillCompleted)
        assert outcome.chunks == 5
        assert len(await fetch_hourly(connection, "team", "team-1")) == 3
        days = await fetch_daily(connection, "team", "team-1")
        assert len(days) == 3
        assert all(row["spend_usd"] == Decimal("2.00000000") for row in days)

    async def test_running_the_backfill_twice_changes_nothing(self, connection, executor):
        for day in range(1, 4):
            await insert_spend_log(connection, request_id=f"d{day}", start_time=END - timedelta(days=day, hours=-5))

        await backfill(executor, days=5, end=END)
        first_hourly = measured(await fetch_hourly(connection, "team", "team-1"))
        first_daily = measured(await fetch_daily(connection, "team", "team-1"))
        await backfill(executor, days=5, end=END)

        assert measured(await fetch_hourly(connection, "team", "team-1")) == first_hourly
        assert measured(await fetch_daily(connection, "team", "team-1")) == first_daily

    async def test_a_day_split_across_chunks_ends_up_complete(self, connection, executor):
        """A chunk boundary inside a day must not leave the daily row at the partial total."""
        day = END - timedelta(days=1)
        for hour in (2, 20):
            await insert_spend_log(connection, request_id=f"h{hour}", start_time=day + timedelta(hours=hour), spend=1.0)

        await backfill(executor, days=2, end=END, chunk=timedelta(hours=12))

        days = await fetch_daily(connection, "team", "team-1")
        assert len(days) == 1
        assert days[0]["request_count"] == 2
        assert days[0]["spend_usd"] == Decimal("2.00000000")

    async def test_the_live_watermark_is_untouched(self, connection, executor):
        """Rewinding it would make the 5-minute job re-sweep the whole backfilled range."""
        config = FinOpsConfig(executor)
        await config.set_watermark(HOURLY_WATERMARK_KEY, END)
        await insert_spend_log(connection, request_id="a", start_time=END - timedelta(days=2))

        await backfill(executor, days=5, end=END)

        assert await config.watermark(HOURLY_WATERMARK_KEY) == END

    async def test_a_second_backfill_is_refused_while_one_is_running(self, connection, migrated_database):
        import psycopg

        from litellm.proxy.witos.finops.backfill import BACKFILL_JOB_NAME
        from litellm.proxy.witos.shared.scheduler import JobLeaseManager
        from tests.witos.finops.conftest import PsycopgExecutor

        async with await psycopg.AsyncConnection.connect(migrated_database, autocommit=True) as holder_conn:
            await JobLeaseManager(PsycopgExecutor(holder_conn), owner_id="pod-a").acquire(
                BACKFILL_JOB_NAME, ttl=timedelta(hours=1)
            )
            outcome = await backfill(PsycopgExecutor(connection), days=1, end=END)

        assert isinstance(outcome, BackfillSkipped)
        assert outcome.held_by == "pod-a"

    async def test_zero_days_is_rejected(self, executor):
        with pytest.raises(ValueError, match="--days must be positive"):
            await backfill(executor, days=0)

    async def test_a_restricted_scope_type_only_writes_that_scope(self, connection, executor):
        await insert_spend_log(connection, request_id="a", start_time=END - timedelta(hours=5))

        await backfill(executor, days=1, end=END, scope_types=("team",))

        rows = await connection.execute('SELECT DISTINCT scope_type FROM "WITOS_FinOpsUsageHourly"')
        assert [row[0] for row in await rows.fetchall()] == ["team"]
