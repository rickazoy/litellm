"""The aggregation instruments an operator pages on (blueprint §1.14)."""

from datetime import datetime, timedelta, timezone

from prometheus_client import REGISTRY

from litellm.proxy.witos.finops.aggregator import FinOpsAggregator
from litellm.proxy.witos.finops.metrics import FinOpsMetrics
from tests.witos.finops.factories import insert_spend_log

NOON = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)


async def test_a_cycle_publishes_lag_and_last_success(connection, executor):
    """Lag is how stale the fact tables are, and every forecast built on them inherits it."""
    now = NOON + timedelta(minutes=42)
    await insert_spend_log(connection, request_id="a", start_time=NOON + timedelta(minutes=10))

    await FinOpsAggregator(executor, clock=lambda: now).run_cycle()

    assert REGISTRY.get_sample_value("finops_aggregation_lag_seconds") == 42 * 60
    assert REGISTRY.get_sample_value("finops_last_success_timestamp") == now.timestamp()


async def test_pricing_coverage_counters_only_move_on_real_gaps(connection, executor):
    before_missing = REGISTRY.get_sample_value("finops_missing_price_requests_total") or 0.0
    before_unknown = REGISTRY.get_sample_value("finops_unknown_model_requests_total") or 0.0

    FinOpsMetrics.record_pricing_coverage(0, 0)
    assert (REGISTRY.get_sample_value("finops_missing_price_requests_total") or 0.0) == before_missing

    FinOpsMetrics.record_pricing_coverage(3, 2)
    assert REGISTRY.get_sample_value("finops_missing_price_requests_total") == before_missing + 3
    assert REGISTRY.get_sample_value("finops_unknown_model_requests_total") == before_unknown + 2


async def test_a_skipped_cycle_is_labelled_as_such(connection, migrated_database):
    import psycopg

    from litellm.proxy.witos.finops.aggregator import AGGREGATION_JOB_NAME
    from litellm.proxy.witos.shared.scheduler import JobLeaseManager
    from tests.witos.finops.conftest import PsycopgExecutor

    before = REGISTRY.get_sample_value("finops_aggregation_cycles_total", {"outcome": "skipped_locked"}) or 0.0
    async with await psycopg.AsyncConnection.connect(migrated_database, autocommit=True) as holder_conn:
        await JobLeaseManager(PsycopgExecutor(holder_conn), owner_id="pod-a").acquire(
            AGGREGATION_JOB_NAME, ttl=timedelta(hours=1)
        )
        await FinOpsAggregator(PsycopgExecutor(connection), clock=lambda: NOON).run_cycle()

    assert REGISTRY.get_sample_value("finops_aggregation_cycles_total", {"outcome": "skipped_locked"}) == before + 1
