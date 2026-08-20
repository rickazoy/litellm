"""The nightly run: what it writes, what it skips, and what it costs per scope."""

import time
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

from litellm.proxy.witos.finops.engine import ScopeForecast, build_forecast
from litellm.proxy.witos.finops.forecast_metrics import ForecastMetrics
from litellm.proxy.witos.finops.history import ScopeRef
from litellm.proxy.witos.finops.jobs import ForecastJob, ForecastRunSkipped, ForecastRunSummary
from litellm.proxy.witos.shared.scheduler import JobLeaseManager

from tests.witos.finops.fakes import RecordingExecutor, seasonal_history

AS_OF = date(2026, 7, 29)
NOW = datetime(2026, 7, 30, 2, 10, tzinfo=timezone.utc)
SCOPE = ScopeRef("team", "finance-ai")


def daily_rows(days: int) -> tuple[dict[str, object], ...]:
    history = seasonal_history(days, scope=SCOPE, start=AS_OF - timedelta(days=days - 1))
    return tuple(
        {
            "bucket_date": datetime.combine(day.day, datetime.min.time(), tzinfo=timezone.utc),
            "request_count": int(day.request_count),
            "prompt_tokens": int(day.prompt_tokens),
            "completion_tokens": int(day.completion_tokens),
            "cache_read_tokens": int(day.cache_read_tokens),
            "spend_usd": day.spend_usd,
            "breakdown": {model: {"tokens": Decimal(str(tokens))} for model, tokens in day.mix_tokens.items()},
        }
        for day in history.days
    )


def executor(*, days: int = 90, lease_taken: bool = False) -> RecordingExecutor:
    lease_rows = () if lease_taken else ({"owner_id": "me", "lease_until": NOW + timedelta(minutes=10)},)
    return RecordingExecutor(
        responses=(
            ('INSERT INTO "WITOS_BackgroundJobLease"', lease_rows),
            (
                "SELECT owner_id, lease_until FROM",
                ({"owner_id": "other-pod", "lease_until": NOW + timedelta(minutes=10)},),
            ),
            ('FROM "WITOS_FinOpsAlertRule"', ()),
            (
                "GROUP BY scope_type, scope_id",
                (
                    {
                        "scope_type": SCOPE.scope_type,
                        "scope_id": SCOPE.scope_id,
                        "scope_label": "Finance AI",
                        "spend_usd": Decimal("1200"),
                    },
                ),
            ),
            ("ORDER BY bucket_date", daily_rows(days)),
            (
                "COALESCE(SUM(spend_usd), 0)::numeric(18, 8) AS spend_usd,\n       COALESCE(SUM(total_tokens)",
                (
                    {
                        "spend_usd": Decimal("120"),
                        "total_tokens": 1_000_000,
                        "prompt_tokens": 800_000,
                        "completion_tokens": 200_000,
                        "request_count": 400,
                    },
                ),
            ),
            ('FROM "WITOS_FinOpsQuota"', ()),
            ("AND metric = $3", ()),
        )
    )


async def test_a_run_writes_a_forecast_for_every_active_scope() -> None:
    recorded = executor()
    summary = await ForecastJob(recorded).run(as_of=AS_OF, now=NOW)

    assert isinstance(summary, ForecastRunSummary)
    assert summary.scopes_seen == 1
    assert summary.states["forecast"] == 1
    assert summary.forecasts_written == 1
    assert summary.stopped_early is False
    assert any('INSERT INTO "WITOS_FinOpsForecast"' in sql for sql in recorded.statements())


async def test_a_scope_below_the_ladder_is_superseded_rather_than_forecast() -> None:
    recorded = executor(days=2)
    summary = await ForecastJob(recorded).run(as_of=AS_OF, now=NOW)

    assert isinstance(summary, ForecastRunSummary)
    assert summary.states["insufficient_history"] == 1
    assert summary.forecasts_written == 0
    assert not any('INSERT INTO "WITOS_FinOpsForecast"' in sql for sql in recorded.statements())
    assert any("SET is_latest = false" in sql for sql in recorded.statements())


async def test_a_run_on_another_pod_is_a_no_op_for_this_one() -> None:
    recorded = executor(lease_taken=True)
    outcome = await ForecastJob(recorded).run(as_of=AS_OF, now=NOW)

    assert isinstance(outcome, ForecastRunSkipped)
    assert outcome.held_by == "other-pod"
    assert not any('INSERT INTO "WITOS_FinOpsForecast"' in sql for sql in recorded.statements())


async def test_a_single_scope_rebuild_does_not_touch_the_rest_of_the_fleet() -> None:
    recorded = executor()
    outcome = await ForecastJob(recorded).run_for(SCOPE, as_of=AS_OF, now=NOW)

    assert outcome.state == "forecast"
    assert outcome.forecasts == 1
    assert not any("GROUP BY scope_type, scope_id" in sql for sql in recorded.statements())


async def test_the_lease_is_released_even_though_the_run_completed() -> None:
    recorded = executor()
    await ForecastJob(recorded, lease_manager=JobLeaseManager(recorded, owner_id="me")).run(as_of=AS_OF, now=NOW)

    assert any("lease_until = '-infinity'" in sql for sql in recorded.statements())


def test_one_scope_stays_well_inside_the_per_scope_budget() -> None:
    """§1.14 budgets 5 minutes for 2,000 scopes: 150ms each.

    The margin here is deliberately wide because CI hardware is not a laptop.
    What it actually catches is the failure that motivated the expensive-model
    gate: fitting maximum-likelihood models on every series of every scope took
    2.8 seconds, twenty times the budget, and nothing else in the engine is
    within an order of magnitude of that.
    """
    history = seasonal_history(90, scope=SCOPE)
    started = time.perf_counter()
    built = build_forecast(history, as_of=history.end_day)
    elapsed = time.perf_counter() - started

    assert isinstance(built, ScopeForecast)
    assert elapsed < 1.0, f"one scope took {elapsed:.2f}s against a 0.15s budget"


def test_a_year_of_history_does_not_cost_quadratically() -> None:
    short = seasonal_history(90, scope=SCOPE)
    long = seasonal_history(360, scope=SCOPE)

    short_started = time.perf_counter()
    build_forecast(short, as_of=short.end_day)
    short_elapsed = time.perf_counter() - short_started

    long_started = time.perf_counter()
    build_forecast(long, as_of=long.end_day)
    long_elapsed = time.perf_counter() - long_started

    assert long_elapsed < max(0.2, short_elapsed * 8)


def test_metrics_recording_never_raises_even_without_prometheus() -> None:
    ForecastMetrics.record_scope("forecast")
    ForecastMetrics.record_run(1.5)
    ForecastMetrics.record_accuracy("team", wape=0.1, bias=-0.02)
    ForecastMetrics.record_accuracy("team", wape=None, bias=None)
    ForecastMetrics.record_alert("forecast_exceeds_budget")
