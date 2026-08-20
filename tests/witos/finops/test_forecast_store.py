"""What the store writes: supersede before insert, one statement per scope, UTC everywhere."""

from datetime import date, datetime, timezone
from decimal import Decimal

from litellm.proxy.witos.finops.engine import ScopeForecast, build_forecast
from litellm.proxy.witos.finops.forecast_store import (
    PERSISTED_METRICS,
    ForecastStore,
    forecast_payload,
)
from litellm.proxy.witos.finops.history import ScopeRef
from litellm.proxy.witos.finops.runway import Quota, compute_runway

from tests.witos.finops.fakes import RecordingExecutor, seasonal_history

SCOPE = ScopeRef("team", "finance-ai")
GENERATED = datetime(2026, 8, 21, 2, 10, tzinfo=timezone.utc)


def forecast() -> ScopeForecast:
    history = seasonal_history(90, scope=SCOPE)
    built = build_forecast(history, as_of=history.end_day, generated_at=GENERATED)
    assert isinstance(built, ScopeForecast)
    return built


def quota() -> Quota:
    return Quota(
        quota_id="q1",
        scope=SCOPE,
        metric="usd",
        limit_value=Decimal("500"),
        period="month",
        period_start=datetime(2026, 7, 1, tzinfo=timezone.utc),
        period_end=datetime(2026, 7, 31, tzinfo=timezone.utc),
        reset_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
        source_type="litellm_budget",
    )


async def test_a_save_supersedes_the_scope_before_inserting_anything() -> None:
    """Two current forecasts for one scope would make "latest" ambiguous for every reader."""
    executor = RecordingExecutor()
    built = forecast()

    await ForecastStore(executor).save(built, actual_to_date=Decimal("100"))
    statements = [sql for sql, _ in executor.executions]

    assert "SET is_latest = false" in statements[0]
    assert "INSERT INTO" in statements[1]
    assert statements.index([s for s in statements if "INSERT" in s][0]) > 0


async def test_every_persisted_metric_is_written_in_one_statement() -> None:
    executor = RecordingExecutor()
    built = forecast()

    counts = await ForecastStore(executor).save(built, actual_to_date=Decimal("100"))
    inserts = [sql for sql, _ in executor.executions if 'INSERT INTO "WITOS_FinOpsForecast"' in sql]

    assert len(inserts) == 1, "2,000 scopes must not become 10,000 round trips"
    assert counts.forecasts == 1
    assert inserts[0].count("(") >= len(PERSISTED_METRICS)


async def test_runway_rows_are_written_with_the_forecast() -> None:
    executor = RecordingExecutor()
    built = forecast()
    runway = compute_runway(built, quota(), Decimal("10"), today=built.forecast_start)
    assert runway is not None

    await ForecastStore(executor).save(built, actual_to_date=Decimal("100"), runway=(runway,))
    inserts = [sql for sql, _ in executor.executions if 'INSERT INTO "WITOS_FinOpsRunway"' in sql]

    assert len(inserts) == 1
    assert "gen_random_uuid()::text" in inserts[0]


async def test_every_timestamp_is_bound_as_utc_and_every_parameter_is_cast() -> None:
    """Naive TIMESTAMP(3) columns plus a session timezone is how two pods disagree about a date."""
    executor = RecordingExecutor()

    await ForecastStore(executor).save(forecast(), actual_to_date=Decimal("100"))
    insert = next(sql for sql, _ in executor.executions if "INSERT INTO" in sql)

    assert "::timestamptz AT TIME ZONE 'UTC'" in insert
    assert "::jsonb" in insert
    assert "$1::text" in insert


async def test_dates_reach_the_driver_as_datetimes() -> None:
    executor = RecordingExecutor()
    built = forecast()
    runway = compute_runway(built, quota(), Decimal("10"), today=built.forecast_start)
    assert runway is not None

    await ForecastStore(executor).save(built, actual_to_date=Decimal("1"), runway=(runway,))
    _, args = next((sql, args) for sql, args in executor.executions if 'INSERT INTO "WITOS_FinOpsRunway"' in sql)

    assert not any(isinstance(value, date) and not isinstance(value, datetime) for value in args)


async def test_superseding_a_dormant_scope_writes_no_new_rows() -> None:
    executor = RecordingExecutor()

    await ForecastStore(executor).supersede(SCOPE)

    assert len(executor.executions) == 2
    assert all("SET is_latest = false" in sql for sql, _ in executor.executions)


def test_the_payload_carries_the_provenance_a_reader_needs_to_judge_it() -> None:
    built = forecast()
    payload = forecast_payload(built, "spend_usd", built.spend)

    assert payload["tier"] == "full_ensemble"
    assert payload["path_count"] == 500
    assert payload["has_band"] is True
    assert payload["seed"] == built.seed
    assert isinstance(payload["backtest"], dict)
    assert len(payload["days"]) == built.spend.horizon  # pyright: ignore[reportArgumentType]  # the payload is this module's own document
    assert len(payload["cumulative"]) == built.spend.horizon  # pyright: ignore[reportArgumentType]  # the payload is this module's own document


def test_the_payload_records_which_models_could_not_be_priced() -> None:
    built = forecast()
    payload = forecast_payload(built, "spend_usd", built.spend)

    assert payload["unknown_models"] == ()
    assert payload["unknown_model_share"] == 0.0
    assert payload["pricing_calibration"] is not None
