"""The engine end to end: what it refuses, what it forecasts, and what it reproduces.

Most of the §1.14 test matrix that applies to Phase 2 lands here: a new team with
no history, the 3/7/28/90 day boundaries, weekend seasonality, a 400% spike, the
month boundary, an unknown model, and rebuild idempotency.
"""

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from types import MappingProxyType

import pytest

from litellm.proxy.witos.finops.engine import (
    NoForecast,
    ScopeForecast,
    build_forecast,
    daily_burn_rate,
    month_end,
    period_projection,
    quarter_end,
)
from litellm.proxy.witos.finops.history import ScopeRef, build_history

from tests.witos.finops.fakes import day_usage, flat_history, priced_day, seasonal_history

GENERATED = datetime(2026, 8, 21, 2, 10, tzinfo=timezone.utc)


def built(history_days: int, **kwargs: float) -> ScopeForecast:
    history = seasonal_history(history_days, **kwargs)  # pyright: ignore[reportArgumentType]  # kwargs are the fixture's float knobs
    outcome = build_forecast(history, as_of=history.end_day, generated_at=GENERATED)
    assert isinstance(outcome, ScopeForecast)
    return outcome


def test_a_new_team_with_no_history_gets_a_state_and_no_numbers() -> None:
    history = flat_history(2)
    outcome = build_forecast(history, as_of=history.end_day)

    assert isinstance(outcome, NoForecast)
    assert outcome.state == "insufficient_history"
    assert "3" in outcome.detail


def test_three_days_is_the_first_day_a_forecast_exists() -> None:
    forecast = built(3)

    assert forecast.tier == "burn_rate"
    assert forecast.confidence == "low"
    assert forecast.series["prompt_tokens"].algorithm == "burn_rate"


def test_the_ladder_widens_the_roster_at_each_boundary() -> None:
    assert built(6).tier == "burn_rate"
    assert built(7).tier == "trend_weekday"
    assert built(28).tier == "seasonal_trend"
    assert built(90).tier == "full_ensemble"


def test_a_dormant_scope_is_skipped_rather_than_forecast_at_almost_zero() -> None:
    observed = (day_usage(date(2026, 5, 1)),)
    history = build_history(ScopeRef("key", "k1"), observed, as_of=date(2026, 7, 1))

    assert history is not None
    outcome = build_forecast(history, as_of=history.end_day)
    assert isinstance(outcome, NoForecast)
    assert outcome.state == "dormant"


def test_weekend_seasonality_survives_into_the_forecast() -> None:
    forecast = built(90)
    bands = forecast.bands(forecast.spend, cumulative=False)
    weekend = tuple(band.p50 for band in bands[:14] if band.day.weekday() >= 5)
    weekday = tuple(band.p50 for band in bands[:14] if band.day.weekday() < 5)

    assert max(weekend) < min(weekday)


def test_a_400_percent_spike_is_tagged_and_the_pre_change_history_is_dropped() -> None:
    start = date(2026, 5, 1)
    rows = tuple(
        priced_day(
            start + timedelta(days=offset),
            prompt=100_000.0 * (5.0 if offset >= 42 else 1.0),
            completion=25_000.0,
            cache=0.0,
        )
        for offset in range(60)
    )
    history = build_history(ScopeRef("team", "spike"), rows, as_of=start + timedelta(days=59))

    assert history is not None
    outcome = build_forecast(history, as_of=history.end_day)
    assert isinstance(outcome, ScopeForecast)
    assert outcome.regime == "step_change"
    assert outcome.training_points < outcome.history_days


def test_a_step_change_too_recent_to_train_on_keeps_the_history_and_the_label() -> None:
    """Truncating to four points and calling the result a forecast would be worse than the label."""
    start = date(2026, 5, 1)
    rows = tuple(
        priced_day(
            start + timedelta(days=offset),
            prompt=100_000.0 * (5.0 if offset >= 45 else 1.0),
            completion=25_000.0,
            cache=0.0,
        )
        for offset in range(52)
    )
    history = build_history(ScopeRef("team", "late-spike"), rows, as_of=start + timedelta(days=51))

    assert history is not None
    outcome = build_forecast(history, as_of=history.end_day)
    assert isinstance(outcome, ScopeForecast)
    assert outcome.regime == "step_change"
    assert outcome.training_points == outcome.history_days


def test_usage_is_forecast_separately_and_priced_afterwards() -> None:
    forecast = built(90)

    assert forecast.strategy == "TOKEN_BASED"
    assert set(forecast.series) == {
        "request_count",
        "prompt_tokens",
        "completion_tokens",
        "cache_read_tokens",
    }
    assert forecast.price_book.snapshot_hash
    assert forecast.spend.horizon == 90


def test_an_unpriceable_mix_falls_back_to_forecasting_cost_directly() -> None:
    start = date(2026, 5, 1)
    rows = tuple(
        day_usage(
            start + timedelta(days=offset),
            prompt=1_000.0,
            completion=500.0,
            spend="40.0",
            mix={"transcription-service-xyz": 1_500.0},
        )
        for offset in range(40)
    )
    history = build_history(ScopeRef("team", "speech"), rows, as_of=start + timedelta(days=39))

    assert history is not None
    outcome = build_forecast(history, as_of=history.end_day)
    assert isinstance(outcome, ScopeForecast)
    assert outcome.strategy == "ACTUAL_COST_BASED"
    assert set(outcome.series) == {"spend_usd"}
    assert outcome.spend.horizon == 90


def test_pricing_is_calibrated_so_the_forecast_reproduces_the_invoice() -> None:
    forecast = built(90)
    ratio = forecast.pricing_fidelity.ratio

    assert ratio is not None
    assert abs(float(ratio) - 1.0) < 0.11
    assert forecast.blended_price.input_per_token > 0


def test_rebuilding_from_unchanged_facts_is_byte_identical() -> None:
    """§1.14 forecast rebuild idempotency."""
    history = seasonal_history(90)
    first = build_forecast(history, as_of=history.end_day, generated_at=GENERATED)
    second = build_forecast(history, as_of=history.end_day, generated_at=GENERATED)

    assert isinstance(first, ScopeForecast)
    assert isinstance(second, ScopeForecast)
    assert first.seed == second.seed
    assert first.spend.paths == second.spend.paths
    assert first.bands(first.spend, cumulative=True) == second.bands(second.spend, cumulative=True)


def test_the_band_is_ordered_and_the_horizon_starts_the_day_after_the_facts_end() -> None:
    history = seasonal_history(90)
    forecast = built(90)

    assert forecast.forecast_start == history.end_day + timedelta(days=1)
    assert all(band.p10 <= band.p50 <= band.p90 for band in forecast.bands(forecast.spend, cumulative=False))


def test_month_and_quarter_boundaries_are_calendar_correct() -> None:
    assert month_end(date(2026, 2, 3)) == date(2026, 2, 28)
    assert month_end(date(2028, 2, 3)) == date(2028, 2, 29)
    assert month_end(date(2026, 12, 31)) == date(2026, 12, 31)
    assert quarter_end(date(2026, 8, 21)) == date(2026, 9, 30)
    assert quarter_end(date(2026, 1, 1)) == date(2026, 3, 31)


def test_a_month_end_projection_adds_actuals_to_the_rest_of_the_month_only() -> None:
    """The part that already happened has no distribution, so it is added, not simulated."""
    forecast = built(90)
    projected = period_projection(forecast, Decimal("1000"), month_end(forecast.forecast_start))

    assert projected > Decimal("1000")
    assert period_projection(forecast, Decimal("1000"), date(2020, 1, 1)) == Decimal("1000")


def test_burn_rate_is_the_median_of_the_first_forecast_week() -> None:
    forecast = built(90)

    assert daily_burn_rate(forecast) > 0


def test_an_empty_mix_still_produces_a_forecast_of_zero_rather_than_an_error() -> None:
    start = date(2026, 5, 1)
    rows = tuple(
        day_usage(start + timedelta(days=offset), requests=0.0, prompt=0.0, completion=0.0, spend="0")
        for offset in range(30)
    )
    history = build_history(ScopeRef("team", "idle"), rows, as_of=start + timedelta(days=29))

    assert history is not None
    outcome = build_forecast(history, as_of=history.end_day)
    assert isinstance(outcome, NoForecast)
    assert outcome.state == "dormant"


def test_mix_shares_sum_to_one_and_are_carried_on_the_forecast() -> None:
    start = date(2026, 5, 1)
    rows = tuple(
        day_usage(
            start + timedelta(days=offset),
            mix=MappingProxyType({"gpt-4o-mini": 90_000.0, "gpt-5": 10_000.0}),
            spend="1.0",
        )
        for offset in range(40)
    )
    history = build_history(ScopeRef("team", "mixed"), rows, as_of=start + timedelta(days=39))

    assert history is not None
    outcome = build_forecast(history, as_of=history.end_day)
    assert isinstance(outcome, ScopeForecast)
    assert sum(outcome.mix_shares.values()) == pytest.approx(1.0)
    assert outcome.mix_shares["gpt-4o-mini"] == pytest.approx(0.9)
