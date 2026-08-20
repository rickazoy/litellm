"""Runway probabilities: the CFO sentence, and the cases that must not produce one."""

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

from litellm.proxy.witos.finops.engine import ScopeForecast, build_forecast
from litellm.proxy.witos.finops.history import ScopeRef
from litellm.proxy.witos.finops.runway import Quota, compute_runway, runway_statement

from tests.witos.finops.fakes import seasonal_history

SCOPE = ScopeRef("team", "finance-ai")
FORECAST_START = date(2026, 7, 30)


def forecast() -> ScopeForecast:
    history = seasonal_history(90, scope=SCOPE)
    built = build_forecast(history, as_of=history.end_day, generated_at=datetime(2026, 8, 1, tzinfo=timezone.utc))
    assert isinstance(built, ScopeForecast)
    return built


def quota(
    limit: str,
    *,
    metric: str = "usd",
    reset_in_days: int = 30,
    period: str = "month",
    start: date | None = None,
) -> Quota:
    period_start = start or FORECAST_START
    return Quota(
        quota_id="q1",
        scope=SCOPE,
        metric=metric,
        limit_value=Decimal(limit),
        period=period,
        period_start=datetime.combine(period_start, datetime.min.time(), tzinfo=timezone.utc),
        period_end=datetime.combine(
            period_start + timedelta(days=reset_in_days), datetime.min.time(), tzinfo=timezone.utc
        ),
        reset_at=datetime.combine(
            period_start + timedelta(days=reset_in_days), datetime.min.time(), tzinfo=timezone.utc
        ),
        source_type="litellm_budget",
    )


def test_a_generous_budget_survives_the_cycle_with_no_exhaustion_date() -> None:
    built = forecast()
    result = compute_runway(built, quota("1000000"), Decimal("10"), today=built.forecast_start)

    assert result is not None
    assert result.exhaustion_p50 is None
    assert result.prob_exhaust_before_reset == 0.0
    assert result.survives_cycle is True


def test_a_budget_that_cannot_last_produces_dates_in_the_right_order() -> None:
    built = forecast()
    burn = float(built.spend.daily_quantiles((0.5,))[0][0])
    result = compute_runway(built, quota(str(burn * 10)), Decimal(0), today=built.forecast_start)

    assert result is not None
    assert result.exhaustion_p50 is not None
    assert result.exhaustion_p90 is not None
    assert result.exhaustion_p90 <= result.exhaustion_p50, "the conservative date is the earlier one"
    assert result.prob_exhaust_before_reset > 0.5
    assert result.survives_cycle is False


def test_a_quota_that_is_already_exhausted_says_so_without_forecasting() -> None:
    built = forecast()
    result = compute_runway(built, quota("100"), Decimal("250"), today=built.forecast_start)

    assert result is not None
    assert result.remaining < 0
    assert result.prob_exhaust_before_reset == 1.0
    assert result.survives_cycle is False
    assert "already exhausted" in runway_statement(result)


def test_a_quota_that_resets_before_the_burn_catches_it_survives_the_cycle() -> None:
    """§1.14: the reset landing first is a different answer from never exhausting."""
    built = forecast()
    burn = float(built.spend.daily_quantiles((0.5,))[0][0])
    tight = quota(str(burn * 40), reset_in_days=2, start=built.forecast_start - timedelta(days=1))
    result = compute_runway(built, tight, Decimal(0), today=built.forecast_start)

    assert result is not None
    assert result.prob_exhaust_before_reset < 0.5
    assert result.survives_cycle is True


def test_probabilities_stay_inside_zero_and_one() -> None:
    built = forecast()
    burn = float(built.spend.daily_quantiles((0.5,))[0][0])

    for multiple in (0.5, 5, 20, 100, 5000):
        result = compute_runway(built, quota(str(burn * multiple)), Decimal(0), today=built.forecast_start)
        assert result is not None
        assert 0.0 <= result.prob_exhaust_before_reset <= 1.0
        assert 0.0 <= result.prob_overrun_this_period <= 1.0


def test_a_token_quota_uses_the_token_paths_not_the_cost_paths() -> None:
    built = forecast()
    token_burn = float(built.total_tokens.daily_quantiles((0.5,))[0][0])
    money_burn = float(built.spend.daily_quantiles((0.5,))[0][0])
    tokens = compute_runway(
        built, quota(str(token_burn * 5), metric="total_tokens"), Decimal(0), today=built.forecast_start
    )
    money = compute_runway(built, quota(str(money_burn * 20)), Decimal(0), today=built.forecast_start)

    assert tokens is not None
    assert money is not None
    assert tokens.exhaustion_p50 is not None
    assert money.exhaustion_p50 is not None
    assert tokens.exhaustion_p50 < money.exhaustion_p50


def test_a_metric_the_forecast_does_not_carry_returns_nothing_rather_than_zero() -> None:
    built = forecast()

    assert compute_runway(built, quota("10", metric="gpu_hours"), Decimal(0), today=built.forecast_start) is None


def test_the_statement_is_the_blueprint_sentence_built_from_the_numbers() -> None:
    built = forecast()
    burn = float(built.spend.daily_quantiles((0.5,))[0][0])
    result = compute_runway(built, quota(str(burn * 10)), Decimal(0), today=built.forecast_start)

    assert result is not None
    statement = runway_statement(result)
    assert "probability of exceeding the $" in statement
    assert "monthly budget" in statement
    assert "Median exhaustion" in statement
    assert "conservative" in statement


def test_runway_uses_the_forecast_s_own_paths() -> None:
    """Two runway computations from one forecast must agree exactly."""
    built = forecast()
    first = compute_runway(built, quota("500"), Decimal("100"), today=built.forecast_start)
    second = compute_runway(built, quota("500"), Decimal("100"), today=built.forecast_start)

    assert first == second
