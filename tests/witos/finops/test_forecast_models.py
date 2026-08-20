"""The candidate roster: what each model claims, and what it refuses to claim."""

from datetime import date, timedelta

import pytest

from litellm.proxy.witos.finops.forecast_models import (
    DEFAULT_ROSTER,
    BurnRate,
    DayOfWeekSeasonalTrend,
    Ewma,
    HoltWintersEts,
    RecentWeightedSeasonalTrend,
    RobustLinear,
    SeasonalNaive,
    TrainingWindow,
    WeightedEnsemble,
    roster_for_tier,
    statsmodels_available,
    theil_sen,
)

MONDAY = date(2026, 5, 4)


def window(values: tuple[float, ...], start: date = MONDAY) -> TrainingWindow:
    return TrainingWindow(start_day=start, values=values)


def test_seasonal_naive_repeats_the_last_week_aligned_to_weekday() -> None:
    week = (10.0, 11.0, 12.0, 13.0, 14.0, 2.0, 1.0)
    predicted = SeasonalNaive().predict(window(week), 9)

    assert predicted[:7] == week
    assert predicted[7:] == week[:2]


def test_burn_rate_is_the_trailing_mean_and_nothing_else() -> None:
    predicted = BurnRate(lookback=3).predict(window((1.0, 2.0, 30.0, 40.0, 50.0)), 3)

    assert predicted == (40.0, 40.0, 40.0)


def test_ewma_weights_recent_days_more_heavily() -> None:
    predicted = Ewma(alpha=0.5).predict(window((0.0, 0.0, 0.0, 100.0)), 1)

    assert predicted[0] == pytest.approx(50.0)


def test_theil_sen_ignores_a_single_runaway_day() -> None:
    clean = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0)
    spiked = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 5000.0)

    assert theil_sen(clean)[0] == pytest.approx(10.0)
    assert theil_sen(spiked)[0] == pytest.approx(10.0)


def test_models_never_forecast_negative_usage() -> None:
    collapsing = window(tuple(float(max(0, 100 - 12 * step)) for step in range(14)))

    for model in (RobustLinear(), DayOfWeekSeasonalTrend(), RecentWeightedSeasonalTrend()):
        assert min(model.predict(collapsing, 30)) >= 0.0, model.name


def test_weekday_seasonal_model_reproduces_the_weekend_dip() -> None:
    values = tuple(300.0 if (MONDAY + timedelta(days=offset)).weekday() >= 5 else 1000.0 for offset in range(28))
    predicted = DayOfWeekSeasonalTrend().predict(window(values), 7)

    weekend = tuple(predicted[index] for index in (5, 6))
    weekday = tuple(predicted[index] for index in range(5))
    assert max(weekend) < min(weekday)


def test_recent_weighted_model_follows_a_level_shift_faster_than_the_plain_trend() -> None:
    values = tuple([100.0] * 21 + [400.0] * 7)
    recent = RecentWeightedSeasonalTrend().predict(window(values), 1)[0]
    plain = DayOfWeekSeasonalTrend().predict(window(values), 1)[0]

    assert recent > plain


def test_ensemble_is_the_inverse_wape_weighted_blend_of_its_members() -> None:
    ensemble = WeightedEnsemble(members=((BurnRate(lookback=1), 3.0), (SeasonalNaive(), 1.0)))
    values = (10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 100.0)
    predicted = ensemble.predict(window(values), 1)[0]

    assert predicted == pytest.approx((100.0 * 3 + 10.0 * 1) / 4)


def test_tier_roster_widens_with_history_and_never_offers_an_unfittable_model() -> None:
    burn = tuple(model.name for model in roster_for_tier("burn_rate"))
    trend = tuple(model.name for model in roster_for_tier("trend_weekday"))
    full = tuple(model.name for model in roster_for_tier("full_ensemble"))

    assert burn == ("burn_rate",)
    assert "dow_seasonal_trend" not in trend
    assert "dow_seasonal_trend" in full
    assert set(trend).issubset(set(full))


def test_statsmodels_candidates_drop_out_when_the_extra_is_absent() -> None:
    expected = {"holt_damped", "holt_winters_ets"} if statsmodels_available() else set()
    available = {model.name for model in DEFAULT_ROSTER if model.expensive and model.available()}

    assert available == expected


@pytest.mark.skipif(not statsmodels_available(), reason="statsmodels extra is not installed")
def test_ets_produces_a_finite_seasonal_forecast() -> None:
    values = tuple(
        1000.0 + 10 * offset - (600.0 if (MONDAY + timedelta(days=offset)).weekday() >= 5 else 0.0)
        for offset in range(56)
    )
    predicted = HoltWintersEts().predict(window(values), 14)

    assert len(predicted) == 14
    assert all(value >= 0 for value in predicted)
    assert max(predicted) < 10_000
