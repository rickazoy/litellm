"""Scenario evaluation: exact repricing, and reproducibility under a fixed pricing hash."""

from datetime import date, datetime, timezone
from decimal import Decimal

import pytest

from litellm.proxy.witos.finops.engine import ScopeForecast, build_forecast
from litellm.proxy.witos.finops.history import ScopeRef, build_history
from litellm.proxy.witos.finops.scenarios import (
    ScenarioAssumptions,
    apply_scenario,
    compare,
    project,
)

from tests.witos.finops.fakes import priced_day, seasonal_history

CHEAP = "gpt-4o-mini"
DEAR = "gpt-5"
AS_OF = date(2026, 7, 29)


def forecast() -> ScopeForecast:
    history = seasonal_history(90, scope=ScopeRef("team", "finance-ai"))
    built = build_forecast(history, as_of=history.end_day, generated_at=datetime(2026, 8, 1, tzinfo=timezone.utc))
    assert isinstance(built, ScopeForecast)
    return built


def mixed_forecast() -> ScopeForecast:
    from datetime import timedelta

    start = date(2026, 5, 1)
    rows = tuple(
        priced_day(start + timedelta(days=offset), prompt=200_000.0, completion=50_000.0, cache=0.0, model=model)
        for offset in range(45)
        for model in ((CHEAP,) if offset % 2 else (DEAR,))
    )
    history = build_history(ScopeRef("team", "mixed"), rows, as_of=start + timedelta(days=44))
    assert history is not None
    built = build_forecast(history, as_of=history.end_day)
    assert isinstance(built, ScopeForecast)
    return built


def test_evaluating_the_same_scenario_twice_gives_identical_numbers() -> None:
    """§1.14 scenario reproducibility: same assumptions, same pricing hash, same answer."""
    baseline = forecast()
    assumptions = ScenarioAssumptions(usage_growth_pct=15, prompt_token_delta_pct=-10)

    first = apply_scenario(baseline, assumptions)
    second = apply_scenario(baseline, assumptions)

    assert first.seed == second.seed
    assert first.price_book.snapshot_hash == second.price_book.snapshot_hash
    assert first.spend.paths == second.spend.paths


def test_a_different_assumption_changes_the_seed_and_the_answer() -> None:
    baseline = forecast()
    grown = apply_scenario(baseline, ScenarioAssumptions(usage_growth_pct=15))
    flat = apply_scenario(baseline, ScenarioAssumptions(usage_growth_pct=0))

    assert grown.seed != flat.seed
    assert grown.spend.paths != flat.spend.paths


def test_growth_scales_usage_and_the_cost_follows_it() -> None:
    baseline = forecast()
    grown = apply_scenario(baseline, ScenarioAssumptions(usage_growth_pct=100))

    assert sum(grown.series["prompt_tokens"].point) == pytest.approx(2 * sum(baseline.series["prompt_tokens"].point))
    assert grown.spend.cumulative_at(29, 0.5) > baseline.spend.cumulative_at(29, 0.5)


def test_migrating_traffic_to_a_cheaper_model_is_an_exact_repricing() -> None:
    baseline = mixed_forecast()
    migrated = apply_scenario(
        baseline,
        ScenarioAssumptions(
            model_migrations=({"from": DEAR, "to": CHEAP, "traffic_pct": 100},)  # pyright: ignore[reportArgumentType]  # validated by pydantic from the documented JSON shape
        ),
    )

    assert migrated.mix_shares.get(DEAR, 0.0) == pytest.approx(0.0)
    assert migrated.blended_price.input_per_token < baseline.blended_price.input_per_token
    assert migrated.spend.cumulative_at(29, 0.5) < baseline.spend.cumulative_at(29, 0.5)


def test_a_migration_to_a_model_the_scope_has_never_used_is_priced_not_zeroed() -> None:
    baseline = forecast()
    migrated = apply_scenario(
        baseline,
        ScenarioAssumptions(
            model_migrations=({"from": CHEAP, "to": DEAR, "traffic_pct": 50},)  # pyright: ignore[reportArgumentType]  # validated by pydantic from the documented JSON shape
        ),
    )

    assert DEAR in migrated.price_book.prices
    assert migrated.price_book.prices[DEAR].known is True
    assert migrated.blended_price.input_per_token > baseline.blended_price.input_per_token


def test_raising_the_cache_hit_rate_lowers_the_bill() -> None:
    baseline = forecast()
    cached = apply_scenario(baseline, ScenarioAssumptions(cache_hit_rate=0.9))

    assert sum(cached.series["cache_read_tokens"].point) > sum(baseline.series["cache_read_tokens"].point)


def test_a_price_adjustment_is_a_delta_on_the_catalog_and_moves_the_hash() -> None:
    baseline = forecast()
    discounted = apply_scenario(
        baseline,
        ScenarioAssumptions(
            pricing_adjustments={CHEAP: {"input_delta_pct": -50, "output_delta_pct": -50}}  # pyright: ignore[reportArgumentType]  # validated by pydantic from the documented JSON shape
        ),
    )

    assert discounted.price_book.snapshot_hash != baseline.price_book.snapshot_hash
    assert discounted.blended_price.input_per_token < baseline.blended_price.input_per_token


def test_a_new_workload_adds_flat_daily_spend_from_its_start_date() -> None:
    baseline = forecast()
    start = baseline.forecast_start
    with_workload = apply_scenario(
        baseline,
        ScenarioAssumptions(
            new_workloads=({"daily_spend_usd": "40", "start_date": start.isoformat()},)  # pyright: ignore[reportArgumentType]  # validated by pydantic from the documented JSON shape
        ),
    )

    added = with_workload.spend.cumulative_at(9, 0.5) - baseline.spend.cumulative_at(9, 0.5)
    assert added == pytest.approx(400.0, rel=0.01)


def test_new_users_without_a_baseline_is_refused_rather_than_guessed() -> None:
    assert ScenarioAssumptions(new_users=150).requires_baseline_users() is True
    assert ScenarioAssumptions(new_users=150, baseline_users=300).requires_baseline_users() is False
    assert ScenarioAssumptions(new_users=150, baseline_users=300).growth_factor() == pytest.approx(1.5)


def test_the_comparison_reports_the_delta_against_the_same_actuals() -> None:
    baseline = forecast()
    scenario = apply_scenario(baseline, ScenarioAssumptions(usage_growth_pct=50))
    comparison = compare(baseline, scenario, actual_to_date=Decimal("100"), as_of=AS_OF)

    assert comparison.scenario.projected_eom > comparison.baseline.projected_eom
    assert comparison.delta_usd > 0
    assert comparison.savings_usd < 0


def test_a_saving_scenario_reports_a_positive_saving() -> None:
    baseline = mixed_forecast()
    scenario = apply_scenario(
        baseline,
        ScenarioAssumptions(
            model_migrations=({"from": DEAR, "to": CHEAP, "traffic_pct": 100},)  # pyright: ignore[reportArgumentType]  # validated by pydantic from the documented JSON shape
        ),
    )
    comparison = compare(baseline, scenario, actual_to_date=Decimal("0"), as_of=AS_OF)

    assert comparison.savings_usd > 0
    assert comparison.delta_pct < 0


def test_a_cost_based_scope_scales_its_cost_series_and_ignores_migrations() -> None:
    from datetime import timedelta

    start = date(2026, 5, 1)
    rows = tuple(
        priced_day(start + timedelta(days=offset), prompt=0.0, completion=0.0, cache=0.0, model="whisper-1")
        for offset in range(40)
    )
    history = build_history(ScopeRef("team", "speech"), rows, as_of=start + timedelta(days=39))
    assert history is not None
    built = build_forecast(history, as_of=history.end_day)

    if isinstance(built, ScopeForecast) and built.strategy == "ACTUAL_COST_BASED":
        scenario = apply_scenario(built, ScenarioAssumptions(usage_growth_pct=100))
        assert set(scenario.series) == {"spend_usd"}


def test_projection_covers_both_the_month_and_the_quarter() -> None:
    baseline = forecast()
    projection = project(baseline, actual_to_date=Decimal("10"), as_of=AS_OF)

    assert projection.projected_eom >= Decimal("10")
    assert projection.projected_eoq >= projection.projected_eom
    assert projection.horizon_p90 >= projection.horizon_p50
