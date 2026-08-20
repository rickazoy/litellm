"""The residual bootstrap, and the properties runway depends on."""

from datetime import date

import pytest

from litellm.proxy.witos.finops.simulation import (
    P10,
    P50,
    P90,
    simulate,
    stable_seed,
    weighted_sum,
)

START = date(2026, 8, 1)


def test_the_same_inputs_produce_the_same_paths() -> None:
    """A nightly rebuild from unchanged facts must not move the published P90."""
    points = {"a": (10.0, 10.0, 10.0)}
    residuals = {"a": (-2.0, -1.0, 0.0, 1.0, 2.0)}
    first = simulate(points, residuals, start_day=START, seed=7, path_count=100)
    second = simulate(points, residuals, start_day=START, seed=7, path_count=100)

    assert first["a"].paths == second["a"].paths


def test_a_different_seed_produces_different_paths() -> None:
    points = {"a": (10.0,) * 5}
    residuals = {"a": tuple(float(step) for step in range(-5, 6))}
    first = simulate(points, residuals, start_day=START, seed=1, path_count=100)
    second = simulate(points, residuals, start_day=START, seed=2, path_count=100)

    assert first["a"].paths != second["a"].paths


def test_draws_are_shared_across_series_so_errors_stay_correlated() -> None:
    """Sampling each series independently would understate the variance of their sum."""
    points = {"prompt": (100.0,) * 4, "completion": (100.0,) * 4}
    residuals = {"prompt": (-50.0, 50.0), "completion": (-50.0, 50.0)}
    simulated = simulate(points, residuals, start_day=START, seed=3, path_count=50)

    assert simulated["prompt"].paths == simulated["completion"].paths


def test_simulation_refuses_series_whose_residuals_do_not_line_up() -> None:
    with pytest.raises(ValueError, match="residual count"):
        simulate(
            {"a": (1.0,), "b": (1.0,)},
            {"a": (0.1, 0.2), "b": (0.1,)},
            start_day=START,
            seed=1,
            path_count=5,
        )


def test_paths_never_go_negative() -> None:
    simulated = simulate({"a": (5.0, 5.0)}, {"a": (-100.0, -100.0)}, start_day=START, seed=1, path_count=10)

    assert min(value for path in simulated["a"].paths for value in path) == 0.0


def test_no_residuals_means_no_band_rather_than_a_fabricated_one() -> None:
    simulated = simulate({"a": (5.0, 5.0)}, {"a": ()}, start_day=START, seed=1, path_count=10)
    low, mid, high = simulated["a"].daily_quantiles((P10, P50, P90))

    assert simulated["a"].has_band is False
    assert low == mid == high == (5.0, 5.0)


def test_quantiles_are_ordered_and_cumulative_widens_with_the_horizon() -> None:
    simulated = simulate(
        {"a": (10.0,) * 30},
        {"a": tuple(float(step) for step in range(-5, 6))},
        start_day=START,
        seed=11,
        path_count=500,
    )
    low, mid, high = simulated["a"].cumulative_quantiles((P10, P50, P90))

    assert all(l <= m <= h for l, m, h in zip(low, mid, high, strict=True))
    assert (high[-1] - low[-1]) > (high[0] - low[0])


def test_crossing_offsets_distinguish_never_from_the_last_day() -> None:
    simulated = simulate({"a": (10.0,) * 5}, {"a": (0.0,)}, start_day=START, seed=1, path_count=4)
    reachable = simulated["a"].crossing_offsets(25.0)
    unreachable = simulated["a"].crossing_offsets(10_000.0)

    assert set(reachable) == {2}
    assert set(unreachable) == {None}


def test_probability_of_crossing_is_a_share_of_all_paths() -> None:
    simulated = simulate(
        {"a": (10.0,) * 10},
        {"a": (-10.0, 10.0)},
        start_day=START,
        seed=5,
        path_count=200,
    )

    assert 0.0 <= simulated["a"].probability_of_crossing_by(4, 60.0) <= 1.0
    assert simulated["a"].probability_of_crossing_by(9, 0.0) == 1.0


def test_weighted_sum_prices_inside_each_path() -> None:
    """Pricing the quantiles instead would assume the worst case of each series lands together."""
    simulated = simulate(
        {"prompt": (100.0, 100.0), "completion": (10.0, 10.0)},
        {"prompt": (0.0,), "completion": (0.0,)},
        start_day=START,
        seed=1,
        path_count=3,
    )
    priced = weighted_sum((simulated["prompt"], simulated["completion"]), ((2.0, 2.0), (3.0, 3.0)))

    assert priced.paths[0] == (230.0, 230.0)


def test_weighted_sum_adds_a_flat_daily_amount() -> None:
    simulated = simulate({"a": (1.0, 1.0)}, {"a": (0.0,)}, start_day=START, seed=1, path_count=2)
    priced = weighted_sum((simulated["a"],), ((1.0, 1.0),), addend=(5.0, 7.0))

    assert priced.paths[0] == (6.0, 8.0)


def test_stable_seed_depends_only_on_what_the_forecast_is_of() -> None:
    assert stable_seed("team:a", "hash") == stable_seed("team:a", "hash")
    assert stable_seed("team:a", "hash") != stable_seed("team:b", "hash")
