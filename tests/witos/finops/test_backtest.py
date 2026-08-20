"""Rolling-origin scoring, WAPE over MAPE, and the tie-break that favours simplicity."""

from datetime import date

import pytest

from litellm.proxy.witos.finops.backtest import (
    TIE_TOLERANCE,
    Fold,
    ModelScore,
    in_sample_residuals,
    plan_folds,
    run_competition,
    score_model,
    select_score,
)
from litellm.proxy.witos.finops.forecast_models import BurnRate, SeasonalNaive, TrainingWindow

MONDAY = date(2026, 5, 4)


def window(values: tuple[float, ...]) -> TrainingWindow:
    return TrainingWindow(start_day=MONDAY, values=values)


def score(name: str, *, wape: float | None, complexity: int, mae: float = 1.0) -> ModelScore:
    return ModelScore(name=name, complexity=complexity, wape=wape, mae=mae, bias=0.0, folds=1, residuals=(0.1,))


def test_fold_plan_walks_the_origin_forward_and_is_length_only() -> None:
    plan = plan_folds(90)

    assert plan[-1].train_end == 83
    assert all(fold.horizon == 7 for fold in plan)
    assert tuple(fold.train_end for fold in plan) == tuple(range(48, 84, 7))
    assert min(fold.train_end for fold in plan) >= 45
    assert plan_folds(90) == plan_folds(90)


def test_fold_plan_shrinks_rather_than_disappearing_on_a_short_history() -> None:
    assert plan_folds(3) == (Fold(train_end=2, horizon=1),)
    assert plan_folds(5) == (Fold(train_end=3, horizon=2),)
    assert plan_folds(2) == ()
    assert plan_folds(14) == (Fold(train_end=7, horizon=7),)


def test_a_zero_volume_day_costs_error_without_exploding_the_score() -> None:
    """The reason §1.6 scores WAPE and not MAPE: MAPE here would be infinite."""
    values = (10.0, 10.0, 0.0, 10.0, 10.0, 10.0, 0.0, 10.0)
    scored = score_model(BurnRate(lookback=2), window(values), plan_folds(len(values)))

    assert scored is not None
    assert scored.wape is not None
    assert scored.wape < 10.0


def test_wape_is_undefined_rather_than_zero_when_a_scope_did_nothing() -> None:
    scored = score_model(BurnRate(), window((0.0,) * 10), plan_folds(10))

    assert scored is not None
    assert scored.wape is None
    assert scored.mae == 0.0


def test_bias_is_signed_and_separate_from_accuracy() -> None:
    """A model that is consistently low is a specific, correctable failure."""
    rising = tuple(float(100 + 10 * step) for step in range(20))
    scored = score_model(BurnRate(lookback=3), window(rising), plan_folds(20))

    assert scored is not None
    assert scored.bias is not None
    assert scored.bias < 0


def test_a_tie_goes_to_the_simpler_model() -> None:
    winner = select_score(
        (
            score("complicated", wape=0.100, complexity=5),
            score("simple", wape=0.100 + TIE_TOLERANCE / 2, complexity=1),
        )
    )

    assert winner.name == "simple"


def test_a_real_improvement_beats_simplicity() -> None:
    winner = select_score(
        (
            score("complicated", wape=0.05, complexity=5),
            score("simple", wape=0.20, complexity=1),
        )
    )

    assert winner.name == "complicated"


def test_an_all_zero_scoreboard_falls_back_to_mae_then_complexity() -> None:
    winner = select_score((score("a", wape=None, complexity=4, mae=2.0), score("b", wape=None, complexity=2, mae=1.0)))

    assert winner.name == "b"


def test_expensive_candidates_are_not_fitted_when_the_cheap_ones_already_fit() -> None:
    flat = window((100.0,) * 60)
    competition = run_competition(flat, (BurnRate(), SeasonalNaive(), _CountingExpensive()))

    assert competition.score_of("counting_expensive") is None
    assert _CountingExpensive.calls == 0


def test_expensive_candidates_do_compete_when_the_cheap_ones_do_not_fit() -> None:
    _CountingExpensive.calls = 0
    noisy = window(tuple(100.0 if step % 3 else 900.0 for step in range(60)))
    competition = run_competition(noisy, (BurnRate(), SeasonalNaive(), _CountingExpensive()))

    assert competition.score_of("counting_expensive") is not None
    assert _CountingExpensive.calls > 0


def test_in_sample_residuals_give_a_band_to_a_history_too_short_to_backtest() -> None:
    residuals = in_sample_residuals(BurnRate(), window((10.0, 20.0, 30.0, 40.0)), start=2)

    assert len(residuals) == 2
    assert residuals[0] == pytest.approx(15.0)


def test_residual_tuples_line_up_across_series_of_one_scope() -> None:
    """The joint bootstrap indexes residuals by day, so every series needs the same count."""
    plan = plan_folds(40)
    prompt = score_model(BurnRate(), window(tuple(float(step) for step in range(40))), plan)
    completion = score_model(BurnRate(), window(tuple(float(step) * 3 for step in range(40))), plan)

    assert prompt is not None
    assert completion is not None
    assert len(prompt.residuals) == len(completion.residuals)


class _CountingExpensive:
    """A stand-in for a statsmodels candidate that records whether it was ever fitted."""

    calls = 0
    name = "counting_expensive"
    complexity = 4
    min_points = 5
    expensive = True

    def available(self) -> bool:
        return True

    def predict(self, window_: TrainingWindow, horizon: int) -> tuple[float, ...]:
        type(self).calls += 1
        return (window_.values[-1],) * horizon
