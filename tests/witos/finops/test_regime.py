"""Regime detection: the 400% spike, and everything it must not fire on."""

from litellm.proxy.witos.finops.regime import (
    MIN_REGIME_POINTS,
    STEP_CHANGE_Z,
    detect_regime,
    training_start,
)


def flat(days: int, level: float = 100.0) -> tuple[float, ...]:
    return (level,) * days


def noisy(days: int, level: float = 100.0) -> tuple[float, ...]:
    return tuple(level + (5.0 if step % 2 else -5.0) for step in range(days))


def test_a_400_percent_spike_is_a_step_change() -> None:
    """§1.14's regime case: the last week is four times the weeks before it."""
    series = noisy(30) + (500.0, 505.0, 495.0, 500.0, 510.0, 490.0, 500.0)
    assessment = detect_regime(series)

    assert assessment.regime == "step_change"
    assert assessment.robust_z >= STEP_CHANGE_Z
    assert assessment.change_point is not None


def test_a_collapse_is_also_a_step_change() -> None:
    series = noisy(30) + (2.0,) * 7
    assessment = detect_regime(series)

    assert assessment.regime == "step_change"
    assert assessment.robust_z <= -STEP_CHANGE_Z


def test_steady_growth_is_not_a_step_change() -> None:
    """A CUSUM used as the classifier would label every growing workload a step, nightly."""
    series = tuple(100.0 * 1.02**step for step in range(90))
    assessment = detect_regime(series)

    assert assessment.regime in {"stable", "accelerating"}
    assert assessment.regime != "step_change"


def test_weekend_seasonality_alone_is_not_a_regime_change() -> None:
    series = tuple(100.0 if step % 7 < 5 else 30.0 for step in range(60))
    assessment = detect_regime(series)

    assert assessment.regime != "step_change"


def test_a_flat_series_is_stable() -> None:
    assert detect_regime(flat(40)).regime == "stable"


def test_a_wildly_dispersed_series_is_labelled_volatile_rather_than_extrapolated() -> None:
    series = tuple(1.0 if step % 3 else 400.0 for step in range(40))
    assessment = detect_regime(series)

    assert assessment.regime in {"volatile", "step_change"}


def test_too_little_history_says_so_instead_of_guessing() -> None:
    assessment = detect_regime(flat(MIN_REGIME_POINTS - 1))

    assert assessment.regime == "insufficient_history"
    assert assessment.change_point is None


def test_training_is_truncated_at_the_step_only_when_enough_remains() -> None:
    series = noisy(30) + (500.0,) * 7
    assessment = detect_regime(series)

    assert training_start(assessment, length=len(series), min_points=5) == assessment.change_point
    assert training_start(assessment, length=len(series), min_points=200) == 0


def test_a_stable_regime_never_truncates() -> None:
    assessment = detect_regime(flat(40))

    assert training_start(assessment, length=40, min_points=5) == 0
