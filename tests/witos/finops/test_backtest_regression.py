"""Backtest-regression gate (blueprint §1.6).

Six synthetic series, each shaped like a workload this system actually meets, are
scored by the same competition the engine runs and checked against committed
baselines. A change to a model, to the fold plan, or to the selection rule that
makes any of them more than two WAPE points worse fails here rather than in
somebody's month-end report.

The baselines were generated with the optional ``statsmodels`` extra absent, so
they are a floor: an environment that has it can only do better, never worse.
Regenerate with ``python -m tests.witos.finops.test_backtest_regression`` and
commit the diff *with the reason in the commit message*, because a moved baseline
is a claim that the engine got better.
"""

import json
from collections.abc import Mapping
from datetime import date
from math import sin
from pathlib import Path

import pytest

from litellm.proxy.witos.finops.backtest import run_competition
from litellm.proxy.witos.finops.forecast_models import TrainingWindow, roster_for_tier
from litellm.proxy.witos.finops.history import classify_history

BASELINE_PATH = Path(__file__).with_name("backtest_baselines.json")

# Two WAPE points, as an absolute difference. WAPE is already a ratio, so this is
# the same "2 points" the blueprint asks for.
MAX_REGRESSION = 0.02

MONDAY = date(2026, 5, 4)


def _weekend(offset: int) -> bool:
    return (MONDAY.toordinal() + offset - MONDAY.toordinal()) % 7 in (5, 6)


def steady_growth(days: int) -> tuple[float, ...]:
    return tuple(1000.0 * 1.01**offset for offset in range(days))


def weekend_seasonal(days: int) -> tuple[float, ...]:
    return tuple(1000.0 * (0.35 if _weekend(offset) else 1.0) for offset in range(days))


def seasonal_growth(days: int) -> tuple[float, ...]:
    return tuple(1000.0 * 1.008**offset * (0.4 if _weekend(offset) else 1.0) for offset in range(days))


def spiky(days: int) -> tuple[float, ...]:
    return tuple(1000.0 + 400.0 * sin(offset / 3.0) + (900.0 if offset % 11 == 0 else 0.0) for offset in range(days))


def step_change(days: int) -> tuple[float, ...]:
    return tuple(500.0 if offset < days * 2 // 3 else 2200.0 for offset in range(days))


def zero_heavy(days: int) -> tuple[float, ...]:
    return tuple(0.0 if offset % 3 else 900.0 for offset in range(days))


SERIES = {
    "steady_growth": steady_growth,
    "weekend_seasonal": weekend_seasonal,
    "seasonal_growth": seasonal_growth,
    "spiky": spiky,
    "step_change": step_change,
    "zero_heavy": zero_heavy,
}

DAYS = 90


def measure(name: str) -> tuple[str, float]:
    values = SERIES[name](DAYS)
    window = TrainingWindow(start_day=MONDAY, values=values)
    competition = run_competition(window, roster_for_tier(classify_history(DAYS)))
    score = competition.winning_score
    assert score is not None, name
    assert score.wape is not None, name
    return score.name, round(score.wape, 6)


def load_baselines() -> Mapping[str, Mapping[str, object]]:
    return json.loads(BASELINE_PATH.read_text())


@pytest.mark.parametrize("name", sorted(SERIES))
def test_wape_has_not_regressed(name: str) -> None:
    baseline = load_baselines()[name]
    model, wape = measure(name)

    assert wape <= float(baseline["wape"]) + MAX_REGRESSION, (  # pyright: ignore[reportArgumentType]  # baseline is this module's own JSON
        f"{name}: WAPE {wape:.4f} regressed past {baseline['wape']} + {MAX_REGRESSION} "
        f"(winner {model}, was {baseline['model']})"
    )


def test_every_series_has_a_committed_baseline() -> None:
    assert set(load_baselines()) == set(SERIES)


def test_selection_is_deterministic() -> None:
    assert measure("weekend_seasonal") == measure("weekend_seasonal")


def _regenerate() -> None:
    baselines = {name: dict(zip(("model", "wape"), measure(name), strict=True)) for name in sorted(SERIES)}
    BASELINE_PATH.write_text(json.dumps(baselines, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    _regenerate()
