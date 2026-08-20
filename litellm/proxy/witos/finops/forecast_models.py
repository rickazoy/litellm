"""The candidate roster forecasts compete on (blueprint §1.6).

Every model here answers the same question, ``given these daily values, what are
the next H days``, and none of them is trusted: which one is used for a given
scope is decided by the rolling-origin backtest in ``backtest.py``, never by an
author's intuition about the workload.

Six of the eight are implemented directly because they are a handful of lines of
arithmetic each, and depending on a library to take a median would be a heavier
dependency than the code it replaces. The two exponential-smoothing models are
maximum-likelihood fits and are delegated to ``statsmodels``, which stays an
optional extra: where it is absent those two candidates report themselves
unavailable and do not compete, and the forecast records which algorithm actually
won. Where it is present but the fit will not converge on a particular series,
the candidate degrades to its simpler sibling's numbers, which is safe only
because a tie in the backtest is resolved in favour of the simpler model, so the
degraded candidate cannot win under the sophisticated one's name.

Every model clamps at zero. Negative usage is not a forecast, it is an artifact
of fitting a line to a series that stopped.
"""

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from functools import cache, reduce
from importlib.util import find_spec
from statistics import median
from types import MappingProxyType
from typing import Final, Protocol

from litellm.proxy.witos.finops.history import HistoryTier

WEEK: Final = 7

# Theil-Sen is O(n^2) in pairs and runs inside every backtest fold of every
# competing model. Above this many points the pairs are strided rather than
# exhaustive, which keeps the estimator's breakdown point while bounding the
# forecast job's cost per scope (§1.14 budgets 150ms per scope).
_MAX_EXACT_PAIRS_POINTS: Final = 60


@dataclass(frozen=True, slots=True)
class TrainingWindow:
    """Daily values, oldest first, and the calendar day the first value belongs to.

    The calendar day is part of the input because a weekday-seasonal model that
    guesses which value was a Sunday is worse than no seasonal model at all.
    """

    start_day: date
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("A training window needs at least one value")

    @property
    def size(self) -> int:
        return len(self.values)

    def weekday_at(self, index: int) -> int:
        return (self.start_day + timedelta(days=index)).weekday()

    def future_weekday(self, step: int) -> int:
        return (self.start_day + timedelta(days=self.size + step)).weekday()


class ForecastModel(Protocol):
    """One candidate. ``complexity`` breaks backtest ties in favour of the simpler model."""

    @property
    def name(self) -> str: ...

    @property
    def complexity(self) -> int: ...

    @property
    def min_points(self) -> int: ...

    @property
    def expensive(self) -> bool:
        """True for a maximum-likelihood fit, which costs ~15ms against ~0.1ms for the rest.

        The competition uses this to decide when the extra accuracy is worth
        paying for; see ``backtest.run_competition``.
        """
        ...

    def available(self) -> bool: ...

    def predict(self, window: TrainingWindow, horizon: int) -> tuple[float, ...]: ...


def _clamp(values: Sequence[float]) -> tuple[float, ...]:
    return tuple(max(0.0, value) for value in values)


def _stride_for(count: int) -> int:
    return 1 if count <= _MAX_EXACT_PAIRS_POINTS else count // _MAX_EXACT_PAIRS_POINTS


def theil_sen(values: Sequence[float]) -> tuple[float, float]:
    """Median-of-pairwise-slopes trend, returned as ``(slope, intercept)``.

    Chosen over least squares because a single day of runaway spend moves an OLS
    line for the rest of the horizon, and in this domain that day is a Tuesday
    someone left a batch job running, not the new normal.
    """
    count: Final = len(values)
    if count < 2:
        return 0.0, values[0] if values else 0.0
    stride: Final = _stride_for(count)
    slopes: Final = tuple(
        (values[j] - values[i]) / (j - i) for i in range(0, count - 1, stride) for j in range(i + stride, count, stride)
    )
    if not slopes:
        return 0.0, median(values)
    slope: Final = median(slopes)
    return slope, median(tuple(value - slope * index for index, value in enumerate(values)))


def _weekday_factors(window: TrainingWindow, baseline: Sequence[float]) -> tuple[float, ...]:
    """Multiplicative day-of-week factors against a de-trended baseline, mean-normalised.

    Multiplicative rather than additive because weekend traffic is a fraction of
    weekday traffic, not a fixed number of tokens below it, so the factor has to
    survive the series doubling.
    """
    ratios: Final = tuple(
        tuple(
            window.values[index] / baseline[index]
            for index in range(window.size)
            if window.weekday_at(index) == weekday and baseline[index] > 0
        )
        for weekday in range(WEEK)
    )
    raw: Final = tuple(median(day) if day else 1.0 for day in ratios)
    mean_factor: Final = sum(raw) / WEEK
    return raw if mean_factor <= 0 else tuple(factor / mean_factor for factor in raw)


@dataclass(frozen=True, slots=True)
class SeasonalNaive:
    """Last week, repeated. The benchmark every other candidate has to beat."""

    period: int = WEEK
    name: str = "seasonal_naive"
    complexity: int = 1
    expensive: bool = False

    @property
    def min_points(self) -> int:
        return self.period

    def available(self) -> bool:
        return True

    def predict(self, window: TrainingWindow, horizon: int) -> tuple[float, ...]:
        base: Final = window.values[-self.period :]
        return _clamp(tuple(base[step % len(base)] for step in range(horizon)))


@dataclass(frozen=True, slots=True)
class Ewma:
    """Exponentially weighted level, held flat. No trend, no season, no opinions."""

    alpha: float = 0.3
    name: str = "ewma"
    complexity: int = 1
    min_points: int = 3
    expensive: bool = False

    def available(self) -> bool:
        return True

    def level(self, values: Sequence[float]) -> float:
        return reduce(lambda smoothed, value: self.alpha * value + (1 - self.alpha) * smoothed, values)

    def predict(self, window: TrainingWindow, horizon: int) -> tuple[float, ...]:
        return _clamp((self.level(window.values),) * horizon)


@dataclass(frozen=True, slots=True)
class BurnRate:
    """Mean of the trailing window, held flat: the only claim 3 days of history supports."""

    lookback: int = WEEK
    name: str = "burn_rate"
    complexity: int = 0
    min_points: int = 2
    expensive: bool = False

    def available(self) -> bool:
        return True

    def predict(self, window: TrainingWindow, horizon: int) -> tuple[float, ...]:
        recent: Final = window.values[-self.lookback :]
        return _clamp((sum(recent) / len(recent),) * horizon)


@dataclass(frozen=True, slots=True)
class RobustLinear:
    """Theil-Sen trend extended forward."""

    name: str = "robust_linear"
    complexity: int = 2
    min_points: int = 5
    expensive: bool = False

    def available(self) -> bool:
        return True

    def predict(self, window: TrainingWindow, horizon: int) -> tuple[float, ...]:
        slope, intercept = theil_sen(window.values)
        return _clamp(tuple(intercept + slope * (window.size + step) for step in range(horizon)))


@dataclass(frozen=True, slots=True)
class DayOfWeekSeasonalTrend:
    """Theil-Sen trend times day-of-week factors."""

    name: str = "dow_seasonal_trend"
    complexity: int = 3
    min_points: int = 14
    expensive: bool = False

    def available(self) -> bool:
        return True

    def predict(self, window: TrainingWindow, horizon: int) -> tuple[float, ...]:
        slope, intercept = theil_sen(window.values)
        trend: Final = tuple(intercept + slope * index for index in range(window.size))
        factors: Final = _weekday_factors(window, trend)
        return _clamp(
            tuple(
                (intercept + slope * (window.size + step)) * factors[window.future_weekday(step)]
                for step in range(horizon)
            )
        )


@dataclass(frozen=True, slots=True)
class RecentWeightedSeasonalTrend:
    """Exponentially weighted linear trend times day-of-week factors from recent weeks.

    The half-life is what separates this from ``DayOfWeekSeasonalTrend``: after a
    product launch the two months before it should not get an equal vote on next
    Tuesday.
    """

    half_life_days: float = 14.0
    seasonal_lookback: int = 28
    name: str = "recent_weighted_seasonal_trend"
    complexity: int = 3
    min_points: int = 14
    expensive: bool = False

    def available(self) -> bool:
        return True

    def _weighted_line(self, values: Sequence[float]) -> tuple[float, float]:
        count: Final = len(values)
        weights: Final[tuple[float, ...]] = tuple(
            0.5 ** ((count - 1 - index) / self.half_life_days) for index in range(count)
        )
        total: Final = sum(weights)
        mean_x: Final = sum(weight * index for index, weight in enumerate(weights)) / total
        mean_y: Final = sum(weight * value for weight, value in zip(weights, values, strict=True)) / total
        variance: Final = sum(weight * (index - mean_x) ** 2 for index, weight in enumerate(weights))
        if variance <= 0:
            return 0.0, mean_y
        covariance: Final = sum(
            weight * (index - mean_x) * (value - mean_y)
            for index, (weight, value) in enumerate(zip(weights, values, strict=True))
        )
        slope: Final = covariance / variance
        return slope, mean_y - slope * mean_x

    def predict(self, window: TrainingWindow, horizon: int) -> tuple[float, ...]:
        slope, intercept = self._weighted_line(window.values)
        trend: Final = tuple(intercept + slope * index for index in range(window.size))
        seasonal_window: Final = TrainingWindow(
            start_day=window.start_day + timedelta(days=max(0, window.size - self.seasonal_lookback)),
            values=window.values[-self.seasonal_lookback :],
        )
        factors: Final = _weekday_factors(seasonal_window, trend[-self.seasonal_lookback :])
        return _clamp(
            tuple(
                (intercept + slope * (window.size + step)) * factors[window.future_weekday(step)]
                for step in range(horizon)
            )
        )


@cache
def statsmodels_available() -> bool:
    return find_spec("statsmodels") is not None


def _exponential_smoothing(
    values: Sequence[float], horizon: int, *, seasonal_periods: int | None
) -> tuple[float, ...] | None:
    """Fit a damped exponential-smoothing model and forecast, or give up cleanly.

    Returning ``None`` rather than raising is deliberate: a candidate that cannot
    be fitted on this particular series has to drop out of the competition
    without taking the scope's whole forecast with it.
    """
    try:
        from statsmodels.tsa.holtwinters import (  # pyright: ignore[reportMissingTypeStubs]  # optional extra, no stubs
            ExponentialSmoothing,  # pyright: ignore[reportUnknownVariableType]  # optional extra, no stubs
        )
    except ImportError:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted: Final = ExponentialSmoothing(  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]  # untyped optional extra
                tuple(values),
                trend="add",
                damped_trend=True,
                seasonal=None if seasonal_periods is None else "add",
                seasonal_periods=seasonal_periods,
                initialization_method="estimated",
            ).fit(use_brute=False)
            predicted: Final = fitted.forecast(horizon)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]  # untyped optional extra
            return tuple(float(value) for value in predicted)  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]  # untyped optional extra
    except Exception:  # noqa: BLE001  # a candidate failing to converge must not fail the scope
        return None


@dataclass(frozen=True, slots=True)
class HoltDamped:
    """Damped-trend exponential smoothing (statsmodels), no seasonality."""

    name: str = "holt_damped"
    complexity: int = 4
    min_points: int = 10
    expensive: bool = True

    def available(self) -> bool:
        return statsmodels_available()

    def predict(self, window: TrainingWindow, horizon: int) -> tuple[float, ...]:
        predicted: Final = _exponential_smoothing(window.values, horizon, seasonal_periods=None)
        if predicted is None:
            return _clamp((window.values[-1],) * horizon)
        return _clamp(predicted)


@dataclass(frozen=True, slots=True)
class HoltWintersEts:
    """Weekly-seasonal damped ETS (statsmodels). Needs two full seasons to fit."""

    name: str = "holt_winters_ets"
    complexity: int = 5
    min_points: int = 2 * WEEK + 1
    expensive: bool = True

    def available(self) -> bool:
        return statsmodels_available()

    def predict(self, window: TrainingWindow, horizon: int) -> tuple[float, ...]:
        predicted: Final = _exponential_smoothing(window.values, horizon, seasonal_periods=WEEK)
        if predicted is None:
            return SeasonalNaive().predict(window, horizon)
        return _clamp(predicted)


@dataclass(frozen=True, slots=True)
class WeightedEnsemble:
    """Inverse-WAPE weighted blend of the candidates that survived the backtest.

    Built by the selector once scores exist, so it is the one candidate that
    cannot be instantiated before backtesting.
    """

    members: tuple[tuple[ForecastModel, float], ...]
    name: str = "weighted_ensemble"
    complexity: int = 6

    @property
    def min_points(self) -> int:
        return max(member.min_points for member, _ in self.members)

    @property
    def expensive(self) -> bool:
        return any(member.expensive for member, _ in self.members)

    def available(self) -> bool:
        return all(member.available() for member, _ in self.members)

    def member_names(self) -> tuple[str, ...]:
        return tuple(member.name for member, _ in self.members)

    def predict(self, window: TrainingWindow, horizon: int) -> tuple[float, ...]:
        total: Final = sum(weight for _, weight in self.members)
        if total <= 0:
            raise ValueError("A weighted ensemble needs at least one positive weight")
        predictions: Final = tuple(
            tuple(value * weight for value in member.predict(window, horizon)) for member, weight in self.members
        )
        return _clamp(tuple(sum(column) / total for column in zip(*predictions, strict=True)))


DEFAULT_ROSTER: Final[tuple[ForecastModel, ...]] = (
    BurnRate(),
    SeasonalNaive(),
    Ewma(),
    RobustLinear(),
    DayOfWeekSeasonalTrend(),
    RecentWeightedSeasonalTrend(),
    HoltDamped(),
    HoltWintersEts(),
)

# Blueprint §1.6's ladder, as the roster each rung may compete. The rung is a
# statement about what the history can validate, so a model that needs a season
# it has never seen is not offered the chance to win by luck.
TIER_ROSTER: Final[Mapping[HistoryTier, tuple[str, ...]]] = MappingProxyType(
    {
        "insufficient_history": (),
        "burn_rate": ("burn_rate",),
        "trend_weekday": ("burn_rate", "seasonal_naive", "ewma", "robust_linear", "holt_damped"),
        "seasonal_trend": (
            "burn_rate",
            "seasonal_naive",
            "ewma",
            "robust_linear",
            "dow_seasonal_trend",
            "recent_weighted_seasonal_trend",
            "holt_damped",
            "holt_winters_ets",
        ),
        "full_ensemble": tuple(model.name for model in DEFAULT_ROSTER),
    }
)


def roster_for_tier(tier: HistoryTier, roster: Sequence[ForecastModel] = DEFAULT_ROSTER) -> tuple[ForecastModel, ...]:
    """The candidates this rung of the ladder allows, minus any whose extra is absent."""
    allowed: Final = TIER_ROSTER[tier]
    return tuple(model for model in roster if model.name in allowed and model.available())
