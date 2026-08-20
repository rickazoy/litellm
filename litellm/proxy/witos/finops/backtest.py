"""Rolling-origin backtest and model selection (blueprint §1.6).

Selection is empirical. Every candidate the history's rung of the ladder allows
is trained on a prefix and scored on the days that follow it, walking the origin
forward, and the one that was actually most accurate on this scope's own history
wins. No model is the default.

**WAPE, not MAPE.** MAPE divides by the actual, and in this domain the actual is
frequently zero: a team that made no calls on Sunday, a key used only during a
migration, a new service in its first week. One zero-volume day makes MAPE
infinite and a handful of near-zero days make it enormous, so a MAPE-ranked
competition is decided by which model happened to predict smallest on the
quietest day. WAPE divides the summed absolute error by the summed actual, so
zero days contribute error without exploding the denominator, and the score keeps
the units of the thing being forecast.

Bias is scored separately and never folded into the ranking. A model that is
accurate but consistently 8% low is a specific, correctable failure, and a CFO
needs to be told about it rather than have it averaged away.

Ties go to the simpler model. Two candidates within half a point of WAPE are not
distinguishable on this much data, and the simpler one will not surprise anyone
next month.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from litellm.proxy.witos.finops.forecast_models import ForecastModel, TrainingWindow, WeightedEnsemble

DEFAULT_HORIZON: Final = 7
DEFAULT_STEP: Final = 7

# Folds are capped at the most recent 8 origins: a year of history would
# otherwise spend 48 fits per candidate scoring a model on traffic patterns from
# two quarters ago, which is both the slowest and the least relevant evidence.
DEFAULT_MAX_FOLDS: Final = 8
MIN_TRAIN_POINTS: Final = 2

# Half a WAPE point. Below this two candidates are indistinguishable on a series
# this short, and the tie-break belongs to whichever is easier to reason about.
TIE_TOLERANCE: Final = 0.005

# Under this many residuals a bootstrap resamples the same two or three numbers
# and produces a confidence band that is an artifact of the sample size.
MIN_BOOTSTRAP_RESIDUALS: Final = 5

ENSEMBLE_MEMBERS: Final = 3
_ENSEMBLE_WEIGHT_FLOOR: Final = 1e-6

# WAPE below which the maximum-likelihood candidates are not fitted at all. See
# `_worth_paying_for`: this is the forecast job's cost boundary, in the one place
# where it is visible and adjustable.
EXPENSIVE_WAPE_GATE: Final = 0.05


@dataclass(frozen=True, slots=True)
class Fold:
    """One rolling origin: train on ``[0, train_end)``, score the next ``horizon`` days."""

    train_end: int
    horizon: int


@dataclass(frozen=True, slots=True)
class ModelScore:
    """How one candidate did, and the residuals its bands would be drawn from."""

    name: str
    complexity: int
    wape: float | None
    mae: float
    bias: float | None
    folds: int
    residuals: tuple[float, ...]

    @property
    def rank_wape(self) -> float:
        return float("inf") if self.wape is None else self.wape


@dataclass(frozen=True, slots=True)
class Competition:
    """The full scoreboard plus the candidate that won it."""

    scores: tuple[ModelScore, ...]
    winner: ForecastModel
    winning_score: ModelScore | None

    def score_of(self, name: str) -> ModelScore | None:
        matched: Final = tuple(score for score in self.scores if score.name == name)
        return matched[0] if matched else None


def plan_folds(
    length: int,
    *,
    horizon: int = DEFAULT_HORIZON,
    step: int = DEFAULT_STEP,
    max_folds: int = DEFAULT_MAX_FOLDS,
    min_train: int = MIN_TRAIN_POINTS,
) -> tuple[Fold, ...]:
    """Rolling origins for a series of ``length`` days, oldest fold first.

    The plan depends only on the length, so every series in a scope and every
    candidate competing on it is scored over exactly the same days. That is what
    makes the residuals from different series line up, which the joint bootstrap
    in ``simulation.py`` depends on.

    No origin trains on less than half the history. Walking further back would
    buy a couple of extra folds at the price of scoring a weekly-seasonal
    candidate on a four-day prefix it cannot be fitted to, which does not lower
    its score, it removes it from the competition entirely.
    """
    if length < min_train + 1:
        return ()
    first_end: Final = max(min_train, (length + 1) // 2)
    fold_horizon: Final = min(horizon, length - first_end)
    if fold_horizon < 1:
        return ()
    last_end: Final = length - fold_horizon
    ends: Final = tuple(end for end in range(last_end, first_end - 1, -step))[:max_folds]
    return tuple(Fold(train_end=end, horizon=min(fold_horizon, length - end)) for end in reversed(ends))


def _fold_predictions(model: ForecastModel, window: TrainingWindow, fold: Fold) -> tuple[float, ...]:
    train: Final = TrainingWindow(start_day=window.start_day, values=window.values[: fold.train_end])
    return model.predict(train, fold.horizon)


def score_model(model: ForecastModel, window: TrainingWindow, folds: Sequence[Fold]) -> ModelScore | None:
    """Score one candidate over the fold plan, or ``None`` if it cannot compete on it."""
    if not folds or folds[0].train_end < model.min_points:
        return None
    residuals: Final = tuple(
        actual - predicted
        for fold in folds
        for actual, predicted in zip(
            window.values[fold.train_end : fold.train_end + fold.horizon],
            _fold_predictions(model, window, fold),
            strict=True,
        )
    )
    actuals: Final = tuple(
        value for fold in folds for value in window.values[fold.train_end : fold.train_end + fold.horizon]
    )
    absolute_actual: Final = sum(abs(value) for value in actuals)
    absolute_error: Final = sum(abs(residual) for residual in residuals)
    return ModelScore(
        name=model.name,
        complexity=model.complexity,
        wape=(absolute_error / absolute_actual) if absolute_actual > 0 else None,
        mae=absolute_error / len(residuals),
        bias=(-sum(residuals) / absolute_actual) if absolute_actual > 0 else None,
        folds=len(folds),
        residuals=residuals,
    )


def select_score(scores: Sequence[ModelScore]) -> ModelScore:
    """Lowest WAPE wins; anything within ``TIE_TOLERANCE`` of it is a tie the simpler model takes."""
    if not scores:
        raise ValueError("Cannot select a winner from an empty scoreboard")
    scorable: Final = tuple(score for score in scores if score.wape is not None)
    if not scorable:
        return min(scores, key=lambda score: (score.mae, score.complexity))
    best: Final = min(score.rank_wape for score in scorable)
    contenders: Final = tuple(score for score in scorable if score.rank_wape <= best + TIE_TOLERANCE)
    return min(contenders, key=lambda score: (score.complexity, score.rank_wape))


def _ensemble_of(
    models: Sequence[ForecastModel], scores: Sequence[ModelScore], members: int
) -> WeightedEnsemble | None:
    """Inverse-WAPE weighted blend of the best few candidates, if there are a few."""
    by_name: Final = MappingProxyType({score.name: score for score in scores if score.wape is not None})
    eligible: Final = tuple(model for model in models if model.name in by_name)
    ranked: Final = tuple(sorted(eligible, key=lambda model: by_name[model.name].rank_wape))[:members]
    if len(ranked) < 2:
        return None
    return WeightedEnsemble(
        members=tuple((model, 1.0 / max(by_name[model.name].rank_wape, _ENSEMBLE_WEIGHT_FLOOR)) for model in ranked)
    )


def _score_all(
    models: Sequence[ForecastModel], window: TrainingWindow, plan: Sequence[Fold]
) -> tuple[tuple[ForecastModel, ModelScore], ...]:
    return tuple(
        (model, score)
        for model, score in ((model, score_model(model, window, plan)) for model in models)
        if score is not None
    )


def _worth_paying_for(scores: Sequence[ModelScore], gate: float | None) -> bool:
    """Whether the maximum-likelihood candidates should be fitted at all.

    A statsmodels fit is roughly a hundred times the cost of every other
    candidate in the roster put together, and the forecast job has two thousand
    scopes and five minutes (§1.14). It is paid for where it can change the
    answer: when the cheap candidates already track the series inside the gate,
    the remaining headroom is smaller than the noise in the fact tables, and the
    tie-break would hand the win back to the simpler model regardless.

    This is a deliberate cost boundary, not a claim that ETS could never do
    better. Raising ``expensive_gate`` buys accuracy with wall-clock time.
    """
    if gate is None:
        return False
    scorable: Final = tuple(score.rank_wape for score in scores if score.wape is not None)
    return not scorable or min(scorable) > gate


def run_competition(
    window: TrainingWindow,
    roster: Sequence[ForecastModel],
    *,
    folds: Sequence[Fold] | None = None,
    with_ensemble: bool = True,
    expensive_gate: float | None = EXPENSIVE_WAPE_GATE,
) -> Competition:
    """Score every eligible candidate on the same folds and return the scoreboard and winner."""
    if not roster:
        raise ValueError("A competition needs at least one candidate")
    plan: Final = tuple(folds) if folds is not None else plan_folds(window.size)
    cheap: Final = _score_all(tuple(model for model in roster if not model.expensive), window, plan)
    costly: Final = (
        _score_all(tuple(model for model in roster if model.expensive), window, plan)
        if _worth_paying_for(tuple(score for _, score in cheap), expensive_gate)
        else ()
    )
    scored: Final = cheap + costly
    if not scored:
        simplest: Final = min(roster, key=lambda model: (model.complexity, model.min_points))
        return Competition(scores=(), winner=simplest, winning_score=None)

    base_scores: Final = tuple(score for _, score in scored)
    ensemble: Final = (
        _ensemble_of(tuple(model for model, _ in scored), base_scores, ENSEMBLE_MEMBERS) if with_ensemble else None
    )
    ensemble_score: Final = score_model(ensemble, window, plan) if ensemble is not None else None
    all_scores: Final = base_scores + ((ensemble_score,) if ensemble_score is not None else ())
    winning_score: Final = select_score(all_scores)
    candidates: Final = tuple(model for model, _ in scored) + ((ensemble,) if ensemble is not None else ())
    winner: Final = next(model for model in candidates if model.name == winning_score.name)
    return Competition(scores=all_scores, winner=winner, winning_score=winning_score)


def in_sample_residuals(model: ForecastModel, window: TrainingWindow, *, start: int | None = None) -> tuple[float, ...]:
    """Expanding one-day-ahead errors, for histories too short to backtest.

    Three days of history cannot fill a fold plan, but it can still say how wrong
    yesterday's level was about today, and a band drawn from that is honest in a
    way that a band of zero width is not.

    ``start`` is passed explicitly when several series of one scope need residual
    tuples of equal length for the joint bootstrap.
    """
    first: Final = max(model.min_points, MIN_TRAIN_POINTS) if start is None else start
    return tuple(
        window.values[index]
        - model.predict(TrainingWindow(start_day=window.start_day, values=window.values[:index]), 1)[0]
        for index in range(first, window.size)
    )
