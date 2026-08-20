"""Residual-bootstrap Monte Carlo (blueprint §1.6), and the paths runway reuses (§1.7).

The band around a forecast is drawn from how wrong this scope's winning model
actually was on this scope's own history, not from an assumed distribution. Each
simulated day is the point forecast plus a residual resampled from the backtest,
which means a workload whose errors are fat-tailed and one-sided gets a band that
is fat-tailed and one-sided, without anyone choosing a distribution for it.

Two decisions in here are load-bearing.

**The draws are joint across series.** One index is drawn per simulated day and
used to look up the residual of every series at once, so a path that samples a
busy day samples it for prompt tokens and completion tokens and requests
together. Sampling each series independently would quietly assume the errors are
uncorrelated, and the cost paths built from them would have a variance that is
too small in exactly the place a CFO reads.

**These paths are the only paths.** Runway (§1.7) computes exhaustion
probabilities by walking the same simulated cumulative sums, so the P90 date on
the runway table and the P90 band on the chart cannot disagree. Computing runway
from its own separate simulation is the fastest way to publish two numbers that
contradict each other on the same screen.
"""

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import accumulate
from math import ceil
from random import Random
from types import MappingProxyType
from typing import Final

DEFAULT_PATH_COUNT: Final = 500
P10: Final = 0.10
P50: Final = 0.50
P90: Final = 0.90


def stable_seed(*parts: str) -> int:
    """A seed that depends only on what the forecast is of.

    Rebuilding a scope's forecast from unchanged facts has to reproduce the same
    bands, or every nightly run would move the published P90 by a few dollars for
    no reason anybody could explain.
    """
    digest: Final = hashlib.sha256("␟".join(parts).encode()).hexdigest()
    return int(digest[:16], 16)


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    """Nearest-rank quantile: an observed path value, never an interpolation between two."""
    if not sorted_values:
        return 0.0
    rank: Final = min(len(sorted_values), max(1, ceil(probability * len(sorted_values))))
    return sorted_values[rank - 1]


@dataclass(frozen=True, slots=True)
class PathEnsemble:
    """Simulated daily values and their running totals, one row per path."""

    start_day: date
    paths: tuple[tuple[float, ...], ...]
    cumulative: tuple[tuple[float, ...], ...]
    residual_count: int

    @property
    def horizon(self) -> int:
        return len(self.paths[0]) if self.paths else 0

    @property
    def path_count(self) -> int:
        return len(self.paths)

    @property
    def has_band(self) -> bool:
        """False when there were no residuals to resample, so P10 and P90 collapse onto P50."""
        return self.residual_count > 0

    def day_at(self, offset: int) -> date:
        return self.start_day + timedelta(days=offset)

    def offset_of(self, day: date) -> int:
        return (day - self.start_day).days

    def _columns(self, rows: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(sorted(column)) for column in zip(*rows, strict=True))

    def daily_quantiles(self, probabilities: Sequence[float]) -> tuple[tuple[float, ...], ...]:
        """One row per requested probability, each with one value per forecast day."""
        columns: Final = self._columns(self.paths)
        return tuple(tuple(_quantile(column, probability) for column in columns) for probability in probabilities)

    def cumulative_quantiles(self, probabilities: Sequence[float]) -> tuple[tuple[float, ...], ...]:
        columns: Final = self._columns(self.cumulative)
        return tuple(tuple(_quantile(column, probability) for column in columns) for probability in probabilities)

    def cumulative_at(self, offset: int, probability: float) -> float:
        if not self.cumulative or offset < 0:
            return 0.0
        clamped: Final = min(offset, self.horizon - 1)
        return _quantile(tuple(sorted(path[clamped] for path in self.cumulative)), probability)

    def crossing_offsets(self, threshold: float) -> tuple[int | None, ...]:
        """For each path, the first day its running total reaches ``threshold``.

        ``None`` means that path never reaches it inside the horizon, which is a
        real outcome and must stay distinguishable from "reaches it on the last
        day": collapsing the two is how a runway table starts claiming certainty
        it does not have.
        """
        return tuple(_first_crossing(path, threshold) for path in self.cumulative)

    def probability_of_crossing_by(self, offset: int, threshold: float) -> float:
        if offset < 0 or not self.cumulative:
            return 0.0
        clamped: Final = min(offset, self.horizon - 1)
        return sum(1 for path in self.cumulative if path[clamped] >= threshold) / self.path_count


def _first_crossing(path: Sequence[float], threshold: float) -> int | None:
    crossed: Final = tuple(offset for offset, total in enumerate(path) if total >= threshold)
    return crossed[0] if crossed else None


def _draw_indices(residual_count: int, *, path_count: int, horizon: int, seed: int) -> tuple[int, ...]:
    if residual_count <= 0:
        return ()
    return tuple(Random(seed).choices(range(residual_count), k=path_count * horizon))


def _build_paths(
    points: Sequence[float],
    residuals: Sequence[float],
    draws: Sequence[int],
    *,
    path_count: int,
    horizon: int,
) -> tuple[tuple[float, ...], ...]:
    if not residuals:
        return (tuple(points),) * path_count
    return tuple(
        tuple(max(0.0, points[offset] + residuals[draws[path * horizon + offset]]) for offset in range(horizon))
        for path in range(path_count)
    )


def simulate(
    points: Mapping[str, Sequence[float]],
    residuals: Mapping[str, Sequence[float]],
    *,
    start_day: date,
    seed: int,
    path_count: int = DEFAULT_PATH_COUNT,
) -> Mapping[str, PathEnsemble]:
    """Bootstrap every series over one shared set of draws.

    Every series must supply the same number of residuals, because index *i*
    means "the model's error on the same historical day" for all of them. The
    backtest guarantees this by scoring every series of a scope over one fold
    plan.
    """
    if not points:
        return MappingProxyType({})
    horizons: Final = frozenset(len(series) for series in points.values())
    if len(horizons) != 1:
        raise ValueError(f"Every series must share one horizon, got {sorted(horizons)}")
    counts: Final = frozenset(len(residuals.get(name, ())) for name in points)
    if len(counts) != 1:
        raise ValueError(f"Every series must supply the same residual count, got {sorted(counts)}")
    horizon: Final = next(iter(horizons))
    residual_count: Final = next(iter(counts))
    draws: Final = _draw_indices(residual_count, path_count=path_count, horizon=horizon, seed=seed)
    return MappingProxyType(
        {
            name: _ensemble(
                _build_paths(
                    series,
                    residuals.get(name, ()),
                    draws,
                    path_count=path_count,
                    horizon=horizon,
                ),
                start_day=start_day,
                residual_count=residual_count,
            )
            for name, series in points.items()
        }
    )


def _ensemble(paths: tuple[tuple[float, ...], ...], *, start_day: date, residual_count: int) -> PathEnsemble:
    return PathEnsemble(
        start_day=start_day,
        paths=paths,
        cumulative=tuple(tuple(accumulate(path)) for path in paths),
        residual_count=residual_count,
    )


def weighted_sum(
    ensembles: Sequence[PathEnsemble],
    daily_weights: Sequence[Sequence[float]],
    *,
    addend: Sequence[float] = (),
) -> PathEnsemble:
    """Combine ensembles path by path with a per-day weight for each.

    This is how simulated token paths become simulated cost paths: the weight of
    a series on a given day is that day's blended unit price. Because the
    combination happens inside each path rather than on the quantiles, the cost
    band inherits the correlation between the token series instead of assuming
    the worst case of each lands on the same day.
    """
    if not ensembles:
        raise ValueError("Nothing to combine")
    first: Final = ensembles[0]
    offsets: Final = tuple(addend) if addend else (0.0,) * first.horizon
    return _ensemble(
        tuple(
            tuple(
                offsets[offset]
                + sum(
                    ensemble.paths[path][offset] * weights[offset]
                    for ensemble, weights in zip(ensembles, daily_weights, strict=True)
                )
                for offset in range(first.horizon)
            )
            for path in range(first.path_count)
        ),
        start_day=first.start_day,
        residual_count=first.residual_count,
    )
