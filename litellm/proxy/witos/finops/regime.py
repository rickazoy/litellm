"""Regime-change detection (blueprint §1.6).

AI workloads step. A team wires the gateway into one more service, a nightly
batch job doubles, a prompt template grows a retrieved-context block, and the
series has a new level that has nothing to do with the old one. A forecaster that
averages across that step is not merely imprecise, it is confidently wrong in a
specific direction for as long as the pre-change history stays in the window, so
detection here is not a refinement of the forecast, it is a precondition for it.

Two questions, two estimators, because they are not the same question:

*Is the last week different from the weeks before it* is answered by a robust
z-score, median and MAD rather than mean and standard deviation. The mean and
standard deviation of a window containing the step are both dragged by it, which
is exactly the failure mode being detected.

*When did it change* is answered by a CUSUM against the earliest stretch of the
series, which accumulates small persistent deviations that no single-day z-score
would flag and gives back an index the training window can be cut at.

A CUSUM alone cannot declare a step, because steady growth drifts away from its
own past exactly the way a step does, and a CUSUM used as the classifier labels
every healthy growing workload a step change and truncates its history every
night. So a step is declared by either of two things: the last week is four
robust sigmas from the weeks before it, or the series moves by most of its own
level across the CUSUM's change point in a single week. Compounding growth clears
neither test until it reaches roughly 8% a day, at which point calling it a
regime change is correct.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate
from math import copysign
from statistics import median
from typing import Final, Literal, TypeAlias

Regime: TypeAlias = Literal["stable", "accelerating", "decelerating", "step_change", "volatile", "insufficient_history"]

MIN_REGIME_POINTS: Final = 14
RECENT_WINDOW: Final = 7
REFERENCE_WINDOW: Final = 28

# A robust z of 4 is roughly a one-in-fifteen-thousand deviation under normality,
# which is the point where "this week is unusual" stops being a plausible reading
# of natural variance in daily token volume.
STEP_CHANGE_Z: Final = 4.0
DIRECTIONAL_Z: Final = 2.0

# MAD-to-median ratio above which a series is too dispersed for its own trend to
# mean anything, so it is labelled rather than extrapolated with confidence.
VOLATILE_DISPERSION: Final = 0.75

CUSUM_DRIFT: Final = 0.5
CUSUM_THRESHOLD: Final = 5.0

# How much of its own level a series has to move across the change point for the
# move to be a step rather than a trend. Compounding growth would need 8% a day
# to reach it inside a week.
STEP_JUMP_RATIO: Final = 0.75
_CONSTANT_DEPARTURE: Final = 0.5

# A perfectly flat history has zero MAD, so every deviation from it is infinitely
# significant. The z is capped so the stored number stays finite and comparable.
_Z_CAP: Final = 8.0
_MAD_TO_SIGMA: Final = 1.4826
_MEAN_AD_TO_SIGMA: Final = 1.253314


@dataclass(frozen=True, slots=True)
class RegimeAssessment:
    """What the history is doing, and where it started doing it."""

    regime: Regime
    robust_z: float
    cusum: float
    change_point: int | None

    @property
    def is_step_change(self) -> bool:
        return self.regime == "step_change"


def _robust_scale(values: Sequence[float], center: float) -> float:
    """MAD-based scale, falling back to the mean absolute deviation when the MAD is zero.

    A weekday/weekend workload has a zero MAD, because more than half its days
    are the same weekday level. Treating that as "no variance" makes every
    Saturday an infinitely significant deviation and labels ordinary seasonality
    a step change. The mean-absolute-deviation fallback (Iglewicz and Hoaglin's
    recommendation for exactly this case) still sees the weekend. Only a series
    that is genuinely constant reaches zero here, and for that one any change
    really is a step.
    """
    deviations: Final = tuple(abs(value - center) for value in values)
    mad: Final = _MAD_TO_SIGMA * median(deviations)
    if mad > 0:
        return mad
    return _MEAN_AD_TO_SIGMA * (sum(deviations) / len(deviations)) if deviations else 0.0


def _robust_z(recent_mean: float, center: float, scale: float) -> float:
    if scale <= 0:
        return 0.0 if recent_mean == center else copysign(_Z_CAP, recent_mean - center)
    return max(-_Z_CAP, min(_Z_CAP, (recent_mean - center) / scale))


def _local_jump(values: Sequence[float], change_point: int) -> float:
    """Relative size of the discontinuity at ``change_point``, week against week.

    This is what separates a step from growth. Both drift away from where they
    started, so both trip a CUSUM; only a step moves by most of its own level
    inside a single boundary. Two percent a day compounds to fifteen percent over
    the week either side of any point you pick, and stays well under the
    threshold.
    """
    before: Final = values[max(0, change_point - RECENT_WINDOW) : change_point]
    after: Final = values[change_point : change_point + RECENT_WINDOW]
    if not before or not after:
        return 0.0
    level: Final = median(before)
    return abs(median(after) - level) / max(abs(level), 1.0)


def _cusum_change_point(values: Sequence[float]) -> tuple[float, int | None]:
    """Two-sided CUSUM against the earliest half of the series.

    The reference is the oldest stretch rather than the whole series so that the
    statistic measures drift away from where the series started, not away from an
    average that already contains the drift.
    """
    reference: Final = values[: max(RECENT_WINDOW, len(values) // 2)]
    center: Final = median(reference)
    scale: Final = _robust_scale(reference, center)
    if scale <= 0:
        return _departure_from_constant(values, center)
    standardized: Final = tuple((value - center) / scale for value in values)
    walks: Final = tuple(
        accumulate(
            standardized,
            lambda carried, point: (
                max(0.0, carried[0] + point - CUSUM_DRIFT),
                min(0.0, carried[1] + point + CUSUM_DRIFT),
            ),
            initial=(0.0, 0.0),
        )
    )
    excursions: Final = tuple(max(high, -low) for high, low in walks[1:])
    crossed: Final = tuple(index for index, value in enumerate(excursions) if value >= CUSUM_THRESHOLD)
    return (max(excursions) if excursions else 0.0), (crossed[0] if crossed else None)


def _departure_from_constant(values: Sequence[float], center: float) -> tuple[float, int | None]:
    """Change point for a reference stretch with no spread at all.

    A perfectly flat history has zero MAD and zero mean deviation, so a
    standardised CUSUM cannot be formed: every deviation divides by zero. The
    series that produce this are exactly the ones where a step is unmistakable
    without statistics, so the first day that departs from the constant by half
    of it is the change point.
    """
    departures: Final = tuple(
        index for index, value in enumerate(values) if abs(value - center) > _CONSTANT_DEPARTURE * max(abs(center), 1.0)
    )
    return (CUSUM_THRESHOLD if departures else 0.0), (departures[0] if departures else None)


def detect_regime(values: Sequence[float]) -> RegimeAssessment:
    """Classify the series, and locate the step when there is one."""
    if len(values) < MIN_REGIME_POINTS:
        return RegimeAssessment(regime="insufficient_history", robust_z=0.0, cusum=0.0, change_point=None)

    recent: Final = values[-RECENT_WINDOW:]
    prior: Final = values[-(RECENT_WINDOW + REFERENCE_WINDOW) : -RECENT_WINDOW]
    center: Final = median(prior)
    scale: Final = _robust_scale(prior, center)
    robust_z: Final = _robust_z(sum(recent) / len(recent), center, scale)
    cusum, change_point = _cusum_change_point(values)

    stepped: Final = abs(robust_z) >= STEP_CHANGE_Z or (
        change_point is not None and _local_jump(values, change_point) >= STEP_JUMP_RATIO
    )
    if stepped:
        return RegimeAssessment(regime="step_change", robust_z=robust_z, cusum=cusum, change_point=change_point)
    if center > 0 and scale / center > VOLATILE_DISPERSION:
        return RegimeAssessment(regime="volatile", robust_z=robust_z, cusum=cusum, change_point=None)
    if robust_z >= DIRECTIONAL_Z:
        return RegimeAssessment(regime="accelerating", robust_z=robust_z, cusum=cusum, change_point=None)
    if robust_z <= -DIRECTIONAL_Z:
        return RegimeAssessment(regime="decelerating", robust_z=robust_z, cusum=cusum, change_point=None)
    return RegimeAssessment(regime="stable", robust_z=robust_z, cusum=cusum, change_point=None)


def training_start(assessment: RegimeAssessment, *, length: int, min_points: int) -> int:
    """Where training should begin, given the regime.

    A step change makes the pre-change history actively misleading, so it is
    dropped rather than down-weighted. It is only dropped when what remains can
    still support a fit: below that the whole history is kept and the forecast
    carries the ``step_change`` label, which is the honest outcome. Truncating to
    four points and reporting a normal forecast would not be.
    """
    if not assessment.is_step_change or assessment.change_point is None:
        return 0
    remaining: Final = length - assessment.change_point
    return assessment.change_point if remaining >= min_points else 0
