"""Runway and exhaustion probabilities (blueprint §1.7).

The output of this module is one sentence a CFO can act on: *"78% probability of
exceeding the $100,000 monthly budget. Median exhaustion Aug 27; conservative
Aug 24."* Everything here exists to make each number in that sentence defensible.

The probabilities come from the same Monte Carlo paths that drew the forecast
band, not from a second calculation. Runway asks each simulated path when its
running total crosses what is left of the quota and reads the answer off the
distribution of those dates. A separately-derived runway (remaining divided by
average burn, say) would be a different model of the same future, and the day it
disagreed with the chart above it, the chart would be the one that lost the
argument.

Two distinctions this module refuses to blur:

A path that never exhausts inside the horizon is not a path that exhausts on the
last day. It stays ``None`` all the way to the API, so "no median exhaustion
date" reads as what it is.

``exhaustion_p90`` is the *early* date, not the late one. It answers "how soon
could this happen", so it is the 10th percentile of the date distribution, which
is the 90th percentile of spend. The naming follows the blueprint's "as early
as"; the arithmetic follows the definition.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from math import ceil
from types import MappingProxyType
from typing import Final, TypeAlias

from litellm.proxy.witos.finops.engine import ScopeForecast
from litellm.proxy.witos.finops.history import ScopeRef
from litellm.proxy.witos.finops.simulation import P50, PathEnsemble

QuotaMetric: TypeAlias = str

MONEY_METRICS: Final = frozenset({"usd", "spend_usd"})

CONSERVATIVE_PROBABILITY: Final = 0.10
MEDIAN_PROBABILITY: Final = 0.50


@dataclass(frozen=True, slots=True)
class Quota:
    """One entitlement, mirrored from a LiteLLM budget or set by hand (§1.4)."""

    quota_id: str
    scope: ScopeRef
    metric: QuotaMetric
    limit_value: Decimal
    period: str
    period_start: datetime
    period_end: datetime | None
    reset_at: datetime | None
    source_type: str
    soft_threshold_pct: Decimal | None = None

    @property
    def is_read_only(self) -> bool:
        """Mirrored budgets are owned by LiteLLM; editing them here would be edited over."""
        return self.source_type == "litellm_budget"

    def cycle_end(self) -> date | None:
        boundary: Final = self.reset_at or self.period_end
        return boundary.date() if boundary else None


@dataclass(frozen=True, slots=True)
class RunwayResult:
    """What the simulated paths say about one quota."""

    quota_id: str
    scope: ScopeRef
    metric: QuotaMetric
    limit_value: Decimal
    consumed: Decimal
    remaining: Decimal
    exhaustion_p50: date | None
    exhaustion_p90: date | None
    prob_exhaust_before_reset: float
    prob_overrun_this_period: float
    survives_cycle: bool
    projected_consumption_pct: float | None
    burn_per_day: Decimal
    cycle_end: date | None
    period: str

    @property
    def consumed_pct(self) -> float:
        return 0.0 if self.limit_value <= 0 else float(self.consumed / self.limit_value * 100)


def _date_at_probability(crossings: Sequence[int | None], probability: float, ensemble: PathEnsemble) -> date | None:
    """The earliest day by which ``probability`` of the paths have exhausted the quota."""
    if not crossings:
        return None
    reached: Final = tuple(sorted(offset for offset in crossings if offset is not None))
    rank: Final = max(1, ceil(probability * len(crossings)))
    if rank > len(reached):
        return None
    return ensemble.day_at(reached[rank - 1])


def _fraction_crossing_by(crossings: Sequence[int | None], offset: int) -> float:
    if not crossings:
        return 0.0
    return sum(1 for crossing in crossings if crossing is not None and crossing <= offset) / len(crossings)


def compute_runway(forecast: ScopeForecast, quota: Quota, consumed: Decimal, *, today: date) -> RunwayResult | None:
    """Runway for one quota, or ``None`` when the forecast holds no series for its metric."""
    ensemble: Final = forecast.ensemble_for(quota.metric)
    if ensemble is None:
        return None
    remaining: Final = quota.limit_value - consumed
    cycle_end: Final = quota.cycle_end()
    if remaining <= 0:
        return _already_exhausted(forecast, quota, consumed, remaining, today=today, cycle_end=cycle_end)

    crossings: Final = ensemble.crossing_offsets(float(remaining))
    cycle_offset: Final = ensemble.offset_of(cycle_end) if cycle_end else ensemble.horizon - 1
    period_offset: Final = ensemble.offset_of(quota.period_end.date()) if quota.period_end else ensemble.horizon - 1
    exhaustion_p50: Final = _date_at_probability(crossings, MEDIAN_PROBABILITY, ensemble)
    projected: Final = (
        float((consumed + Decimal(str(ensemble.cumulative_at(cycle_offset, P50)))) / quota.limit_value * 100)
        if quota.limit_value > 0
        else None
    )
    return RunwayResult(
        quota_id=quota.quota_id,
        scope=quota.scope,
        metric=quota.metric,
        limit_value=quota.limit_value,
        consumed=consumed,
        remaining=remaining,
        exhaustion_p50=exhaustion_p50,
        exhaustion_p90=_date_at_probability(crossings, CONSERVATIVE_PROBABILITY, ensemble),
        prob_exhaust_before_reset=_fraction_crossing_by(crossings, cycle_offset),
        prob_overrun_this_period=ensemble.probability_of_crossing_by(period_offset, float(remaining)),
        survives_cycle=_survives(exhaustion_p50, cycle_end),
        projected_consumption_pct=projected,
        burn_per_day=_burn_per_day(ensemble),
        cycle_end=cycle_end,
        period=quota.period,
    )


def _survives(exhaustion_p50: date | None, cycle_end: date | None) -> bool:
    if exhaustion_p50 is None:
        return True
    return cycle_end is not None and exhaustion_p50 > cycle_end


def _burn_per_day(ensemble: PathEnsemble) -> Decimal:
    band: Final = ensemble.daily_quantiles((P50,))[0]
    if not band:
        return Decimal(0)
    days: Final = min(7, len(band))
    return Decimal(str(sum(band[:days]) / days))


def _already_exhausted(
    forecast: ScopeForecast,
    quota: Quota,
    consumed: Decimal,
    remaining: Decimal,
    *,
    today: date,
    cycle_end: date | None,
) -> RunwayResult:
    """A quota with nothing left is not a forecasting problem, and is not reported as one."""
    ensemble: Final = forecast.ensemble_for(quota.metric)
    return RunwayResult(
        quota_id=quota.quota_id,
        scope=quota.scope,
        metric=quota.metric,
        limit_value=quota.limit_value,
        consumed=consumed,
        remaining=remaining,
        exhaustion_p50=today,
        exhaustion_p90=today,
        prob_exhaust_before_reset=1.0,
        prob_overrun_this_period=1.0,
        survives_cycle=False,
        projected_consumption_pct=(float(consumed / quota.limit_value * 100) if quota.limit_value > 0 else None),
        burn_per_day=_burn_per_day(ensemble) if ensemble else Decimal(0),
        cycle_end=cycle_end,
        period=quota.period,
    )


def runway_statement(result: RunwayResult) -> str:
    """The §1.7 sentence, built from the numbers rather than written about them."""
    unit: Final = "$" if result.metric in MONEY_METRICS else ""
    limit: Final = f"{unit}{result.limit_value:,.0f}" if unit else f"{result.limit_value:,.0f} {result.metric}"
    if result.remaining <= 0:
        return f"The {limit} {_period_word(result)} allowance is already exhausted."
    probability: Final = f"{result.prob_exhaust_before_reset * 100:.0f}%"
    if result.exhaustion_p50 is None:
        return (
            f"{probability} probability of exceeding the {limit} {_period_word(result)} budget. "
            "No median exhaustion date inside the forecast horizon."
        )
    conservative: Final = f" conservative {result.exhaustion_p90:%b %d}." if result.exhaustion_p90 else "."
    return (
        f"{probability} probability of exceeding the {limit} {_period_word(result)} budget. "
        f"Median exhaustion {result.exhaustion_p50:%b %d};{conservative}"
    )


_PERIOD_WORDS: Final[Mapping[str, str]] = MappingProxyType(
    {"day": "daily", "week": "weekly", "month": "monthly", "quarter": "quarterly", "year": "annual"}
)


def _period_word(result: RunwayResult) -> str:
    return _PERIOD_WORDS.get(result.period, result.period)
