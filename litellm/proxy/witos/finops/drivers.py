"""Driver decomposition: why the number moved (blueprint §1.9).

Deterministic counterfactual repricing, never an LLM's reading of a chart. The
spend of a window is modelled as

    requests x SUM_models share_m x (prompt_per_request x input_m
                                     + completion_per_request x output_m
                                     + cache_reads_per_request x cache_m)

and each factor is moved from its baseline value to its current value one at a
time, with everything else frozen. The difference each move makes is that
factor's contribution. Because the model is calibrated so that evaluating it at
the baseline reproduces the spend that was actually booked, the contributions are
in real dollars rather than in units of the model's own opinion.

The residual line is not decoration. Moving factors one at a time attributes none
of the interaction between them (more requests *and* bigger prompts is more than
the sum of its parts), so whatever the contributions fail to explain is printed
as its own line. A decomposition that silently spreads its interaction terms over
the named factors reads better and lies.

An LLM may narrate this table (§1.9 permits exactly that). It may not produce it.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from litellm.proxy.witos.finops.history import DailyUsage
from litellm.proxy.witos.finops.pricing import PriceBook

DriverFactor: TypeAlias = Literal[
    "request_volume", "model_mix", "prompt_size", "completion_size", "caching", "price_map", "residual"
]

DRIVER_ORDER: Final[tuple[DriverFactor, ...]] = (
    "request_volume",
    "model_mix",
    "prompt_size",
    "completion_size",
    "caching",
    "price_map",
)

_DRIVER_LABELS: Final[Mapping[DriverFactor, str]] = MappingProxyType(
    {
        "request_volume": "request volume",
        "model_mix": "expensive-model share",
        "prompt_size": "average prompt size",
        "completion_size": "average completion size",
        "caching": "caching",
        "price_map": "price changes",
        "residual": "residual",
    }
)


@dataclass(frozen=True, slots=True)
class FactorState:
    """One window's spend, taken apart into the things that can move independently."""

    requests: float
    prompt_per_request: float
    completion_per_request: float
    cache_reads_per_request: float
    mix: Mapping[str, float]
    price_scale: Decimal
    actual_spend: Decimal


@dataclass(frozen=True, slots=True)
class DriverContribution:
    factor: DriverFactor
    label: str
    delta_usd: Decimal
    delta_pct: float


@dataclass(frozen=True, slots=True)
class DriverDecomposition:
    """The signed contribution list §1.9 renders, plus the totals it has to add up to."""

    baseline_spend: Decimal
    current_spend: Decimal
    delta_usd: Decimal
    delta_pct: float
    contributions: tuple[DriverContribution, ...]


def _totals(days: Sequence[DailyUsage]) -> tuple[float, float, float, float]:
    return (
        sum(day.request_count for day in days),
        sum(day.prompt_tokens for day in days),
        sum(day.completion_tokens for day in days),
        sum(day.cache_read_tokens for day in days),
    )


def _mix_shares(days: Sequence[DailyUsage]) -> Mapping[str, float]:
    groups: Final = frozenset(group for day in days for group in day.mix_tokens)
    totals: Final = MappingProxyType({group: sum(day.mix_tokens.get(group, 0.0) for day in days) for group in groups})
    grand: Final = sum(totals.values())
    if grand <= 0:
        return MappingProxyType({})
    return MappingProxyType({group: tokens / grand for group, tokens in totals.items()})


def build_state(days: Sequence[DailyUsage], book: PriceBook) -> FactorState:
    """Factorise a window, calibrating the price scale so the model reproduces its spend."""
    requests, prompt, completion, cache = _totals(days)
    span: Final = max(1, len(days))
    actual: Final = sum((day.spend_usd for day in days), Decimal(0))
    mix: Final = _mix_shares(days)
    unpriced: Final = FactorState(
        requests=requests / span,
        prompt_per_request=prompt / requests if requests else 0.0,
        completion_per_request=completion / requests if requests else 0.0,
        cache_reads_per_request=cache / requests if requests else 0.0,
        mix=mix,
        price_scale=Decimal(1),
        actual_spend=actual,
    )
    modelled: Final = evaluate(unpriced, book) * Decimal(span)
    scale: Final = (actual / modelled) if modelled > 0 else Decimal(1)
    return replace(unpriced, price_scale=scale)


def evaluate(state: FactorState, book: PriceBook) -> Decimal:
    """Daily spend implied by one factor state, at the state's own price scale."""
    blended: Final = book.blend(state.mix)
    per_request: Final = blended.cost_of(
        prompt_tokens=Decimal(str(state.prompt_per_request)),
        completion_tokens=Decimal(str(state.completion_per_request)),
        cache_read_tokens=Decimal(str(state.cache_reads_per_request)),
    )
    return per_request * Decimal(str(state.requests)) * state.price_scale


def _with_factor(baseline: FactorState, current: FactorState, factor: DriverFactor) -> FactorState:
    match factor:
        case "request_volume":
            return replace(baseline, requests=current.requests)
        case "model_mix":
            return replace(baseline, mix=current.mix)
        case "prompt_size":
            return replace(baseline, prompt_per_request=current.prompt_per_request)
        case "completion_size":
            return replace(baseline, completion_per_request=current.completion_per_request)
        case "caching":
            return replace(baseline, cache_reads_per_request=current.cache_reads_per_request)
        case "price_map":
            return replace(baseline, price_scale=current.price_scale)
        case "residual":
            return baseline


def decompose(
    baseline_days: Sequence[DailyUsage], current_days: Sequence[DailyUsage], book: PriceBook
) -> DriverDecomposition:
    """Attribute the change in spend between two windows to the factors that moved.

    Both windows are normalised to a per-day figure, so comparing a 7-day window
    with a 6-day one (a month boundary, a backfill still running) compares rates
    rather than lengths.
    """
    baseline: Final = build_state(baseline_days, book)
    current: Final = build_state(current_days, book)
    baseline_daily: Final = evaluate(baseline, book)
    current_daily: Final = evaluate(current, book)
    delta: Final = current_daily - baseline_daily
    contributions: Final = tuple(
        _contribution(factor, evaluate(_with_factor(baseline, current, factor), book) - baseline_daily, baseline_daily)
        for factor in DRIVER_ORDER
    )
    attributed: Final = sum((item.delta_usd for item in contributions), Decimal(0))
    residual: Final = _contribution("residual", delta - attributed, baseline_daily)
    return DriverDecomposition(
        baseline_spend=baseline_daily,
        current_spend=current_daily,
        delta_usd=delta,
        delta_pct=_pct(delta, baseline_daily),
        contributions=contributions + (residual,),
    )


def _contribution(factor: DriverFactor, delta: Decimal, baseline: Decimal) -> DriverContribution:
    return DriverContribution(
        factor=factor, label=_DRIVER_LABELS[factor], delta_usd=delta, delta_pct=_pct(delta, baseline)
    )


def _pct(delta: Decimal, baseline: Decimal) -> float:
    return 0.0 if baseline <= 0 else float(delta / baseline * 100)
