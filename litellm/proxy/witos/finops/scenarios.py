"""What-if evaluation (blueprint §1.10).

A scenario is not a second forecasting model. It replays the forecast's own usage
series with the assumptions applied, reprices the result through the same price
book, and re-runs the Monte Carlo over the same residuals. That is only possible
because the forecast is usage-based (§1.3): "migrate 30% of Opus traffic to
Sonnet" is an exact repricing of a known token volume, not a guess at how a
dollar series would have behaved.

Determinism is a property the tests pin. Given the same assumptions and the same
``pricing_snapshot_hash``, evaluation returns byte-identical numbers, because the
seed is derived from those two things and nothing else. A scenario a CFO screen
shotted last week has to still say the same thing this week unless the prices or
the assumptions moved.

The result is a full ``ScopeForecast``, not a chart. Runway, drivers and the CSV
export all take a forecast, so the scenario's exhaustion dates come out of the
same function that produced the baseline's, and cannot drift from it.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal
from types import MappingProxyType
from typing import Annotated, Final

from pydantic import BaseModel, ConfigDict, Field

from litellm.proxy.witos.finops.engine import ScopeForecast, SeriesForecast, month_end, quarter_end
from litellm.proxy.witos.finops.pricing import BlendedPrice, PriceBook, build_price_book
from litellm.proxy.witos.finops.simulation import P50, P90, PathEnsemble, simulate, stable_seed, weighted_sum

_PERCENT: Final = Decimal(100)


class ModelMigration(BaseModel):
    """Move a share of one model group's traffic onto another."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=(), populate_by_name=True)

    from_model: Annotated[str, Field(alias="from")]
    to_model: Annotated[str, Field(alias="to")]
    traffic_pct: Annotated[float, Field(ge=0, le=100)]


class PriceAdjustment(BaseModel):
    """A percentage move on a model's published price, not a replacement for it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_delta_pct: Annotated[Decimal, Field(default=Decimal(0), ge=-100)]
    output_delta_pct: Annotated[Decimal, Field(default=Decimal(0), ge=-100)]


class NewWorkload(BaseModel):
    """Spend that has no usage history to scale, added as a flat daily cost."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    daily_spend_usd: Annotated[Decimal, Field(ge=0)]
    start_date: date


def _no_adjustments() -> Mapping[str, PriceAdjustment]:
    """A fresh empty mapping per model, in a shape pydantic can serialise back out."""
    return {}  # mutable-ok: pydantic owns this value and serialises it; the field's type is a read-only Mapping


class ScenarioAssumptions(BaseModel):
    """The §1.10 assumptions document, validated.

    ``new_users`` needs ``baseline_users`` to mean anything: without knowing how
    many users produce today's traffic, "150 new users" cannot be converted into
    tokens, and inventing a per-user average would make the scenario's headline
    number a fabrication. The API rejects the combination rather than guessing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    usage_growth_pct: float = 0.0
    new_users: Annotated[int, Field(default=0, ge=0)]
    baseline_users: Annotated[int | None, Field(default=None, ge=1)]
    model_migrations: tuple[ModelMigration, ...] = ()
    cache_hit_rate: Annotated[float | None, Field(default=None, ge=0, le=1)] = None
    prompt_token_delta_pct: float = 0.0
    pricing_adjustments: Annotated[Mapping[str, PriceAdjustment], Field(default_factory=_no_adjustments)]
    new_workloads: tuple[NewWorkload, ...] = ()

    def growth_factor(self) -> float:
        seats: Final = (
            1.0
            if not self.new_users or not self.baseline_users
            else (self.baseline_users + self.new_users) / self.baseline_users
        )
        return max(0.0, 1.0 + self.usage_growth_pct / 100.0) * seats

    def requires_baseline_users(self) -> bool:
        return self.new_users > 0 and self.baseline_users is None

    def fingerprint(self) -> str:
        return self.model_dump_json()


@dataclass(frozen=True, slots=True)
class ScenarioProjection:
    """One side of the comparison, in the terms a CFO reads."""

    horizon_p50: Decimal
    horizon_p90: Decimal
    projected_eom: Decimal
    projected_eoq: Decimal


@dataclass(frozen=True, slots=True)
class ScenarioComparison:
    baseline: ScenarioProjection
    scenario: ScenarioProjection
    delta_usd: Decimal
    delta_pct: float

    @property
    def savings_usd(self) -> Decimal:
        return -self.delta_usd


def _shift_mix(mix: Mapping[str, float], migrations: Sequence[ModelMigration]) -> Mapping[str, float]:
    """Move traffic share between model groups, leaving untouched groups alone."""

    def moved_out(group: str) -> float:
        return sum(
            mix.get(group, 0.0) * migration.traffic_pct / 100.0
            for migration in migrations
            if migration.from_model == group
        )

    def moved_in(group: str) -> float:
        return sum(
            mix.get(migration.from_model, 0.0) * migration.traffic_pct / 100.0
            for migration in migrations
            if migration.to_model == group
        )

    groups: Final = frozenset(mix) | frozenset(migration.to_model for migration in migrations)
    return MappingProxyType(
        {group: max(0.0, mix.get(group, 0.0) - moved_out(group) + moved_in(group)) for group in sorted(groups)}
    )


def _book_covering(book: PriceBook, mix: Mapping[str, float]) -> PriceBook:
    """Extend the price book to any model a migration introduced.

    Without this a migration onto a model the scope has never used would be
    priced at zero and the scenario would report a saving of exactly 100%.
    """
    missing: Final = frozenset(mix) - frozenset(book.prices)
    if not missing:
        return book
    return build_price_book(tuple(book.prices) + tuple(missing))


def _adjustment_factors(
    assumptions: ScenarioAssumptions,
) -> Mapping[str, tuple[Decimal, Decimal]]:
    return MappingProxyType(
        {
            model: (
                Decimal(1) + adjustment.input_delta_pct / _PERCENT,
                Decimal(1) + adjustment.output_delta_pct / _PERCENT,
            )
            for model, adjustment in assumptions.pricing_adjustments.items()
        }
    )


def _scaled(values: Sequence[float], factor: float) -> tuple[float, ...]:
    return tuple(value * factor for value in values)


def _transformed_points(forecast: ScopeForecast, assumptions: ScenarioAssumptions) -> Mapping[str, tuple[float, ...]]:
    growth: Final = assumptions.growth_factor()
    prompt_factor: Final = growth * max(0.0, 1.0 + assumptions.prompt_token_delta_pct / 100.0)
    prompt: Final = _scaled(forecast.series["prompt_tokens"].point, prompt_factor)
    cached: Final = (
        tuple(value * assumptions.cache_hit_rate for value in prompt)
        if assumptions.cache_hit_rate is not None
        else _scaled(forecast.series["cache_read_tokens"].point, growth)
    )
    return MappingProxyType(
        {
            "request_count": _scaled(forecast.series["request_count"].point, growth),
            "prompt_tokens": prompt,
            "completion_tokens": _scaled(forecast.series["completion_tokens"].point, growth),
            "cache_read_tokens": cached,
        }
    )


def _transformed_residuals(
    forecast: ScopeForecast, points: Mapping[str, tuple[float, ...]]
) -> Mapping[str, tuple[float, ...]]:
    """Scale each series' residuals by the same factor its level moved.

    A workload that doubles does not keep the absolute error of the workload that
    was half its size. Scaling the residuals with the level keeps the band's
    width proportional, which is what the historical errors actually said.
    """
    return MappingProxyType(
        {
            name: _scaled(forecast.residuals.get(name, ()), _level_ratio(forecast.series.get(name), points[name]))
            for name in points
        }
    )


def _level_ratio(original: SeriesForecast | None, transformed: Sequence[float]) -> float:
    if original is None:
        return 1.0
    baseline: Final = sum(original.point)
    return 1.0 if baseline <= 0 else sum(transformed) / baseline


def _workload_addend(assumptions: ScenarioAssumptions, ensemble: PathEnsemble) -> tuple[float, ...]:
    return tuple(
        float(
            sum(
                (
                    workload.daily_spend_usd
                    for workload in assumptions.new_workloads
                    if ensemble.day_at(offset) >= workload.start_date
                ),
                Decimal(0),
            )
        )
        for offset in range(ensemble.horizon)
    )


def apply_scenario(forecast: ScopeForecast, assumptions: ScenarioAssumptions) -> ScopeForecast:
    """Replay the forecast under the assumptions and reprice it exactly.

    ``ACTUAL_COST_BASED`` scopes have no token series to transform, so growth and
    price assumptions are applied to the cost series directly and model
    migrations are ignored: pretending to migrate traffic that was never measured
    in tokens would be theatre.
    """
    if forecast.strategy == "ACTUAL_COST_BASED":
        return _apply_to_cost_series(forecast, assumptions)

    points: Final = _transformed_points(forecast, assumptions)
    mix: Final = _shift_mix(forecast.mix_shares, assumptions.model_migrations)
    book: Final = _book_covering(forecast.price_book, mix).with_adjustments(_adjustment_factors(assumptions))
    calibration: Final = forecast.pricing_fidelity.calibration()
    blended: Final = book.blend(mix).scaled(calibration)
    seed: Final = stable_seed(str(forecast.scope), book.snapshot_hash, assumptions.fingerprint())
    ensembles: Final = simulate(
        points,
        _transformed_residuals(forecast, points),
        start_day=forecast.forecast_start,
        seed=seed,
        path_count=forecast.path_count,
    )
    series: Final = MappingProxyType(
        {
            name: replace(forecast.series[name], point=points[name], ensemble=ensembles[name])
            for name in points
            if name in forecast.series
        }
    )
    return replace(
        forecast,
        price_book=book,
        mix_shares=mix,
        blended_price=blended,
        series=series,
        spend=_priced(series, blended, assumptions, forecast.spend.horizon),
        total_tokens=weighted_sum(
            (ensembles["prompt_tokens"], ensembles["completion_tokens"]),
            ((1.0,) * forecast.spend.horizon, (1.0,) * forecast.spend.horizon),
        ),
        seed=seed,
    )


def _priced(
    series: Mapping[str, SeriesForecast],
    blended: BlendedPrice,
    assumptions: ScenarioAssumptions,
    horizon: int,
) -> PathEnsemble:
    ensembles: Final = (
        series["prompt_tokens"].ensemble,
        series["completion_tokens"].ensemble,
        series["cache_read_tokens"].ensemble,
    )
    return weighted_sum(
        ensembles,
        (
            (float(blended.input_per_token),) * horizon,
            (float(blended.output_per_token),) * horizon,
            (float(blended.cache_read_per_token),) * horizon,
        ),
        addend=_workload_addend(assumptions, ensembles[0]),
    )


def _apply_to_cost_series(forecast: ScopeForecast, assumptions: ScenarioAssumptions) -> ScopeForecast:
    growth: Final = assumptions.growth_factor()
    points: Final = MappingProxyType({"spend_usd": _scaled(forecast.series["spend_usd"].point, growth)})
    seed: Final = stable_seed(str(forecast.scope), forecast.price_book.snapshot_hash, assumptions.fingerprint(), "cost")
    ensembles: Final = simulate(
        points,
        _transformed_residuals(forecast, points),
        start_day=forecast.forecast_start,
        seed=seed,
        path_count=forecast.path_count,
    )
    spend: Final = weighted_sum(
        (ensembles["spend_usd"],),
        ((1.0,) * forecast.spend.horizon,),
        addend=_workload_addend(assumptions, ensembles["spend_usd"]),
    )
    return replace(
        forecast,
        series=MappingProxyType(
            {"spend_usd": replace(forecast.series["spend_usd"], point=points["spend_usd"], ensemble=spend)}
        ),
        spend=spend,
        total_tokens=spend,
        seed=seed,
    )


def project(forecast: ScopeForecast, *, actual_to_date: Decimal, as_of: date) -> ScenarioProjection:
    """Headline figures for one side of a comparison."""
    last: Final = forecast.spend.horizon - 1
    return ScenarioProjection(
        horizon_p50=Decimal(str(forecast.spend.cumulative_at(last, P50))),
        horizon_p90=Decimal(str(forecast.spend.cumulative_at(last, P90))),
        projected_eom=_period_total(forecast, actual_to_date, month_end(as_of)),
        projected_eoq=_period_total(forecast, actual_to_date, quarter_end(as_of)),
    )


def _period_total(forecast: ScopeForecast, actual_to_date: Decimal, boundary: date) -> Decimal:
    offset: Final = forecast.spend.offset_of(boundary)
    if offset < 0:
        return actual_to_date
    return actual_to_date + Decimal(str(forecast.spend.cumulative_at(offset, P50)))


def compare(
    baseline: ScopeForecast,
    scenario: ScopeForecast,
    *,
    actual_to_date: Decimal,
    as_of: date,
) -> ScenarioComparison:
    baseline_projection: Final = project(baseline, actual_to_date=actual_to_date, as_of=as_of)
    scenario_projection: Final = project(scenario, actual_to_date=actual_to_date, as_of=as_of)
    delta: Final = scenario_projection.projected_eom - baseline_projection.projected_eom
    return ScenarioComparison(
        baseline=baseline_projection,
        scenario=scenario_projection,
        delta_usd=delta,
        delta_pct=(
            0.0 if baseline_projection.projected_eom <= 0 else float(delta / baseline_projection.projected_eom * 100)
        ),
    )
