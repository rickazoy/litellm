"""Persisting forecasts and runway, and reading them back for the API.

The API's p95 budget is 300ms (§1.14) and the forecast engine's cost is measured
in hundreds of milliseconds per scope, so nothing is ever computed on a read: the
nightly job writes rows and every endpoint reads them. That is also what makes
§0.4 hold, that WIT OS and the LiteLLM UI cannot show different numbers, since
both are reading the same persisted row rather than each recomputing from facts.

Superseding is a two-statement write rather than a delete. Old rows keep their
``forecast_json``, which is what the quality job scores yesterday's prediction
against a day later, and what makes "what did we think last Tuesday" answerable
at all.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from decimal import Decimal
from types import MappingProxyType
from typing import Final, TypedDict

from pydantic import TypeAdapter
from typing_extensions import ReadOnly

from litellm.proxy.witos.finops.drivers import DriverDecomposition
from litellm.proxy.witos.finops.engine import (
    ALGORITHM_VERSION,
    ScopeForecast,
    SeriesForecast,
    daily_burn_rate,
    month_end,
    quarter_end,
)
from litellm.proxy.witos.finops.history import ScopeRef
from litellm.proxy.witos.finops.quality import quality_score
from litellm.proxy.witos.finops.runway import RunwayResult
from litellm.proxy.witos.finops.simulation import PathEnsemble
from litellm.proxy.witos.shared.sql import SqlExecutor

# The metrics a forecast row is written for. `requests` rather than
# `request_count` because §1.4 names the metric that way and the quota table
# uses the same word.
PERSISTED_METRICS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "spend_usd": "spend_usd",
        "prompt_tokens": "prompt_tokens",
        "completion_tokens": "completion_tokens",
        "total_tokens": "total_tokens",
        "requests": "request_count",
    }
)

_FORECAST_COLUMNS: Final = (
    "forecast_id",
    "scope_type",
    "scope_id",
    "generated_at",
    "forecast_start",
    "forecast_end",
    "metric",
    "strategy",
    "algorithm",
    "algorithm_version",
    "history_days",
    "training_points",
    "trend_regime",
    "wape",
    "bias",
    "quality_score",
    "pricing_snapshot_hash",
    "forecast_json",
    "proj_eom",
    "proj_eoq",
    "burn_rate_daily",
    "is_latest",
)

_RUNWAY_COLUMNS: Final = (
    "id",
    "computed_at",
    "quota_id",
    "scope_type",
    "scope_id",
    "consumed",
    "remaining",
    "exhaustion_p50",
    "exhaustion_p90",
    "prob_exhaust_before_reset",
    "prob_overrun_this_period",
    "survives_cycle",
    "is_latest",
)

_SUPERSEDE_FORECAST_SQL: Final = """
UPDATE "WITOS_FinOpsForecast" SET is_latest = false
WHERE scope_type = $1 AND scope_id = $2 AND is_latest = true
"""

_SUPERSEDE_RUNWAY_SQL: Final = """
UPDATE "WITOS_FinOpsRunway" SET is_latest = false
WHERE scope_type = $1 AND scope_id = $2 AND is_latest = true
"""

_LATEST_FORECAST_SQL: Final = """
SELECT forecast_id, scope_type, scope_id, generated_at, forecast_start, forecast_end, metric, strategy,
       algorithm, algorithm_version, history_days, training_points, trend_regime, wape, bias, quality_score,
       pricing_snapshot_hash, forecast_json, proj_eom, proj_eoq, burn_rate_daily
FROM "WITOS_FinOpsForecast"
WHERE scope_type = $1 AND scope_id = $2 AND metric = $3 AND is_latest = true
ORDER BY generated_at DESC
LIMIT 1
"""

_TOP_FORECASTS_SQL: Final = """
SELECT forecast_id, scope_type, scope_id, generated_at, forecast_start, forecast_end, metric, strategy,
       algorithm, algorithm_version, history_days, training_points, trend_regime, wape, bias, quality_score,
       pricing_snapshot_hash, forecast_json, proj_eom, proj_eoq, burn_rate_daily
FROM "WITOS_FinOpsForecast"
WHERE metric = 'spend_usd' AND is_latest = true AND scope_type = ANY($1::text[])
ORDER BY proj_eom DESC NULLS LAST
LIMIT $2
"""

_LATEST_RUNWAY_SQL: Final = """
SELECT r.id, r.computed_at, r.quota_id, r.scope_type, r.scope_id, r.consumed, r.remaining,
       r.exhaustion_p50, r.exhaustion_p90, r.prob_exhaust_before_reset, r.prob_overrun_this_period,
       r.survives_cycle, q.metric, q.limit_value, q.period, q.reset_at, q.source_type
FROM "WITOS_FinOpsRunway" r
JOIN "WITOS_FinOpsQuota" q ON q.quota_id = r.quota_id
WHERE r.is_latest = true AND r.scope_type = $1 AND r.scope_id = $2
ORDER BY r.prob_exhaust_before_reset DESC NULLS LAST
"""

_SPEND_FORECASTS_SQL: Final = """
SELECT forecast_id, scope_type, scope_id, generated_at, forecast_start, forecast_end, metric, strategy,
       algorithm, algorithm_version, history_days, training_points, trend_regime, wape, bias, quality_score,
       pricing_snapshot_hash, forecast_json, proj_eom, proj_eoq, burn_rate_daily
FROM "WITOS_FinOpsForecast"
WHERE metric = 'spend_usd' AND is_latest = true AND scope_type = ANY($1::text[])
ORDER BY scope_type, scope_id
LIMIT $2
"""

_RUNWAY_BY_TYPES_SQL: Final = """
SELECT r.id, r.computed_at, r.quota_id, r.scope_type, r.scope_id, r.consumed, r.remaining,
       r.exhaustion_p50, r.exhaustion_p90, r.prob_exhaust_before_reset, r.prob_overrun_this_period,
       r.survives_cycle, q.metric, q.limit_value, q.period, q.reset_at, q.source_type
FROM "WITOS_FinOpsRunway" r
JOIN "WITOS_FinOpsQuota" q ON q.quota_id = r.quota_id
WHERE r.is_latest = true AND r.scope_type = ANY($1::text[])
ORDER BY r.scope_type, r.scope_id
"""

_LATEST_GENERATED_SQL: Final = """
SELECT MAX(generated_at) AS generated_at, COUNT(DISTINCT scope_id)::int AS scopes
FROM "WITOS_FinOpsForecast" WHERE is_latest = true
"""

_RUNWAY_RISKS_SQL: Final = """
SELECT r.id, r.computed_at, r.quota_id, r.scope_type, r.scope_id, r.consumed, r.remaining,
       r.exhaustion_p50, r.exhaustion_p90, r.prob_exhaust_before_reset, r.prob_overrun_this_period,
       r.survives_cycle, q.metric, q.limit_value, q.period, q.reset_at, q.source_type
FROM "WITOS_FinOpsRunway" r
JOIN "WITOS_FinOpsQuota" q ON q.quota_id = r.quota_id
WHERE r.is_latest = true AND r.survives_cycle = false
ORDER BY r.prob_exhaust_before_reset DESC NULLS LAST, r.exhaustion_p50 ASC NULLS LAST
LIMIT $1
"""


class ForecastRow(TypedDict):
    forecast_id: ReadOnly[str]
    scope_type: ReadOnly[str]
    scope_id: ReadOnly[str]
    generated_at: ReadOnly[datetime]
    forecast_start: ReadOnly[datetime]
    forecast_end: ReadOnly[datetime]
    metric: ReadOnly[str]
    strategy: ReadOnly[str]
    algorithm: ReadOnly[str]
    algorithm_version: ReadOnly[str]
    history_days: ReadOnly[int]
    training_points: ReadOnly[int]
    trend_regime: ReadOnly[str]
    wape: ReadOnly[Decimal | None]
    bias: ReadOnly[Decimal | None]
    quality_score: ReadOnly[int | None]
    pricing_snapshot_hash: ReadOnly[str | None]
    forecast_json: ReadOnly[Mapping[str, object]]
    proj_eom: ReadOnly[Decimal | None]
    proj_eoq: ReadOnly[Decimal | None]
    burn_rate_daily: ReadOnly[Decimal | None]


class RunwayRow(TypedDict):
    id: ReadOnly[str]
    computed_at: ReadOnly[datetime]
    quota_id: ReadOnly[str]
    scope_type: ReadOnly[str]
    scope_id: ReadOnly[str]
    consumed: ReadOnly[Decimal]
    remaining: ReadOnly[Decimal]
    exhaustion_p50: ReadOnly[datetime | None]
    exhaustion_p90: ReadOnly[datetime | None]
    prob_exhaust_before_reset: ReadOnly[Decimal | None]
    prob_overrun_this_period: ReadOnly[Decimal | None]
    survives_cycle: ReadOnly[bool]
    metric: ReadOnly[str]
    limit_value: ReadOnly[Decimal]
    period: ReadOnly[str]
    reset_at: ReadOnly[datetime | None]
    source_type: ReadOnly[str]


class _FreshnessRow(TypedDict):
    generated_at: ReadOnly[datetime | None]
    scopes: ReadOnly[int]


_FRESHNESS_ROWS: Final = TypeAdapter(tuple[_FreshnessRow, ...])
_FORECAST_ROWS: Final = TypeAdapter(tuple[ForecastRow, ...])
_RUNWAY_ROWS: Final = TypeAdapter(tuple[RunwayRow, ...])


@dataclass(frozen=True, slots=True)
class PersistedCounts:
    forecasts: int
    runways: int


class BandRow(TypedDict):
    date: ReadOnly[str]
    p10: ReadOnly[float]
    p50: ReadOnly[float]
    p90: ReadOnly[float]


class ComponentDoc(TypedDict):
    algorithm: ReadOnly[str]
    wape: ReadOnly[float | None]
    bias: ReadOnly[float | None]


class BacktestDoc(TypedDict):
    model: ReadOnly[str]
    wape: ReadOnly[float | None]
    bias: ReadOnly[float | None]
    folds: ReadOnly[int]
    mae: ReadOnly[float | None]


class BlendedPriceDoc(TypedDict):
    input_per_token: ReadOnly[str]
    output_per_token: ReadOnly[str]
    cache_read_per_token: ReadOnly[str]


class DriverDoc(TypedDict):
    factor: ReadOnly[str]
    label: ReadOnly[str]
    delta_usd: ReadOnly[float]
    delta_pct: ReadOnly[float]


class ForecastDocument(TypedDict):
    """``forecast_json`` as written. The API parses this back into ``ForecastPayload``."""

    days: ReadOnly[tuple[BandRow, ...]]
    cumulative: ReadOnly[tuple[BandRow, ...]]
    tier: ReadOnly[str]
    confidence: ReadOnly[str]
    strategy: ReadOnly[str]
    regime: ReadOnly[str]
    backtest: ReadOnly[BacktestDoc]
    components: ReadOnly[Mapping[str, ComponentDoc]]
    model_mix: ReadOnly[Mapping[str, float]]
    unknown_models: ReadOnly[tuple[str, ...]]
    unknown_model_share: ReadOnly[float]
    pricing_fidelity: ReadOnly[float | None]
    pricing_calibration: ReadOnly[float]
    blended_price: ReadOnly[BlendedPriceDoc]
    path_count: ReadOnly[int]
    residual_count: ReadOnly[int]
    has_band: ReadOnly[bool]
    seed: ReadOnly[int]
    metric: ReadOnly[str]
    drivers: ReadOnly[tuple[DriverDoc, ...]]


def _band_row(day: date, p10: float, p50: float, p90: float) -> BandRow:
    row: Final[BandRow] = {
        "date": day.isoformat(),
        "p10": round(p10, 6),
        "p50": round(p50, 6),
        "p90": round(p90, 6),
    }
    return row


def _band_rows(forecast: ScopeForecast, ensemble: PathEnsemble, *, cumulative: bool) -> tuple[BandRow, ...]:
    return tuple(
        _band_row(band.day, band.p10, band.p50, band.p90) for band in forecast.bands(ensemble, cumulative=cumulative)
    )


def _component(series: SeriesForecast) -> ComponentDoc:
    doc: Final[ComponentDoc] = {
        "algorithm": series.algorithm,
        "wape": series.wape,
        "bias": series.bias,
    }
    return doc


def _components(forecast: ScopeForecast) -> Mapping[str, ComponentDoc]:
    return MappingProxyType({name: _component(series) for name, series in forecast.series.items()})


def _driver_doc(factor: str, label: str, delta_usd: Decimal, delta_pct: float) -> DriverDoc:
    doc: Final[DriverDoc] = {
        "factor": factor,
        "label": label,
        "delta_usd": float(delta_usd),
        "delta_pct": round(delta_pct, 4),
    }
    return doc


def _drivers_payload(decomposition: DriverDecomposition | None) -> tuple[DriverDoc, ...]:
    if decomposition is None:
        return ()
    return tuple(
        _driver_doc(item.factor, item.label, item.delta_usd, item.delta_pct) for item in decomposition.contributions
    )


def forecast_payload(
    forecast: ScopeForecast,
    metric: str,
    ensemble: PathEnsemble,
    *,
    drivers: DriverDecomposition | None = None,
) -> ForecastDocument:
    """The ``forecast_json`` document: bands, provenance, and how much to trust it."""
    winning: Final = forecast.series.get("prompt_tokens") or next(iter(forecast.series.values()))
    backtest: Final[BacktestDoc] = {
        "model": winning.algorithm,
        "wape": winning.wape,
        "bias": winning.bias,
        "folds": winning.score.folds if winning.score else 0,
        "mae": winning.score.mae if winning.score else None,
    }
    blended: Final[BlendedPriceDoc] = {
        "input_per_token": str(forecast.blended_price.input_per_token),
        "output_per_token": str(forecast.blended_price.output_per_token),
        "cache_read_per_token": str(forecast.blended_price.cache_read_per_token),
    }
    document: Final[ForecastDocument] = {
        "days": _band_rows(forecast, ensemble, cumulative=False),
        "cumulative": _band_rows(forecast, ensemble, cumulative=True),
        "tier": forecast.tier,
        "confidence": forecast.confidence,
        "strategy": forecast.strategy,
        "regime": forecast.regime,
        "backtest": backtest,
        "components": _components(forecast),
        "model_mix": MappingProxyType({model: round(share, 6) for model, share in forecast.mix_shares.items()}),
        "unknown_models": tuple(sorted(forecast.price_book.unknown_models)),
        "unknown_model_share": round(forecast.price_book.unknown_share(forecast.mix_shares), 6),
        "pricing_fidelity": (
            None if forecast.pricing_fidelity.ratio is None else float(forecast.pricing_fidelity.ratio)
        ),
        "pricing_calibration": float(forecast.pricing_fidelity.calibration()),
        "blended_price": blended,
        "path_count": forecast.path_count,
        "residual_count": ensemble.residual_count,
        "has_band": ensemble.has_band,
        "seed": forecast.seed,
        "metric": metric,
        "drivers": _drivers_payload(drivers),
    }
    return document


# Every placeholder in a multi-row VALUES list carries its own cast. Postgres
# cannot infer the type of a parameter that only ever appears inside VALUES, and
# the timestamp columns are naive TIMESTAMP(3) holding UTC, so they take the
# `timestamptz AT TIME ZONE 'UTC'` form the rest of the WIT OS SQL uses.
_TIMESTAMP: Final = "({}::timestamptz AT TIME ZONE 'UTC')"

_FORECAST_TYPES: Final = (
    "{}::text",
    "{}::text",
    _TIMESTAMP,
    _TIMESTAMP,
    _TIMESTAMP,
    "{}::text",
    "{}::text",
    "{}::text",
    "{}::text",
    "{}::int",
    "{}::int",
    "{}::text",
    "{}::numeric",
    "{}::numeric",
    "{}::int",
    "{}::text",
    "{}::jsonb",
    "{}::numeric",
    "{}::numeric",
    "{}::numeric",
)

_RUNWAY_TYPES: Final = (
    _TIMESTAMP,
    "{}::text",
    "{}::text",
    "{}::text",
    "{}::numeric",
    "{}::numeric",
    _TIMESTAMP,
    _TIMESTAMP,
    "{}::numeric",
    "{}::numeric",
    "{}::boolean",
)


def _as_json(document: ForecastDocument) -> str:
    """Serialise the document, converting the read-only mappings inside it as it goes."""
    return json.dumps(document, default=_jsonable)


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return dict(value)  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]  # isinstance cannot parameterise Mapping  # mutable-ok: json's encoder needs a real dict
    raise TypeError(f"WIT OS forecast documents cannot hold {type(value).__name__}")


def _values_clause(rows: int, types: Sequence[str]) -> str:
    return ", ".join(
        "("
        + ", ".join(sql_type.format(f"${row * len(types) + column + 1}") for column, sql_type in enumerate(types))
        + ")"
        for row in range(rows)
    )


def _forecast_values(
    forecast: ScopeForecast,
    metric: str,
    ensemble: PathEnsemble,
    *,
    actual_to_date: Decimal,
    drivers: DriverDecomposition | None,
) -> tuple[object, ...]:
    winning: Final = forecast.series.get("prompt_tokens") or next(iter(forecast.series.values()))
    is_money: Final = metric == "spend_usd"
    return (
        forecast.scope.scope_type,
        forecast.scope.scope_id,
        forecast.generated_at,
        _at_midnight(forecast.forecast_start),
        _at_midnight(forecast.forecast_end),
        metric,
        forecast.strategy,
        "usage_priced" if is_money and forecast.strategy == "TOKEN_BASED" else winning.algorithm,
        ALGORITHM_VERSION,
        forecast.history_days,
        forecast.training_points,
        forecast.regime,
        None if winning.wape is None else Decimal(str(round(winning.wape, 4))),
        None if winning.bias is None else Decimal(str(round(winning.bias, 4))),
        quality_score(wape=winning.wape, history_days=forecast.history_days, regime=forecast.regime),
        forecast.price_book.snapshot_hash,
        _as_json(forecast_payload(forecast, metric, ensemble, drivers=drivers if is_money else None)),
        _period_total(forecast, ensemble, actual_to_date, month_end(forecast.forecast_start), money=is_money),
        _period_total(forecast, ensemble, actual_to_date, quarter_end(forecast.forecast_start), money=is_money),
        daily_burn_rate(forecast) if is_money else _mean_daily(ensemble),
    )


def _period_total(
    forecast: ScopeForecast,
    ensemble: PathEnsemble,
    actual_to_date: Decimal,
    boundary: date,
    *,
    money: bool,
) -> Decimal:
    offset: Final = ensemble.offset_of(boundary)
    base: Final = actual_to_date if money else Decimal(0)
    if offset < 0:
        return base
    return base + Decimal(str(ensemble.cumulative_at(offset, 0.5)))


def _at_midnight(day: date | None) -> datetime | None:
    """Date columns in the WIT OS models are ``DateTime``; bind them as UTC midnight, never as a bare date."""
    return None if day is None else datetime.combine(day, time.min, tzinfo=timezone.utc)


def _mean_daily(ensemble: PathEnsemble) -> Decimal:
    band: Final = ensemble.daily_quantiles((0.5,))[0]
    if not band:
        return Decimal(0)
    days: Final = min(7, len(band))
    return Decimal(str(sum(band[:days]) / days))


def _runway_values(result: RunwayResult, computed_at: datetime) -> tuple[object, ...]:
    return (
        computed_at,
        result.quota_id,
        result.scope.scope_type,
        result.scope.scope_id,
        result.consumed,
        result.remaining,
        _at_midnight(result.exhaustion_p50),
        _at_midnight(result.exhaustion_p90),
        Decimal(str(round(result.prob_exhaust_before_reset, 4))),
        Decimal(str(round(result.prob_overrun_this_period, 4))),
        result.survives_cycle,
    )


class ForecastStore:
    """Writes the nightly results and answers the API's reads."""

    def __init__(self, executor: SqlExecutor) -> None:
        self._executor: Final = executor

    async def save(
        self,
        forecast: ScopeForecast,
        *,
        actual_to_date: Decimal,
        runway: Sequence[RunwayResult] = (),
        drivers: DriverDecomposition | None = None,
    ) -> PersistedCounts:
        """Supersede this scope's current rows and insert the new ones."""
        await self._executor.execute(_SUPERSEDE_FORECAST_SQL, forecast.scope.scope_type, forecast.scope.scope_id)
        payloads: Final = tuple(
            (metric, forecast.ensemble_for(series_name))
            for metric, series_name in PERSISTED_METRICS.items()
            if forecast.ensemble_for(series_name) is not None
        )
        rows: Final = tuple(
            _forecast_values(forecast, metric, ensemble, actual_to_date=actual_to_date, drivers=drivers)
            for metric, ensemble in payloads
            if ensemble is not None
        )
        forecasts: Final = await self._insert_forecasts(rows)
        return PersistedCounts(forecasts=forecasts, runways=await self._save_runway(forecast, runway))

    async def supersede(self, scope: ScopeRef) -> None:
        """Retire a scope's rows without writing new ones.

        A scope that has gone dormant, or dropped below three days of usable
        history, must stop having a current forecast. Leaving the old row marked
        latest would keep an increasingly stale projection on the dashboard with
        nothing to indicate it had stopped being maintained.
        """
        await self._executor.execute(_SUPERSEDE_FORECAST_SQL, scope.scope_type, scope.scope_id)
        await self._executor.execute(_SUPERSEDE_RUNWAY_SQL, scope.scope_type, scope.scope_id)

    async def _insert_forecasts(self, rows: Sequence[tuple[object, ...]]) -> int:
        return await self._insert(
            table="WITOS_FinOpsForecast", columns=_FORECAST_COLUMNS, types=_FORECAST_TYPES, rows=rows
        )

    async def _save_runway(self, forecast: ScopeForecast, runway: Sequence[RunwayResult]) -> int:
        await self._executor.execute(_SUPERSEDE_RUNWAY_SQL, forecast.scope.scope_type, forecast.scope.scope_id)
        return await self._insert(
            table="WITOS_FinOpsRunway",
            columns=_RUNWAY_COLUMNS,
            types=_RUNWAY_TYPES,
            rows=tuple(_runway_values(result, forecast.generated_at) for result in runway),
        )

    async def _insert(
        self,
        *,
        table: str,
        columns: Sequence[str],
        types: Sequence[str],
        rows: Sequence[tuple[object, ...]],
    ) -> int:
        """One statement per scope rather than one per row: 2,000 scopes, not 10,000 round trips."""
        if not rows:
            return 0
        statement: Final = (
            f'INSERT INTO "{table}" ({", ".join(columns)}) '
            f"SELECT gen_random_uuid()::text, v.*, true FROM (VALUES {_values_clause(len(rows), types)}) "
            f"AS v({', '.join(columns[1:-1])})"
        )
        return await self._executor.execute(statement, *(value for row in rows for value in row))

    async def latest(self, scope: ScopeRef, metric: str) -> ForecastRow | None:
        rows: Final = _FORECAST_ROWS.validate_python(
            await self._executor.query(_LATEST_FORECAST_SQL, scope.scope_type, scope.scope_id, metric)
        )
        return rows[0] if rows else None

    async def top_by_projection(self, *, scope_types: Sequence[str], limit: int) -> tuple[ForecastRow, ...]:
        scopes: Final = list(scope_types)  # mutable-ok: an array bind needs a list, not a tuple
        return _FORECAST_ROWS.validate_python(await self._executor.query(_TOP_FORECASTS_SQL, scopes, limit))

    async def runway_for(self, scope: ScopeRef) -> tuple[RunwayRow, ...]:
        return _RUNWAY_ROWS.validate_python(
            await self._executor.query(_LATEST_RUNWAY_SQL, scope.scope_type, scope.scope_id)
        )

    async def runway_risks(self, *, limit: int) -> tuple[RunwayRow, ...]:
        return _RUNWAY_ROWS.validate_python(await self._executor.query(_RUNWAY_RISKS_SQL, limit))

    async def spend_forecasts(self, *, scope_types: Sequence[str], limit: int) -> tuple[ForecastRow, ...]:
        scopes: Final = list(scope_types)  # mutable-ok: an array bind needs a list, not a tuple
        return _FORECAST_ROWS.validate_python(await self._executor.query(_SPEND_FORECASTS_SQL, scopes, limit))

    async def runway_by_types(self, *, scope_types: Sequence[str]) -> tuple[RunwayRow, ...]:
        scopes: Final = list(scope_types)  # mutable-ok: an array bind needs a list, not a tuple
        return _RUNWAY_ROWS.validate_python(await self._executor.query(_RUNWAY_BY_TYPES_SQL, scopes))

    async def freshness(self) -> tuple[datetime | None, int]:
        """When the newest current forecast was generated, and how many scopes have one."""
        rows: Final = _FRESHNESS_ROWS.validate_python(await self._executor.query(_LATEST_GENERATED_SQL))
        return (rows[0]["generated_at"], rows[0]["scopes"]) if rows else (None, 0)
