"""``/witos/finops/*`` (blueprint §1.11).

Every read is a read. The nightly job (``jobs.py``) computes forecasts, runway
and drivers and writes them down; these endpoints select rows and shape them.
That is what holds the 300ms p95 in §1.14, and it is also what makes §0.4 true:
the LiteLLM UI and WIT OS are looking at one persisted number rather than each
deriving their own.

Two endpoints deliberately compute. ``/drivers`` runs a counterfactual over two
windows of the fact table, which is a handful of aggregate rows, and
``/scenarios/evaluate`` rebuilds the scope's forecast so it can replay it under
the assumptions. The scenario budget in §1.14 is two seconds, which the engine
fits inside comfortably; the alternative would be persisting an answer to a
question nobody has asked yet.

RBAC is applied twice on every route: the key's scope string decides what it may
do, and the entity clamp decides whose numbers it may see. See
``witos/shared/rbac.py``.
"""

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from types import MappingProxyType
from typing import Annotated, Final, Literal

import fastapi
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import PlainTextResponse, StreamingResponse

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.witos.finops.alerts import AlertStore
from litellm.proxy.witos.finops.api_schemas import (
    AdminStatusResponse,
    AlertEventModel,
    AlertEventUpdate,
    AlertRuleCreate,
    AlertRuleModel,
    AlertRuleUpdate,
    BandPoint,
    DriverLine,
    DriversResponse,
    ForecastPayload,
    ForecastQualityResponse,
    ForecastResponse,
    ModelMixEntry,
    ModelMixResponse,
    OverviewCard,
    OverviewResponse,
    QualityPoint,
    QuotaCreate,
    QuotaModel,
    QuotaUpdate,
    RebuildRequest,
    RebuildResponse,
    RunwayItem,
    RunwayResponse,
    ScenarioCreate,
    ScenarioEvaluateRequest,
    ScenarioEvaluateResponse,
    ScenarioModel,
    ScenarioSide,
    SeriesPointModel,
    TimeseriesResponse,
)
from litellm.proxy.witos.finops.drivers import decompose
from litellm.proxy.witos.finops.engine import (
    ALGORITHM_VERSION,
    NoForecast,
    ScopeForecast,
    build_forecast,
    month_end,
)
from litellm.proxy.witos.finops.facts import FactReader, ScopeTotals
from litellm.proxy.witos.finops.forecast_store import ForecastRow, ForecastStore, RunwayRow
from litellm.proxy.witos.finops.history import ScopeRef
from litellm.proxy.witos.finops.pricing import build_price_book
from litellm.proxy.witos.finops.quality import QualityScorer
from litellm.proxy.witos.finops.quota_store import QuotaDraft, QuotaStore
from litellm.proxy.witos.finops.report import build_csv
from litellm.proxy.witos.finops.runway import Quota, RunwayResult, compute_runway, runway_statement
from litellm.proxy.witos.finops.scenario_store import ScenarioStore, assumptions_of
from litellm.proxy.witos.finops.scenarios import (
    ScenarioAssumptions,
    ScenarioProjection,
    apply_scenario,
    compare,
)
from litellm.proxy.witos.finops.stream import FinOpsStream
from litellm.proxy.witos.shared.rbac import (
    forbidden,
    require_permission,
    resolve_scope,
    visible_scope_types,
)
from litellm.proxy.witos.shared.sql import PrismaSqlExecutor, SqlExecutor

_TAGS: Final[list[str | Enum]] = ["WIT OS FinOps"]  # mutable-ok: fastapi types the tags argument as a list

router: Final = APIRouter(prefix="/witos/finops", tags=_TAGS)

_DEFAULT_TOP_SCOPES: Final = 10
_DEFAULT_RISK_LIMIT: Final = 10
_MAX_REPORT_ROWS: Final = 5000
_DEFAULT_DRIVER_WINDOW: Final = 7


def finops_executor() -> SqlExecutor:
    """The proxy's Prisma client, or a 503 that says what is missing.

    A FastAPI dependency rather than a module-level lookup so a test can supply
    its own executor through ``dependency_overrides`` instead of reaching into
    the proxy's globals.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise _fail(status.HTTP_503_SERVICE_UNAVAILABLE, "WIT OS FinOps needs a database; connect one to the proxy")
    return PrismaSqlExecutor.for_proxy(prisma_client)  # pyright: ignore[reportArgumentType]  # PrismaWrapper forwards query_raw/execute_raw through __getattr__, which a Protocol cannot see


def _fail(code: int, message: str) -> HTTPException:
    """The one place a FinOps error body is shaped, so every route refuses identically."""
    detail: Final = {"error": message}  # mutable-ok: FastAPI serialises the detail body it is given
    return HTTPException(status_code=code, detail=detail)


def _headers(pairs: Sequence[tuple[str, str]]) -> Mapping[str, str]:
    return MappingProxyType(dict(pairs))


def _yesterday(now: datetime | None = None) -> date:
    return (now or datetime.now(timezone.utc)).date() - timedelta(days=1)


def _as_date(value: datetime | None) -> date | None:
    return None if value is None else value.date()


def _payload(row: ForecastRow) -> ForecastPayload:
    return ForecastPayload.model_validate(row["forecast_json"])


def _forecast_response(scope: ScopeRef, metric: str, row: ForecastRow | None) -> ForecastResponse:
    if row is None:
        return ForecastResponse(
            scope_type=scope.scope_type,
            scope_id=scope.scope_id,
            metric=metric,
            state="no_forecast",
            detail="No current forecast for this scope; it may be new, dormant, or awaiting the nightly run",
        )
    payload: Final = _payload(row)
    return ForecastResponse(
        scope_type=row["scope_type"],
        scope_id=row["scope_id"],
        metric=row["metric"],
        state="forecast",
        generated_at=row["generated_at"],
        forecast_start=_as_date(row["forecast_start"]),
        forecast_end=_as_date(row["forecast_end"]),
        strategy=row["strategy"],
        algorithm=row["algorithm"],
        algorithm_version=row["algorithm_version"],
        trend_regime=row["trend_regime"],
        history_days=row["history_days"],
        training_points=row["training_points"],
        wape=None if row["wape"] is None else float(row["wape"]),
        bias=None if row["bias"] is None else float(row["bias"]),
        quality_score=row["quality_score"],
        pricing_snapshot_hash=row["pricing_snapshot_hash"],
        proj_eom=row["proj_eom"],
        proj_eoq=row["proj_eoq"],
        burn_rate_daily=row["burn_rate_daily"],
        payload=payload,
    )


def _runway_item(row: RunwayRow) -> RunwayItem:
    consumed_pct: Final = 0.0 if row["limit_value"] <= 0 else float(row["consumed"] / row["limit_value"] * 100)
    return RunwayItem(
        quota_id=row["quota_id"],
        metric=row["metric"],
        period=row["period"],
        limit_value=row["limit_value"],
        consumed=row["consumed"],
        remaining=row["remaining"],
        consumed_pct=consumed_pct,
        exhaustion_p50=_as_date(row["exhaustion_p50"]),
        exhaustion_p90=_as_date(row["exhaustion_p90"]),
        prob_exhaust_before_reset=(
            None if row["prob_exhaust_before_reset"] is None else float(row["prob_exhaust_before_reset"])
        ),
        prob_overrun_this_period=(
            None if row["prob_overrun_this_period"] is None else float(row["prob_overrun_this_period"])
        ),
        survives_cycle=row["survives_cycle"],
        reset_at=row["reset_at"],
        source_type=row["source_type"],
        statement=_statement_from(row),
    )


def _statement_from(row: RunwayRow) -> str:
    return runway_statement(
        RunwayResult(
            quota_id=row["quota_id"],
            scope=ScopeRef(scope_type=row["scope_type"], scope_id=row["scope_id"]),
            metric=row["metric"],
            limit_value=row["limit_value"],
            consumed=row["consumed"],
            remaining=row["remaining"],
            exhaustion_p50=_as_date(row["exhaustion_p50"]),
            exhaustion_p90=_as_date(row["exhaustion_p90"]),
            prob_exhaust_before_reset=float(row["prob_exhaust_before_reset"] or 0),
            prob_overrun_this_period=float(row["prob_overrun_this_period"] or 0),
            survives_cycle=row["survives_cycle"],
            projected_consumption_pct=None,
            burn_per_day=Decimal(0),
            cycle_end=_as_date(row["reset_at"]),
            period=row["period"],
        )
    )


def _quota_model(quota: Quota) -> QuotaModel:
    return QuotaModel(
        quota_id=quota.quota_id,
        scope_type=quota.scope.scope_type,
        scope_id=quota.scope.scope_id,
        metric=quota.metric,
        limit_value=quota.limit_value,
        period=quota.period,
        period_start=quota.period_start,
        period_end=quota.period_end,
        reset_at=quota.reset_at,
        source_type=quota.source_type,
        soft_threshold_pct=quota.soft_threshold_pct,
        read_only=quota.is_read_only,
    )


@router.get("/overview", response_model=OverviewResponse)
async def get_overview(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    executor: Annotated[SqlExecutor, Depends(finops_executor)],
    limit: Annotated[int, fastapi.Query(ge=1, le=100)] = _DEFAULT_TOP_SCOPES,
) -> OverviewResponse:
    """Fleet headline: month to date, projected month end with its P90, top runway risks, open alerts."""
    require_permission(user_api_key_dict, "finops:view")
    store: Final = ForecastStore(executor)
    scope_types: Final = visible_scope_types(user_api_key_dict)
    top: Final = await store.top_by_projection(scope_types=scope_types, limit=limit)
    risks: Final = await store.runway_risks(limit=_DEFAULT_RISK_LIMIT)
    as_of: Final = _yesterday()
    actuals: Final = await FactReader(executor).spend_by_scope(
        start=as_of.replace(day=1), end=as_of, scope_types=("global",)
    )
    global_row: Final = await store.latest(ScopeRef(scope_type="global", scope_id="global"), "spend_usd")
    mtd: Final = actuals.get(("global", "global"), Decimal(0))
    return OverviewResponse(
        generated_at=global_row["generated_at"] if global_row else None,
        mtd_spend=mtd,
        proj_eom_p50=global_row["proj_eom"] if global_row and global_row["proj_eom"] else mtd,
        proj_eom_p90=mtd + _eom_p90(global_row, as_of) if global_row else mtd,
        top_scopes=tuple(
            OverviewCard(
                scope_type=row["scope_type"],
                scope_id=row["scope_id"],
                proj_eom=row["proj_eom"],
                burn_rate_daily=row["burn_rate_daily"],
                trend_regime=row["trend_regime"],
                quality_score=row["quality_score"],
            )
            for row in top
        ),
        runway_risks=tuple(_runway_item(row) for row in risks),
        open_alerts=await AlertStore(executor).open_event_count(),
    )


def _eom_p90(row: ForecastRow | None, as_of: date) -> Decimal:
    if row is None:
        return Decimal(0)
    boundary: Final = month_end(as_of)
    points: Final = tuple(point for point in _payload(row).cumulative if point.date <= boundary)
    return Decimal(str(points[-1].p90)) if points else Decimal(0)


@router.get("/timeseries", response_model=TimeseriesResponse)
async def get_timeseries(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    executor: Annotated[SqlExecutor, Depends(finops_executor)],
    scope_type: str,
    scope_id: str | None = None,
    grain: Literal["day", "hour"] = "day",
    days: Annotated[int, fastapi.Query(ge=1, le=365)] = 90,
) -> TimeseriesResponse:
    """Actuals from the fact tables, for the solid part of the chart."""
    require_permission(user_api_key_dict, "finops:view")
    scope: Final = resolve_scope(user_api_key_dict, scope_type=scope_type, scope_id=scope_id)
    end: Final = datetime.now(timezone.utc)
    points: Final = await FactReader(executor).series(scope, start=end - timedelta(days=days), end=end, grain=grain)
    return TimeseriesResponse(
        scope_type=scope.scope_type,
        scope_id=scope.scope_id,
        grain=grain,
        points=tuple(
            SeriesPointModel(
                bucket=point.bucket,
                request_count=point.request_count,
                prompt_tokens=point.prompt_tokens,
                completion_tokens=point.completion_tokens,
                total_tokens=point.total_tokens,
                cache_read_tokens=point.cache_read_tokens,
                spend_usd=point.spend_usd,
            )
            for point in points
        ),
    )


@router.get("/forecast", response_model=ForecastResponse)
async def get_forecast(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    executor: Annotated[SqlExecutor, Depends(finops_executor)],
    scope_type: str,
    scope_id: str | None = None,
    metric: str = "spend_usd",
) -> ForecastResponse:
    """The persisted forecast: daily and cumulative P10/P50/P90, headlines, quality and drivers.

    A scope with under three days of history has no forecast at all and says so
    in ``state`` (§1.6); it never returns a band of zeros.
    """
    require_permission(user_api_key_dict, "finops:view")
    scope: Final = resolve_scope(user_api_key_dict, scope_type=scope_type, scope_id=scope_id)
    return _forecast_response(scope, metric, await ForecastStore(executor).latest(scope, metric))


@router.get("/runway", response_model=RunwayResponse)
async def get_runway(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    executor: Annotated[SqlExecutor, Depends(finops_executor)],
    scope_type: str,
    scope_id: str | None = None,
) -> RunwayResponse:
    """Per-quota exhaustion dates and probabilities, from the forecast's own Monte Carlo paths."""
    require_permission(user_api_key_dict, "finops:view")
    scope: Final = resolve_scope(user_api_key_dict, scope_type=scope_type, scope_id=scope_id)
    rows: Final = await ForecastStore(executor).runway_for(scope)
    return RunwayResponse(
        scope_type=scope.scope_type,
        scope_id=scope.scope_id,
        computed_at=rows[0]["computed_at"] if rows else None,
        quotas=tuple(_runway_item(row) for row in rows),
    )


@router.get("/drivers", response_model=DriversResponse)
async def get_drivers(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    executor: Annotated[SqlExecutor, Depends(finops_executor)],
    scope_type: str,
    scope_id: str | None = None,
    window_days: Annotated[int, fastapi.Query(ge=1, le=90)] = _DEFAULT_DRIVER_WINDOW,
) -> DriversResponse:
    """Counterfactual decomposition of the change in spend, with an explicit residual line."""
    require_permission(user_api_key_dict, "finops:view")
    scope: Final = resolve_scope(user_api_key_dict, scope_type=scope_type, scope_id=scope_id)
    as_of: Final = _yesterday()
    history: Final = await FactReader(executor).history(scope, as_of=as_of, days=window_days * 2)
    if history is None or history.history_days < window_days * 2:
        raise _fail(
            status.HTTP_409_CONFLICT,
            f"Need {window_days * 2} days of history to compare two {window_days}-day windows",
        )
    book: Final = build_price_book(tuple(history.mix_tokens_total(window_days=window_days * 2)))
    result: Final = decompose(history.days[-window_days * 2 : -window_days], history.days[-window_days:], book)
    return DriversResponse(
        scope_type=scope.scope_type,
        scope_id=scope.scope_id,
        window_days=window_days,
        baseline_daily_spend=result.baseline_spend,
        current_daily_spend=result.current_spend,
        delta_usd=result.delta_usd,
        delta_pct=result.delta_pct,
        contributions=tuple(
            DriverLine(
                factor=item.factor,
                label=item.label,
                delta_usd=float(item.delta_usd),
                delta_pct=item.delta_pct,
            )
            for item in result.contributions
        ),
    )


@router.get("/model-mix", response_model=ModelMixResponse)
async def get_model_mix(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    executor: Annotated[SqlExecutor, Depends(finops_executor)],
    scope_type: str,
    scope_id: str | None = None,
) -> ModelMixResponse:
    """Forecast model-mix shares, and how much of the mix LiteLLM cannot price."""
    require_permission(user_api_key_dict, "finops:view")
    scope: Final = resolve_scope(user_api_key_dict, scope_type=scope_type, scope_id=scope_id)
    row: Final = await ForecastStore(executor).latest(scope, "spend_usd")
    if row is None:
        return ModelMixResponse(
            scope_type=scope.scope_type, scope_id=scope.scope_id, forecast_shares=(), unknown_model_share=0.0
        )
    payload: Final = _payload(row)
    return ModelMixResponse(
        scope_type=scope.scope_type,
        scope_id=scope.scope_id,
        forecast_shares=tuple(
            ModelMixEntry(model_group=model, token_share=share, priced=model not in payload.unknown_models)
            for model, share in sorted(payload.model_mix.items(), key=lambda item: -item[1])
        ),
        unknown_model_share=payload.unknown_model_share,
    )


@router.get("/forecast-quality", response_model=ForecastQualityResponse)
async def get_forecast_quality(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    executor: Annotated[SqlExecutor, Depends(finops_executor)],
    scope_type: str,
    scope_id: str | None = None,
    metric: str = "spend_usd",
) -> ForecastQualityResponse:
    """Rolling one-day-ahead WAPE and bias: how accurate this engine has actually been here."""
    require_permission(user_api_key_dict, "finops:view")
    scope: Final = resolve_scope(user_api_key_dict, scope_type=scope_type, scope_id=scope_id)
    accuracy: Final = tuple(
        item for item in await QualityScorer(executor).measure(as_of=_yesterday(), scope=scope) if item.metric == metric
    )
    row: Final = await ForecastStore(executor).latest(scope, metric)
    measured: Final = accuracy[0] if accuracy else None
    return ForecastQualityResponse(
        scope_type=scope.scope_type,
        scope_id=scope.scope_id,
        metric=metric,
        rolling_wape=measured.wape if measured else None,
        rolling_bias=measured.bias if measured else None,
        sample_days=measured.sample_days if measured else 0,
        trend_regime=row["trend_regime"] if row else None,
        quality_score=row["quality_score"] if row else None,
        points=tuple(
            QualityPoint(date=point.day, predicted=point.predicted, actual=point.actual)
            for point in (measured.points if measured else ())
        ),
    )


@router.get("/quotas", response_model=tuple[QuotaModel, ...])
async def list_quotas(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    executor: Annotated[SqlExecutor, Depends(finops_executor)],
    scope_type: str,
    scope_id: str | None = None,
) -> tuple[QuotaModel, ...]:
    require_permission(user_api_key_dict, "finops:view")
    scope: Final = resolve_scope(user_api_key_dict, scope_type=scope_type, scope_id=scope_id)
    return tuple(_quota_model(quota) for quota in await QuotaStore(executor).for_scope(scope))


@router.post("/quotas", response_model=QuotaModel, status_code=status.HTTP_201_CREATED)
async def create_quota(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    executor: Annotated[SqlExecutor, Depends(finops_executor)],
    payload: QuotaCreate,
) -> QuotaModel:
    """Create a manual or contract quota. Budgets mirrored from LiteLLM are managed there."""
    require_permission(user_api_key_dict, "finops:manage_budget")
    scope: Final = resolve_scope(user_api_key_dict, scope_type=payload.scope_type, scope_id=payload.scope_id)
    created: Final = await QuotaStore(executor).create(
        QuotaDraft(
            scope=scope,
            metric=payload.metric,
            limit_value=payload.limit_value,
            period=payload.period,
            period_start=payload.period_start,
            period_end=payload.period_end,
            reset_at=payload.reset_at,
            soft_threshold_pct=payload.soft_threshold_pct,
        )
    )
    return _quota_model(created)


@router.patch("/quotas/{quota_id}", response_model=QuotaModel)
async def update_quota(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    executor: Annotated[SqlExecutor, Depends(finops_executor)],
    quota_id: str,
    payload: QuotaUpdate,
) -> QuotaModel:
    require_permission(user_api_key_dict, "finops:manage_budget")
    result: Final = await QuotaStore(executor).update(
        quota_id,
        limit_value=payload.limit_value,
        period=payload.period,
        period_end=payload.period_end,
        reset_at=payload.reset_at,
        soft_threshold_pct=payload.soft_threshold_pct,
    )
    if isinstance(result, str):
        raise _quota_refusal(result)
    return _quota_model(result)


@router.delete("/quotas/{quota_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_quota(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    executor: Annotated[SqlExecutor, Depends(finops_executor)],
    quota_id: str,
) -> None:
    require_permission(user_api_key_dict, "finops:manage_budget")
    result: Final = await QuotaStore(executor).delete(quota_id)
    if result != "deleted":
        raise _quota_refusal(result)


def _quota_refusal(reason: str) -> HTTPException:
    if reason == "not_found":
        return _fail(status.HTTP_404_NOT_FOUND, "No such quota")
    return _fail(
        status.HTTP_409_CONFLICT,
        "This quota mirrors a LiteLLM budget and is read-only here; edit the budget in LiteLLM",
    )


@router.get("/scenarios", response_model=tuple[ScenarioModel, ...])
async def list_scenarios(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    executor: Annotated[SqlExecutor, Depends(finops_executor)],
    scope_type: str | None = None,
    scope_id: str | None = None,
) -> tuple[ScenarioModel, ...]:
    require_permission(user_api_key_dict, "finops:view")
    scope: Final = resolve_scope(user_api_key_dict, scope_type=scope_type, scope_id=scope_id) if scope_type else None
    rows: Final = await ScenarioStore(executor).list(scope)
    return tuple(
        ScenarioModel(
            scenario_id=row["scenario_id"],
            name=row["name"],
            description=row["description"],
            scope_type=row["scope_type"],
            scope_id=row["scope_id"],
            assumptions=assumptions_of(row),
            created_by=row["created_by"],
            created_at=row["created_at"],
        )
        for row in rows
    )


@router.post("/scenarios", response_model=ScenarioModel, status_code=status.HTTP_201_CREATED)
async def create_scenario(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    executor: Annotated[SqlExecutor, Depends(finops_executor)],
    payload: ScenarioCreate,
) -> ScenarioModel:
    require_permission(user_api_key_dict, "finops:run_scenarios")
    scope: Final = resolve_scope(user_api_key_dict, scope_type=payload.scope_type, scope_id=payload.scope_id)
    row: Final = await ScenarioStore(executor).create(
        name=payload.name,
        description=payload.description,
        scope=scope,
        assumptions=payload.assumptions,
        created_by=user_api_key_dict.user_id or "unknown",
    )
    return ScenarioModel(
        scenario_id=row["scenario_id"],
        name=row["name"],
        description=row["description"],
        scope_type=row["scope_type"],
        scope_id=row["scope_id"],
        assumptions=assumptions_of(row),
        created_by=row["created_by"],
        created_at=row["created_at"],
    )


@router.delete("/scenarios/{scenario_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_scenario(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    executor: Annotated[SqlExecutor, Depends(finops_executor)],
    scenario_id: str,
) -> None:
    require_permission(user_api_key_dict, "finops:run_scenarios")
    if not await ScenarioStore(executor).delete(scenario_id):
        raise _fail(status.HTTP_404_NOT_FOUND, "No such scenario")


@router.post("/scenarios/evaluate", response_model=ScenarioEvaluateResponse)
async def evaluate_scenario(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    executor: Annotated[SqlExecutor, Depends(finops_executor)],
    payload: ScenarioEvaluateRequest,
) -> ScenarioEvaluateResponse:
    """Replay the scope's forecast under the assumptions and reprice it exactly.

    Both sides are computed from the same rebuild, so the delta is exact.
    Deterministic given the pricing snapshot hash in the response.
    """
    require_permission(user_api_key_dict, "finops:run_scenarios")
    scope: Final = resolve_scope(user_api_key_dict, scope_type=payload.scope_type, scope_id=payload.scope_id)
    assumptions: Final = await _assumptions_for(executor, payload)
    if assumptions.requires_baseline_users():
        raise _fail(
            status.HTTP_400_BAD_REQUEST,
            "new_users needs baseline_users: without it there is no per-user usage to scale",
        )
    as_of: Final = _yesterday()
    baseline: Final = await _rebuild(executor, scope, as_of=as_of)
    scenario: Final = apply_scenario(baseline, assumptions)
    consumed: Final = await FactReader(executor).totals(scope, start=as_of.replace(day=1), end=as_of)
    comparison: Final = compare(baseline, scenario, actual_to_date=consumed.spend_usd, as_of=as_of)
    quotas: Final = await QuotaStore(executor).for_scope(scope)
    return ScenarioEvaluateResponse(
        scope_type=scope.scope_type,
        scope_id=scope.scope_id,
        pricing_snapshot_hash=scenario.price_book.snapshot_hash,
        seed=scenario.seed,
        baseline=_side(baseline, comparison.baseline),
        scenario=_side(scenario, comparison.scenario),
        delta_usd=comparison.delta_usd,
        delta_pct=comparison.delta_pct,
        savings_usd=comparison.savings_usd,
        runway=_scenario_runway(scenario, quotas, consumed),
    )


async def _assumptions_for(executor: SqlExecutor, payload: ScenarioEvaluateRequest) -> ScenarioAssumptions:
    if payload.assumptions is not None:
        return payload.assumptions
    if payload.scenario_id is None:
        raise _fail(status.HTTP_400_BAD_REQUEST, "Pass either assumptions or scenario_id")
    saved: Final = await ScenarioStore(executor).by_id(payload.scenario_id)
    if saved is None:
        raise _fail(status.HTTP_404_NOT_FOUND, "No such scenario")
    return assumptions_of(saved)


async def _rebuild(executor: SqlExecutor, scope: ScopeRef, *, as_of: date) -> ScopeForecast:
    history: Final = await FactReader(executor).history(scope, as_of=as_of)
    if history is None:
        raise _fail(status.HTTP_409_CONFLICT, "No usage history for this scope")
    built: Final = build_forecast(history, as_of=as_of)
    if isinstance(built, NoForecast):
        raise _fail(status.HTTP_409_CONFLICT, f"{built.detail} (state: {built.state})")
    return built


def _side(forecast: ScopeForecast, projection: ScenarioProjection) -> ScenarioSide:
    return ScenarioSide(
        horizon_p50=projection.horizon_p50,
        horizon_p90=projection.horizon_p90,
        projected_eom=projection.projected_eom,
        projected_eoq=projection.projected_eoq,
        days=tuple(
            BandPoint(date=band.day, p10=band.p10, p50=band.p50, p90=band.p90)
            for band in forecast.bands(forecast.spend, cumulative=False)
        ),
    )


def _scenario_runway(forecast: ScopeForecast, quotas: Sequence[Quota], consumed: ScopeTotals) -> tuple[RunwayItem, ...]:
    """Runway under the scenario, against the same consumption the baseline is measured from."""
    results: Final = tuple(
        result
        for result in (
            compute_runway(forecast, quota, consumed.of(quota.metric), today=forecast.forecast_start)
            for quota in quotas
        )
        if result is not None
    )
    return tuple(
        RunwayItem(
            quota_id=result.quota_id,
            metric=result.metric,
            period=result.period,
            limit_value=result.limit_value,
            consumed=result.consumed,
            remaining=result.remaining,
            consumed_pct=result.consumed_pct,
            exhaustion_p50=result.exhaustion_p50,
            exhaustion_p90=result.exhaustion_p90,
            prob_exhaust_before_reset=result.prob_exhaust_before_reset,
            prob_overrun_this_period=result.prob_overrun_this_period,
            survives_cycle=result.survives_cycle,
            reset_at=None,
            source_type="scenario",
            statement=runway_statement(result),
        )
        for result in results
    )


@router.get("/alerts", response_model=tuple[AlertRuleModel, ...])
async def list_alert_rules(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    executor: Annotated[SqlExecutor, Depends(finops_executor)],
) -> tuple[AlertRuleModel, ...]:
    require_permission(user_api_key_dict, "finops:view")
    rules: Final = await AlertStore(executor).rules()
    return tuple(
        AlertRuleModel(
            alert_id=rule.alert_id,
            name=rule.name,
            scope_type=rule.scope_type,
            scope_id=rule.scope_id,
            condition_type=rule.condition_type,
            threshold=rule.threshold,
            lookahead_days=rule.lookahead_days,
            cooldown_min=rule.cooldown_min,
            enabled=rule.enabled,
            last_triggered_at=rule.last_triggered_at,
        )
        for rule in rules
    )


@router.post("/alerts", response_model=AlertRuleModel, status_code=status.HTTP_201_CREATED)
async def create_alert_rule(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    executor: Annotated[SqlExecutor, Depends(finops_executor)],
    payload: AlertRuleCreate,
) -> AlertRuleModel:
    """Create a rule. Webhook channels name an env secret; the secret itself never reaches the database."""
    require_permission(user_api_key_dict, "finops:manage_alerts")
    rule: Final = await AlertStore(executor).create_rule(
        name=payload.name,
        scope_type=payload.scope_type,
        scope_id=payload.scope_id,
        condition_type=payload.condition_type,
        threshold=payload.threshold,
        lookahead_days=payload.lookahead_days,
        channels=payload.channels,
        cooldown_min=payload.cooldown_min,
        enabled=payload.enabled,
    )
    return AlertRuleModel(
        alert_id=rule.alert_id,
        name=rule.name,
        scope_type=rule.scope_type,
        scope_id=rule.scope_id,
        condition_type=rule.condition_type,
        threshold=rule.threshold,
        lookahead_days=rule.lookahead_days,
        cooldown_min=rule.cooldown_min,
        enabled=rule.enabled,
        last_triggered_at=rule.last_triggered_at,
    )


@router.patch("/alerts/{alert_id}", response_model=AlertRuleModel)
async def update_alert_rule(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    executor: Annotated[SqlExecutor, Depends(finops_executor)],
    alert_id: str,
    payload: AlertRuleUpdate,
) -> AlertRuleModel:
    require_permission(user_api_key_dict, "finops:manage_alerts")
    rule: Final = await AlertStore(executor).update_rule(
        alert_id,
        threshold=payload.threshold,
        lookahead_days=payload.lookahead_days,
        cooldown_min=payload.cooldown_min,
        enabled=payload.enabled,
    )
    if rule is None:
        raise _fail(status.HTTP_404_NOT_FOUND, "No such alert rule")
    return AlertRuleModel(
        alert_id=rule.alert_id,
        name=rule.name,
        scope_type=rule.scope_type,
        scope_id=rule.scope_id,
        condition_type=rule.condition_type,
        threshold=rule.threshold,
        lookahead_days=rule.lookahead_days,
        cooldown_min=rule.cooldown_min,
        enabled=rule.enabled,
        last_triggered_at=rule.last_triggered_at,
    )


@router.get("/alert-events", response_model=tuple[AlertEventModel, ...])
async def list_alert_events(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    executor: Annotated[SqlExecutor, Depends(finops_executor)],
    event_status: Annotated[str | None, fastapi.Query(alias="status")] = None,
    scope_type: str | None = None,
    scope_id: str | None = None,
    limit: Annotated[int, fastapi.Query(ge=1, le=500)] = 100,
) -> tuple[AlertEventModel, ...]:
    require_permission(user_api_key_dict, "finops:view")
    scope: Final = resolve_scope(user_api_key_dict, scope_type=scope_type, scope_id=scope_id) if scope_type else None
    rows: Final = await AlertStore(executor).events(status=event_status, scope=scope, limit=limit)
    return tuple(
        AlertEventModel(
            id=row["id"],
            created_at=row["created_at"],
            alert_id=row["alert_id"],
            alert_type=row["alert_type"],
            severity=row["severity"],
            scope_type=row["scope_type"],
            scope_id=row["scope_id"],
            title=row["title"],
            body=row["body"],
            status=row["status"],
            acked_by=row["acked_by"],
        )
        for row in rows
    )


@router.patch("/alert-events/{event_id}", response_model=AlertEventModel)
async def update_alert_event(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    executor: Annotated[SqlExecutor, Depends(finops_executor)],
    event_id: str,
    payload: AlertEventUpdate,
) -> AlertEventModel:
    require_permission(user_api_key_dict, "finops:manage_alerts")
    row: Final = await AlertStore(executor).set_status(
        event_id, status=payload.status, acked_by=user_api_key_dict.user_id
    )
    if row is None:
        raise _fail(status.HTTP_404_NOT_FOUND, "No such alert event")
    return AlertEventModel(
        id=row["id"],
        created_at=row["created_at"],
        alert_id=row["alert_id"],
        alert_type=row["alert_type"],
        severity=row["severity"],
        scope_type=row["scope_type"],
        scope_id=row["scope_id"],
        title=row["title"],
        body=row["body"],
        status=row["status"],
        acked_by=row["acked_by"],
    )


@router.get("/report", response_class=PlainTextResponse)
async def get_report(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    executor: Annotated[SqlExecutor, Depends(finops_executor)],
    report_format: Annotated[Literal["csv"], fastapi.Query(alias="format")] = "csv",
    scope_type: str = "team",
    period: Literal["month"] = "month",
) -> PlainTextResponse:
    """The CFO export: one row per scope, fixed columns, both P50 and P90 month-end figures."""
    require_permission(user_api_key_dict, "finops:view")
    if scope_type not in visible_scope_types(user_api_key_dict):
        raise forbidden(f"Not authorized to export a {scope_type} report")
    store: Final = ForecastStore(executor)
    as_of: Final = _yesterday()
    body: Final = build_csv(
        await store.spend_forecasts(scope_types=(scope_type,), limit=_MAX_REPORT_ROWS),
        runway=await store.runway_by_types(scope_types=(scope_type,)),
        actuals=await FactReader(executor).spend_by_scope(
            start=as_of.replace(day=1), end=as_of, scope_types=(scope_type,)
        ),
        as_of=as_of,
    )
    return PlainTextResponse(
        content=body,
        media_type="text/csv",
        headers=_headers((("Content-Disposition", f'attachment; filename="witos-finops-{period}-{as_of}.csv"'),)),
    )


@router.get("/stream")
async def stream_events(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    executor: Annotated[SqlExecutor, Depends(finops_executor)],
    scope_type: str | None = None,
    scope_id: str | None = None,
) -> StreamingResponse:
    """SSE: alert events as they are raised, and a notice when a forecast run replaces the numbers."""
    require_permission(user_api_key_dict, "finops:view")
    scope: Final = resolve_scope(user_api_key_dict, scope_type=scope_type, scope_id=scope_id) if scope_type else None
    return StreamingResponse(
        FinOpsStream(executor).frames(scope=scope),
        media_type="text/event-stream",
        headers=_headers((("Cache-Control", "no-cache"), ("X-Accel-Buffering", "no"))),
    )


@router.post("/admin/rebuild", response_model=RebuildResponse)
async def rebuild(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    executor: Annotated[SqlExecutor, Depends(finops_executor)],
    payload: RebuildRequest,
) -> RebuildResponse:
    """Recompute forecasts now, for one scope or for the fleet."""
    require_permission(user_api_key_dict, "finops:manage")
    from litellm.proxy.witos.finops.jobs import ForecastJob, ForecastRunSkipped

    job: Final = ForecastJob(executor)
    if payload.scope_type and payload.scope_id:
        scope: Final = resolve_scope(user_api_key_dict, scope_type=payload.scope_type, scope_id=payload.scope_id)
        outcome: Final = await job.run_for(scope, as_of=payload.as_of)
        return RebuildResponse(
            scopes_seen=1,
            forecasts_written=outcome.forecasts,
            runways_written=outcome.runways,
            alerts_raised=outcome.alerts,
            states=MappingProxyType({outcome.state: 1}),
            stopped_early=False,
            duration_seconds=0.0,
        )
    run: Final = await job.run(as_of=payload.as_of)
    if isinstance(run, ForecastRunSkipped):
        raise _fail(status.HTTP_409_CONFLICT, f"A forecast run is already in progress on {run.held_by}")
    return RebuildResponse(
        scopes_seen=run.scopes_seen,
        forecasts_written=run.forecasts_written,
        runways_written=run.runways_written,
        alerts_raised=run.alerts_raised,
        states=run.states,
        stopped_early=run.stopped_early,
        duration_seconds=run.duration_seconds,
    )


@router.get("/admin/status", response_model=AdminStatusResponse)
async def admin_status(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    executor: Annotated[SqlExecutor, Depends(finops_executor)],
) -> AdminStatusResponse:
    """Aggregation lag, forecast freshness and open alerts: what an operator pages on."""
    require_permission(user_api_key_dict, "finops:manage")
    from litellm.proxy.witos.finops.config import HOURLY_WATERMARK_KEY, FinOpsConfig

    watermark: Final = await FinOpsConfig(executor).watermark(HOURLY_WATERMARK_KEY)
    generated_at, scopes = await ForecastStore(executor).freshness()
    return AdminStatusResponse(
        aggregation_watermark=watermark,
        aggregation_lag_seconds=(
            None if watermark is None else (datetime.now(timezone.utc) - watermark).total_seconds()
        ),
        latest_forecast_at=generated_at,
        forecast_scopes=scopes,
        open_alerts=await AlertStore(executor).open_event_count(),
        algorithm_version=ALGORITHM_VERSION,
    )
