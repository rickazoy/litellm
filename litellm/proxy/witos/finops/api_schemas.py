"""Response and request models for ``/witos/finops/*`` (blueprint §1.11).

These are the OpenAPI contract, and WIT OS generates its TypeScript client from
them (§1.13), so a field renamed here is a breaking change for a second product.
They are declared rather than derived for that reason: a response shaped by
whatever the row happened to contain is not a contract.

``forecast_json`` is parsed into ``ForecastPayload`` rather than passed through
as loose JSON. The writer and the reader are the same codebase, so there is no
excuse for the API's own document to be untyped on the way out.
"""

from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from litellm.proxy.witos.finops.scenarios import ScenarioAssumptions

ForecastState: TypeAlias = Literal["forecast", "insufficient_history", "dormant", "no_forecast"]


class BandPoint(BaseModel):
    """One day of the fan chart. ``p10``/``p90`` are the Monte Carlo band, never a fixed ratio."""

    model_config = ConfigDict(frozen=True)

    date: date
    p10: float
    p50: float
    p90: float


class DriverLine(BaseModel):
    model_config = ConfigDict(frozen=True)

    factor: str
    label: str
    delta_usd: float
    delta_pct: float


class BacktestSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str | None = None
    wape: float | None = None
    bias: float | None = None
    mae: float | None = None
    folds: int = 0


class ComponentSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    algorithm: str
    wape: float | None = None
    bias: float | None = None


class ForecastPayload(BaseModel):
    """The stored ``forecast_json`` document."""

    model_config = ConfigDict(frozen=True, extra="ignore", protected_namespaces=())

    days: tuple[BandPoint, ...] = ()
    cumulative: tuple[BandPoint, ...] = ()
    tier: str = "insufficient_history"
    confidence: str = "none"
    strategy: str = "TOKEN_BASED"
    regime: str = "insufficient_history"
    backtest: BacktestSummary = BacktestSummary()
    components: Mapping[str, ComponentSummary] = Field(default_factory=dict)
    model_mix: Mapping[str, float] = Field(default_factory=dict)
    unknown_models: tuple[str, ...] = ()
    unknown_model_share: float = 0.0
    pricing_fidelity: float | None = None
    pricing_calibration: float = 1.0
    path_count: int = 0
    residual_count: int = 0
    has_band: bool = False
    seed: int = 0
    drivers: tuple[DriverLine, ...] = ()


class ForecastResponse(BaseModel):
    """``GET /witos/finops/forecast``. ``state`` is populated instead of the bands when §1.6 refuses."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    scope_type: str
    scope_id: str
    metric: str
    state: ForecastState
    detail: str | None = None
    generated_at: datetime | None = None
    forecast_start: date | None = None
    forecast_end: date | None = None
    strategy: str | None = None
    algorithm: str | None = None
    algorithm_version: str | None = None
    trend_regime: str | None = None
    history_days: int | None = None
    training_points: int | None = None
    wape: float | None = None
    bias: float | None = None
    quality_score: int | None = None
    pricing_snapshot_hash: str | None = None
    proj_eom: Decimal | None = None
    proj_eoq: Decimal | None = None
    burn_rate_daily: Decimal | None = None
    payload: ForecastPayload | None = None


class RunwayItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    quota_id: str
    metric: str
    period: str
    limit_value: Decimal
    consumed: Decimal
    remaining: Decimal
    consumed_pct: float
    exhaustion_p50: date | None
    exhaustion_p90: date | None
    prob_exhaust_before_reset: float | None
    prob_overrun_this_period: float | None
    survives_cycle: bool
    reset_at: datetime | None
    source_type: str
    statement: str


class RunwayResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    scope_type: str
    scope_id: str
    computed_at: datetime | None = None
    quotas: tuple[RunwayItem, ...] = ()


class OverviewCard(BaseModel):
    model_config = ConfigDict(frozen=True)

    scope_type: str
    scope_id: str
    proj_eom: Decimal | None
    burn_rate_daily: Decimal | None
    trend_regime: str
    quality_score: int | None


class OverviewResponse(BaseModel):
    """``GET /witos/finops/overview``: the fleet headline §1.11 asks for."""

    model_config = ConfigDict(frozen=True)

    generated_at: datetime | None
    mtd_spend: Decimal
    proj_eom_p50: Decimal
    proj_eom_p90: Decimal
    top_scopes: tuple[OverviewCard, ...] = ()
    runway_risks: tuple[RunwayItem, ...] = ()
    open_alerts: int = 0


class SeriesPointModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    bucket: datetime
    request_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cache_read_tokens: int
    spend_usd: Decimal


class TimeseriesResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    scope_type: str
    scope_id: str
    grain: Literal["day", "hour"]
    points: tuple[SeriesPointModel, ...]


class DriversResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    scope_type: str
    scope_id: str
    window_days: int
    baseline_daily_spend: Decimal
    current_daily_spend: Decimal
    delta_usd: Decimal
    delta_pct: float
    contributions: tuple[DriverLine, ...]


class ModelMixEntry(BaseModel):
    model_config = ConfigDict(frozen=True, protected_namespaces=())

    model_group: str
    token_share: float
    priced: bool


class ModelMixResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    scope_type: str
    scope_id: str
    forecast_shares: tuple[ModelMixEntry, ...]
    unknown_model_share: float


class QualityPoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    date: date
    predicted: Decimal
    actual: Decimal


class ForecastQualityResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    scope_type: str
    scope_id: str
    metric: str
    rolling_wape: float | None
    rolling_bias: float | None
    sample_days: int
    trend_regime: str | None
    quality_score: int | None
    points: tuple[QualityPoint, ...]


class QuotaModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    quota_id: str
    scope_type: str
    scope_id: str
    metric: str
    limit_value: Decimal
    period: str
    period_start: datetime
    period_end: datetime | None
    reset_at: datetime | None
    source_type: str
    soft_threshold_pct: Decimal | None
    read_only: bool


class QuotaCreate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    scope_type: str
    scope_id: str
    metric: Literal["usd", "total_tokens", "prompt_tokens", "completion_tokens", "requests"]
    limit_value: Annotated[Decimal, Field(gt=0)]
    period: Literal["day", "week", "month", "quarter", "year", "contract", "custom"]
    period_start: datetime
    period_end: datetime | None = None
    reset_at: datetime | None = None
    soft_threshold_pct: Annotated[Decimal | None, Field(default=None, ge=0, le=100)]


class QuotaUpdate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    limit_value: Annotated[Decimal | None, Field(default=None, gt=0)]
    period: str | None = None
    period_end: datetime | None = None
    reset_at: datetime | None = None
    soft_threshold_pct: Annotated[Decimal | None, Field(default=None, ge=0, le=100)]


class ScenarioModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    scenario_id: str
    name: str
    description: str | None
    scope_type: str
    scope_id: str
    assumptions: ScenarioAssumptions
    created_by: str
    created_at: datetime


class ScenarioCreate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str | None = None
    scope_type: str
    scope_id: str
    assumptions: ScenarioAssumptions


class ScenarioEvaluateRequest(BaseModel):
    """Stateless what-if. ``scenario_id`` loads a saved assumption set instead."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scope_type: str
    scope_id: str
    assumptions: ScenarioAssumptions | None = None
    scenario_id: str | None = None


class ScenarioSide(BaseModel):
    model_config = ConfigDict(frozen=True)

    horizon_p50: Decimal
    horizon_p90: Decimal
    projected_eom: Decimal
    projected_eoq: Decimal
    days: tuple[BandPoint, ...]


class ScenarioEvaluateResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    scope_type: str
    scope_id: str
    pricing_snapshot_hash: str
    seed: int
    baseline: ScenarioSide
    scenario: ScenarioSide
    delta_usd: Decimal
    delta_pct: float
    savings_usd: Decimal
    runway: tuple[RunwayItem, ...] = ()


class AlertRuleModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    alert_id: str
    name: str
    scope_type: str
    scope_id: str
    condition_type: str
    threshold: Decimal
    lookahead_days: int | None
    cooldown_min: int
    enabled: bool
    last_triggered_at: datetime | None


class AlertRuleCreate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    scope_type: str
    scope_id: str = "*"
    condition_type: Literal[
        "forecast_exceeds_budget",
        "budget_exhaustion_within",
        "token_exhaustion_within",
        "spend_growth_above",
        "usage_growth_above",
        "forecast_changed_above",
        "anomaly_detected",
        "forecast_quality_below",
    ]
    threshold: Decimal
    lookahead_days: int | None = None
    channels: tuple[Mapping[str, str], ...] = ()
    cooldown_min: int = 240
    enabled: bool = True


class AlertRuleUpdate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    threshold: Decimal | None = None
    lookahead_days: int | None = None
    cooldown_min: int | None = None
    enabled: bool | None = None


class AlertEventModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    created_at: datetime
    alert_id: str | None
    alert_type: str
    severity: str
    scope_type: str
    scope_id: str
    title: str
    body: Mapping[str, object]
    status: str
    acked_by: str | None


class AlertEventUpdate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["open", "acked", "resolved", "muted"]


class RebuildRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    scope_type: str | None = None
    scope_id: str | None = None
    as_of: date | None = None


class RebuildResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    scopes_seen: int
    forecasts_written: int
    runways_written: int
    alerts_raised: int
    states: Mapping[str, int]
    stopped_early: bool
    duration_seconds: float


class AdminStatusResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    aggregation_watermark: datetime | None
    aggregation_lag_seconds: float | None
    latest_forecast_at: datetime | None
    forecast_scopes: int
    open_alerts: int
    algorithm_version: str
