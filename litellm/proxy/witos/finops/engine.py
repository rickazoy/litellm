"""The forecast engine: forecast usage, then price it (blueprint §1.3, §1.6).

The order is the whole design. Cost is not a time series with a life of its own,
it is four usage series multiplied by a price list that changes underneath them.
Forecasting the dollar figure directly folds "we sent 20% more requests" and "our
traffic moved onto a model that costs six times more" into one number that cannot
be taken apart again, and then a scenario about migrating traffic can only be
answered with a fudge factor. Forecasting requests, prompt tokens, completion
tokens and cached reads separately, and pricing the result through LiteLLM's own
calculator, makes the driver decomposition (§1.9) exact and every scenario (§1.10)
an exact repricing.

The engine refuses in three specific places, and each refusal is a feature:

Under three days of history it returns a state, not a number. A P90 computed from
two points is decoration.

When the price book cannot reproduce the spend that already happened, the scope
falls back to forecasting cost directly and says so, rather than publishing a
priced forecast that disagrees with the invoice.

When an entity has been idle for a month it is dormant and gets no forecast at
all, because "$0.03/day and falling" is not a useful thing to have computed for
two thousand dead keys.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from litellm.proxy.witos.finops.backtest import (
    EXPENSIVE_WAPE_GATE,
    MIN_BOOTSTRAP_RESIDUALS,
    Competition,
    Fold,
    ModelScore,
    in_sample_residuals,
    plan_folds,
    run_competition,
)
from litellm.proxy.witos.finops.forecast_models import ForecastModel, TrainingWindow, roster_for_tier
from litellm.proxy.witos.finops.history import (
    DORMANT_AFTER_DAYS,
    Confidence,
    HistoryTier,
    ScopeRef,
    UsageHistory,
    classify_history,
    tier_confidence,
)
from litellm.proxy.witos.finops.pricing import (
    BlendedPrice,
    PriceBook,
    PricingFidelity,
    build_price_book,
)
from litellm.proxy.witos.finops.regime import Regime, detect_regime, training_start
from litellm.proxy.witos.finops.simulation import (
    DEFAULT_PATH_COUNT,
    P10,
    P50,
    P90,
    PathEnsemble,
    simulate,
    stable_seed,
    weighted_sum,
)

# Bumped whenever the numbers this module produces would change for unchanged
# input. The backtest-regression baselines are pinned to it, and a forecast row
# records it so two forecasts computed by different versions are never compared
# as if they were the same measurement.
ALGORITHM_VERSION: Final = "2.0.0"

Strategy: TypeAlias = Literal["TOKEN_BASED", "ACTUAL_COST_BASED"]
ForecastSeriesName: TypeAlias = Literal[
    "request_count", "prompt_tokens", "completion_tokens", "cache_read_tokens", "spend_usd"
]

DEFAULT_HORIZON_DAYS: Final = 90
MIX_WINDOW_DAYS: Final = 28
FIDELITY_WINDOW_DAYS: Final = 28

# Above this share of the mix being unpriceable, a token-based forecast is mostly
# multiplying tokens by zero, so the scope forecasts its cost series instead.
MAX_UNKNOWN_MIX_SHARE: Final = 0.5

_BAND: Final = (P10, P50, P90)
_PRICED_SERIES: Final[tuple[ForecastSeriesName, ...]] = (
    "request_count",
    "prompt_tokens",
    "completion_tokens",
    "cache_read_tokens",
)


@dataclass(frozen=True, slots=True)
class SeriesForecast:
    """One forecast series: its winner, that winner's backtest, and its simulated paths."""

    name: str
    algorithm: str
    score: ModelScore | None
    point: tuple[float, ...]
    ensemble: PathEnsemble

    @property
    def wape(self) -> float | None:
        return self.score.wape if self.score else None

    @property
    def bias(self) -> float | None:
        return self.score.bias if self.score else None


@dataclass(frozen=True, slots=True)
class Band:
    """P10/P50/P90 for one day, or for one running total."""

    day: date
    p10: float
    p50: float
    p90: float


@dataclass(frozen=True, slots=True)
class ScopeForecast:
    """Everything one scope's nightly forecast produced, before it is persisted."""

    scope: ScopeRef
    generated_at: datetime
    forecast_start: date
    forecast_end: date
    strategy: Strategy
    tier: HistoryTier
    confidence: Confidence
    regime: Regime
    history_days: int
    training_points: int
    price_book: PriceBook
    mix_shares: Mapping[str, float]
    pricing_fidelity: PricingFidelity
    blended_price: BlendedPrice
    series: Mapping[str, SeriesForecast]
    spend: PathEnsemble
    total_tokens: PathEnsemble
    residuals: Mapping[str, tuple[float, ...]]
    seed: int
    path_count: int

    @property
    def horizon_days(self) -> int:
        return (self.forecast_end - self.forecast_start).days + 1

    def ensemble_for(self, metric: str) -> PathEnsemble | None:
        match metric:
            case "usd" | "spend_usd":
                return self.spend
            case "total_tokens":
                return self.total_tokens
            case "requests" | "request_count":
                return self.series["request_count"].ensemble if "request_count" in self.series else None
            case "prompt_tokens" | "completion_tokens" | "cache_read_tokens":
                found: Final = self.series.get(metric)
                return found.ensemble if found else None
            case _:
                return None

    def bands(self, ensemble: PathEnsemble, *, cumulative: bool) -> tuple[Band, ...]:
        rows: Final = ensemble.cumulative_quantiles(_BAND) if cumulative else ensemble.daily_quantiles(_BAND)
        low, mid, high = rows
        return tuple(
            Band(day=ensemble.day_at(offset), p10=low[offset], p50=mid[offset], p90=high[offset])
            for offset in range(ensemble.horizon)
        )


@dataclass(frozen=True, slots=True)
class NoForecast:
    """Why a scope produced no numbers. Rendered as a state, never as zeros."""

    scope: ScopeRef
    state: Literal["insufficient_history", "dormant"]
    history_days: int
    detail: str


ForecastOutcome: TypeAlias = ScopeForecast | NoForecast


def _mix_shares(history: UsageHistory, *, window_days: int) -> Mapping[str, float]:
    """Recent share of tokens per model group, held flat across the horizon.

    A drifting mix is forecast by the drift already present in the trailing
    window rather than extrapolated: mix share is bounded in [0, 1] and a linear
    extrapolation of it produces shares above one within a quarter.
    """
    totals: Final = history.mix_tokens_total(window_days=window_days)
    grand: Final = sum(totals.values())
    if grand <= 0:
        return MappingProxyType({})
    return MappingProxyType({model: tokens / grand for model, tokens in totals.items()})


def _reconstruct_spend(history: UsageHistory, book: PriceBook, *, window_days: int) -> PricingFidelity:
    """Reprice the recent past from its own facts, to see whether pricing explains the bill."""
    recent: Final = history.days[-window_days:] if window_days < len(history.days) else history.days
    reconstructed: Final = sum(
        (
            book.blend(day.mix_tokens).cost_of(
                prompt_tokens=Decimal(str(day.prompt_tokens)),
                completion_tokens=Decimal(str(day.completion_tokens)),
                cache_read_tokens=Decimal(str(day.cache_read_tokens)),
            )
            for day in recent
        ),
        Decimal(0),
    )
    return PricingFidelity(reconstructed=reconstructed, actual=sum((day.spend_usd for day in recent), Decimal(0)))


def choose_strategy(
    history: UsageHistory, book: PriceBook, mix: Mapping[str, float]
) -> tuple[Strategy, PricingFidelity]:
    """Token-based unless the price book fails to explain the spend already booked."""
    fidelity: Final = _reconstruct_spend(history, book, window_days=FIDELITY_WINDOW_DAYS)
    tokens: Final = sum(day.total_tokens for day in history.days)
    if tokens <= 0 or book.unknown_share(mix) > MAX_UNKNOWN_MIX_SHARE or not fidelity.is_trustworthy:
        return "ACTUAL_COST_BASED", fidelity
    return "TOKEN_BASED", fidelity


def _window_for(history: UsageHistory, values: Sequence[float], start: int) -> TrainingWindow:
    return TrainingWindow(start_day=history.start_day + timedelta(days=start), values=tuple(values[start:]))


def _series_values(history: UsageHistory, name: ForecastSeriesName) -> tuple[float, ...]:
    if name == "spend_usd":
        return history.spend_series()
    return history.usage_series(name)


def _competitions(
    history: UsageHistory,
    names: Sequence[ForecastSeriesName],
    *,
    tier: HistoryTier,
    start: int,
    expensive_gate: float | None,
) -> Mapping[str, tuple[TrainingWindow, Competition]]:
    roster: Final = roster_for_tier(tier)
    folds: Final = plan_folds(history.history_days - start)
    return MappingProxyType(
        {
            name: _compete(history, name, roster=roster, start=start, folds=folds, expensive_gate=expensive_gate)
            for name in names
        }
    )


def _compete(
    history: UsageHistory,
    name: ForecastSeriesName,
    *,
    roster: Sequence[ForecastModel],
    start: int,
    folds: Sequence[Fold],
    expensive_gate: float | None,
) -> tuple[TrainingWindow, Competition]:
    window: Final = _window_for(history, _series_values(history, name), start)
    return window, run_competition(window, roster, folds=folds, expensive_gate=expensive_gate)


def _residuals(
    competitions: Mapping[str, tuple[TrainingWindow, Competition]],
) -> Mapping[str, tuple[float, ...]]:
    """Backtest residuals when there are enough of them, in-sample errors when there are not.

    The choice is made once for the whole scope rather than per series, because
    the bootstrap draws one index per simulated day and applies it to every
    series: mixing a backtest residual for tokens with an in-sample residual for
    requests would silently pair errors measured on different days.
    """
    pooled: Final = tuple(
        len(competition.winning_score.residuals) if competition.winning_score else 0
        for _, competition in competitions.values()
    )
    if pooled and min(pooled) >= MIN_BOOTSTRAP_RESIDUALS:
        return MappingProxyType(
            {
                name: competition.winning_score.residuals if competition.winning_score else ()
                for name, (_, competition) in competitions.items()
            }
        )
    start: Final = max((competition.winner.min_points for _, competition in competitions.values()), default=0)
    return MappingProxyType(
        {
            name: in_sample_residuals(competition.winner, window, start=start)
            for name, (window, competition) in competitions.items()
        }
    )


def month_end(day: date) -> date:
    first_next: Final = (day.replace(day=28) + timedelta(days=4)).replace(day=1)
    return first_next - timedelta(days=1)


def quarter_end(day: date) -> date:
    final_month: Final = ((day.month - 1) // 3) * 3 + 3
    return month_end(day.replace(month=final_month, day=1))


def build_forecast(
    history: UsageHistory,
    *,
    as_of: date,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    generated_at: datetime | None = None,
    path_count: int = DEFAULT_PATH_COUNT,
    seed_salt: str = "",
    expensive_gate: float | None = EXPENSIVE_WAPE_GATE,
) -> ForecastOutcome:
    """Produce one scope's forecast, or the state explaining why there is not one."""
    tier: Final = classify_history(history.history_days)
    if tier == "insufficient_history":
        return NoForecast(
            scope=history.scope,
            state="insufficient_history",
            history_days=history.history_days,
            detail=f"{history.history_days} day(s) of history; a forecast needs at least 3",
        )
    if history.is_dormant():
        return NoForecast(
            scope=history.scope,
            state="dormant",
            history_days=history.history_days,
            detail=f"No activity in the last {DORMANT_AFTER_DAYS} days",
        )

    mix: Final = _mix_shares(history, window_days=MIX_WINDOW_DAYS)
    book: Final = build_price_book(tuple(mix))
    strategy, fidelity = choose_strategy(history, book, mix)
    names: Final[tuple[ForecastSeriesName, ...]] = _PRICED_SERIES if strategy == "TOKEN_BASED" else ("spend_usd",)

    assessment: Final = detect_regime(
        history.usage_series("prompt_tokens") if strategy == "TOKEN_BASED" else history.spend_series()
    )
    start: Final = training_start(
        assessment, length=history.history_days, min_points=max(MIN_BOOTSTRAP_RESIDUALS * 2, 14)
    )
    competitions: Final = _competitions(history, names, tier=tier, start=start, expensive_gate=expensive_gate)
    points: Final = MappingProxyType(
        {name: competition.winner.predict(window, horizon_days) for name, (window, competition) in competitions.items()}
    )
    seed: Final = stable_seed(str(history.scope), strategy, book.snapshot_hash, as_of.isoformat(), seed_salt)
    forecast_start: Final = as_of + timedelta(days=1)
    residuals: Final = _residuals(competitions)
    ensembles: Final = simulate(
        points,
        residuals,
        start_day=forecast_start,
        seed=seed,
        path_count=path_count,
    )
    series: Final = MappingProxyType(
        {
            name: SeriesForecast(
                name=name,
                algorithm=competition.winner.name,
                score=competition.winning_score,
                point=points[name],
                ensemble=ensembles[name],
            )
            for name, (_, competition) in competitions.items()
        }
    )
    blended: Final = book.blend(mix).scaled(fidelity.calibration())
    return ScopeForecast(
        scope=history.scope,
        generated_at=generated_at or datetime.now(timezone.utc),
        forecast_start=forecast_start,
        forecast_end=forecast_start + timedelta(days=horizon_days - 1),
        strategy=strategy,
        tier=tier,
        confidence=_confidence_for(tier, ensembles),
        regime=assessment.regime,
        history_days=history.history_days,
        training_points=history.history_days - start,
        price_book=book,
        mix_shares=mix,
        pricing_fidelity=fidelity,
        blended_price=blended,
        series=series,
        spend=_spend_paths(series, blended, horizon_days),
        total_tokens=_token_paths(series, horizon_days),
        residuals=residuals,
        seed=seed,
        path_count=path_count,
    )


def _confidence_for(tier: HistoryTier, ensembles: Mapping[str, PathEnsemble]) -> Confidence:
    """The ladder's confidence, lowered when there were no residuals to draw a band from."""
    banded: Final = all(ensemble.has_band for ensemble in ensembles.values())
    tier_level: Final = tier_confidence(tier)
    return tier_level if banded else "low"


def _spend_paths(series: Mapping[str, SeriesForecast], blended: BlendedPrice, horizon_days: int) -> PathEnsemble:
    """Cost paths, priced inside each simulated path rather than off the quantiles."""
    if "spend_usd" in series:
        return series["spend_usd"].ensemble
    return weighted_sum(
        (
            series["prompt_tokens"].ensemble,
            series["completion_tokens"].ensemble,
            series["cache_read_tokens"].ensemble,
        ),
        (
            (float(blended.input_per_token),) * horizon_days,
            (float(blended.output_per_token),) * horizon_days,
            (float(blended.cache_read_per_token),) * horizon_days,
        ),
    )


def _token_paths(series: Mapping[str, SeriesForecast], horizon_days: int) -> PathEnsemble:
    if "prompt_tokens" not in series:
        return series["spend_usd"].ensemble
    return weighted_sum(
        (series["prompt_tokens"].ensemble, series["completion_tokens"].ensemble),
        ((1.0,) * horizon_days, (1.0,) * horizon_days),
    )


def period_projection(forecast: ScopeForecast, actual_to_date: Decimal, period_end_day: date) -> Decimal:
    """Actual so far plus the P50 of the rest of the period.

    Deliberately not the P50 of (actual + rest): the part that already happened
    has no distribution.
    """
    offset: Final = forecast.spend.offset_of(period_end_day)
    if offset < 0:
        return actual_to_date
    return actual_to_date + Decimal(str(forecast.spend.cumulative_at(offset, P50)))


def daily_burn_rate(forecast: ScopeForecast) -> Decimal:
    """Median spend per day over the first forecast week, the number a burn card shows."""
    band: Final = forecast.spend.daily_quantiles((P50,))[0]
    if not band:
        return Decimal(0)
    days: Final = min(7, len(band))
    return Decimal(str(sum(band[:days]) / days))
