"""The nightly forecast run, and the two smaller jobs beside it (blueprint §1.6-§1.8).

One scope at a time: load its history, build the forecast, price it, walk the
Monte Carlo paths against its quotas, decompose what moved, write the rows, and
evaluate the alert rules against numbers that are seconds old. Everything the API
serves is produced here, which is what keeps the API's p95 a read.

The run is bounded in wall-clock time rather than hoping it fits. §1.14 budgets
five minutes for two thousand scopes, and scopes are processed in descending
order of recent spend, so a fleet that grows past the budget degrades by giving
the tail cheaper forecasts (and, past the hard deadline, none this cycle) instead
of overrunning into the morning. Two knobs express that: the maximum-likelihood
candidates stop being offered once most of the budget is spent, and the run stops
entirely at the deadline. Both are reported in the summary rather than being
silent.
"""

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from litellm._logging import verbose_proxy_logger
from litellm.proxy.witos.finops.alerts import AlertRule, AlertStore, ScopeSignals, evaluate_rules
from litellm.proxy.witos.finops.backtest import EXPENSIVE_WAPE_GATE
from litellm.proxy.witos.finops.drivers import DriverDecomposition, decompose
from litellm.proxy.witos.finops.engine import (
    DEFAULT_HORIZON_DAYS,
    NoForecast,
    ScopeForecast,
    build_forecast,
    month_end,
    period_projection,
)
from litellm.proxy.witos.finops.facts import DEFAULT_FORECAST_SCOPE_TYPES, FactReader
from litellm.proxy.witos.finops.forecast_metrics import ForecastMetrics
from litellm.proxy.witos.finops.forecast_store import ForecastStore
from litellm.proxy.witos.finops.history import ScopeRef, UsageHistory
from litellm.proxy.witos.finops.quality import QualityScorer, quality_score
from litellm.proxy.witos.finops.quota_store import QuotaStore
from litellm.proxy.witos.finops.runway import Quota, RunwayResult, compute_runway
from litellm.proxy.witos.finops.simulation import DEFAULT_PATH_COUNT
from litellm.proxy.witos.shared.scheduler import JobLeaseManager, LeaseHeld
from litellm.proxy.witos.shared.sql import SqlExecutor

FORECAST_JOB_NAME: Final = "witos_finops_forecast"
QUALITY_JOB_NAME: Final = "witos_finops_quality"
ANOMALY_JOB_NAME: Final = "witos_finops_anomaly"

DEFAULT_TIME_BUDGET: Final = timedelta(minutes=5)

# Past this share of the budget the run stops offering the expensive candidates,
# so the scopes at the end of the queue still get a forecast rather than a
# timeout. They are the smallest spenders in the fleet, by construction.
_CHEAP_ROSTER_AFTER: Final = 0.6

DRIVER_WINDOW_DAYS: Final = 7

ScopeState: TypeAlias = Literal["forecast", "insufficient_history", "dormant", "no_history"]


@dataclass(frozen=True, slots=True)
class ForecastRunSummary:
    scopes_seen: int
    forecasts_written: int
    runways_written: int
    alerts_raised: int
    states: Mapping[str, int]
    stopped_early: bool
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class ForecastRunSkipped:
    held_by: str


ForecastRunOutcome: TypeAlias = ForecastRunSummary | ForecastRunSkipped


@dataclass(frozen=True, slots=True)
class ScopeOutcome:
    state: ScopeState
    forecasts: int
    runways: int
    alerts: int


class _Deadline:
    """A wall-clock budget the run consults rather than a count of scopes it hopes fits."""

    def __init__(self, budget: timedelta) -> None:
        self._started: Final = time.monotonic()
        self._budget: Final = budget.total_seconds()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._started

    def expired(self) -> bool:
        return self.elapsed >= self._budget

    def expensive_gate(self) -> float | None:
        return EXPENSIVE_WAPE_GATE if self.elapsed < self._budget * _CHEAP_ROSTER_AFTER else None


def _period_start(quota: Quota, *, as_of: date) -> date:
    start: Final = quota.period_start.date()
    return min(start, as_of)


def _usage_growth_pct(history: UsageHistory, *, window: int) -> float | None:
    recent: Final = sum(day.total_tokens for day in history.days[-window:])
    baseline: Final = sum(day.total_tokens for day in history.days[-2 * window : -window])
    if baseline <= 0:
        return None
    return (recent - baseline) / baseline * 100


class ForecastJob:
    """Builds, prices, persists and alerts on every active scope's forecast."""

    def __init__(
        self,
        executor: SqlExecutor,
        *,
        facts: FactReader | None = None,
        store: ForecastStore | None = None,
        quotas: QuotaStore | None = None,
        alerts: AlertStore | None = None,
        lease_manager: JobLeaseManager | None = None,
        scope_types: Sequence[str] = DEFAULT_FORECAST_SCOPE_TYPES,
        horizon_days: int = DEFAULT_HORIZON_DAYS,
        path_count: int = DEFAULT_PATH_COUNT,
        time_budget: timedelta = DEFAULT_TIME_BUDGET,
    ) -> None:
        self._facts: Final = facts or FactReader(executor)
        self._store: Final = store or ForecastStore(executor)
        self._quotas: Final = quotas or QuotaStore(executor)
        self._alerts: Final = alerts or AlertStore(executor)
        self._lease_manager: Final = lease_manager or JobLeaseManager(executor)
        self._scope_types: Final = tuple(scope_types)
        self._horizon_days: Final = horizon_days
        self._path_count: Final = path_count
        self._time_budget: Final = time_budget

    async def run(self, *, as_of: date | None = None, now: datetime | None = None) -> ForecastRunOutcome:
        """One pass over every active scope, under the job lease."""
        async with self._lease_manager.hold(FORECAST_JOB_NAME, ttl=self._time_budget * 2) as lease:
            if isinstance(lease, LeaseHeld):
                ForecastMetrics.record_scope("skipped_locked")
                return ForecastRunSkipped(held_by=lease.held_by)
            moment: Final = now or datetime.now(timezone.utc)
            target: Final = as_of or (moment.date() - timedelta(days=1))
            deadline: Final = _Deadline(self._time_budget)
            rules: Final = await self._alerts.rules()
            scopes: Final = await self._facts.active_scopes(
                since=target - timedelta(days=DRIVER_WINDOW_DAYS * 2), scope_types=self._scope_types
            )
            outcomes: Final = tuple(
                [
                    await self._process(entry.scope, as_of=target, now=moment, rules=rules, deadline=deadline)
                    for entry in scopes
                    if not deadline.expired()
                ]
            )
            summary: Final = _summarise(
                outcomes, scopes_seen=len(scopes), deadline=deadline, stopped_early=deadline.expired()
            )
            ForecastMetrics.record_run(summary.duration_seconds)
            verbose_proxy_logger.info(
                "WIT OS FinOps forecast run: %d scopes, %d forecasts, %d runway rows, %d alerts in %.1fs",
                summary.scopes_seen,
                summary.forecasts_written,
                summary.runways_written,
                summary.alerts_raised,
                summary.duration_seconds,
            )
            return summary

    async def run_for(self, scope: ScopeRef, *, as_of: date | None = None, now: datetime | None = None) -> ScopeOutcome:
        """Rebuild one scope on demand, for ``POST /witos/finops/admin/rebuild``."""
        moment: Final = now or datetime.now(timezone.utc)
        return await self._process(
            scope,
            as_of=as_of or (moment.date() - timedelta(days=1)),
            now=moment,
            rules=await self._alerts.rules(),
            deadline=_Deadline(self._time_budget),
        )

    async def _process(
        self,
        scope: ScopeRef,
        *,
        as_of: date,
        now: datetime,
        rules: Sequence[AlertRule],
        deadline: _Deadline,
    ) -> ScopeOutcome:
        history: Final = await self._facts.history(scope, as_of=as_of)
        if history is None:
            ForecastMetrics.record_scope("no_history")
            return ScopeOutcome(state="no_history", forecasts=0, runways=0, alerts=0)

        outcome: Final = build_forecast(
            history,
            as_of=as_of,
            horizon_days=self._horizon_days,
            generated_at=now,
            path_count=self._path_count,
            expensive_gate=deadline.expensive_gate(),
        )
        if isinstance(outcome, NoForecast):
            await self._store.supersede(scope)
            ForecastMetrics.record_scope(outcome.state)
            return ScopeOutcome(state=outcome.state, forecasts=0, runways=0, alerts=0)

        previous: Final = await self._store.latest(scope, "spend_usd")
        quotas: Final = await self._quotas.for_scope(scope)
        runway: Final = await self._runway_for(outcome, quotas, as_of=as_of)
        drivers: Final = _drivers_for(history, outcome)
        month_to_date: Final = await self._facts.totals(scope, start=as_of.replace(day=1), end=as_of)
        counts: Final = await self._store.save(
            outcome, actual_to_date=month_to_date.spend_usd, runway=runway, drivers=drivers
        )
        ForecastMetrics.record_scope("forecast")
        _record_accuracy(outcome)
        alerts: Final = await self._raise_alerts(
            outcome,
            history,
            rules=rules,
            runway=runway,
            drivers=drivers,
            previous_eom=previous["proj_eom"] if previous else None,
            month_to_date=month_to_date.spend_usd,
            as_of=as_of,
            now=now,
        )
        return ScopeOutcome(state="forecast", forecasts=counts.forecasts, runways=counts.runways, alerts=alerts)

    async def _runway_for(
        self, forecast: ScopeForecast, quotas: Sequence[Quota], *, as_of: date
    ) -> tuple[RunwayResult, ...]:
        consumed: Final = tuple(
            [
                (
                    quota,
                    (await self._facts.totals(forecast.scope, start=_period_start(quota, as_of=as_of), end=as_of)).of(
                        quota.metric
                    ),
                )
                for quota in quotas
            ]
        )
        return tuple(
            result
            for result in (
                compute_runway(forecast, quota, used, today=forecast.forecast_start) for quota, used in consumed
            )
            if result is not None
        )

    async def _raise_alerts(
        self,
        forecast: ScopeForecast,
        history: UsageHistory,
        *,
        rules: Sequence[AlertRule],
        runway: Sequence[RunwayResult],
        drivers: DriverDecomposition | None,
        previous_eom: Decimal | None,
        month_to_date: Decimal,
        as_of: date,
        now: datetime,
    ) -> int:
        projected: Final = _projected_eom(forecast, month_to_date)
        signals: Final = ScopeSignals(
            scope=forecast.scope,
            projected_eom=projected,
            previous_projected_eom=previous_eom,
            quality_score=quality_score(
                wape=_headline_wape(forecast), history_days=forecast.history_days, regime=forecast.regime
            ),
            runway=runway,
            drivers=drivers,
            usage_growth_pct=_usage_growth_pct(history, window=DRIVER_WINDOW_DAYS),
        )
        drafts: Final = evaluate_rules(rules, signals, today=as_of, now=now)
        for draft in drafts:
            ForecastMetrics.record_alert(draft.alert_type)
        raised: Final = tuple(
            [await self._alerts.raise_event(draft, _channels_for(rules, draft.alert_id)) for draft in drafts]
        )
        return len(raised)


def _channels_for(rules: Sequence[AlertRule], alert_id: str | None) -> Sequence[Mapping[str, str]]:
    if alert_id is None:
        return ()
    matched: Final = tuple(rule for rule in rules if rule.alert_id == alert_id)
    return matched[0].channels if matched else ()


def _projected_eom(forecast: ScopeForecast, month_to_date: Decimal) -> Decimal:
    return period_projection(forecast, month_to_date, month_end(forecast.forecast_start))


def _headline_wape(forecast: ScopeForecast) -> float | None:
    series: Final = forecast.series.get("prompt_tokens") or next(iter(forecast.series.values()))
    return series.wape


def _record_accuracy(forecast: ScopeForecast) -> None:
    series: Final = forecast.series.get("prompt_tokens") or next(iter(forecast.series.values()))
    ForecastMetrics.record_accuracy(forecast.scope.scope_type, wape=series.wape, bias=series.bias)


def _drivers_for(history: UsageHistory, forecast: ScopeForecast) -> DriverDecomposition | None:
    if history.history_days < DRIVER_WINDOW_DAYS * 2:
        return None
    return decompose(
        history.days[-DRIVER_WINDOW_DAYS * 2 : -DRIVER_WINDOW_DAYS],
        history.days[-DRIVER_WINDOW_DAYS:],
        forecast.price_book,
    )


def _summarise(
    outcomes: Sequence[ScopeOutcome], *, scopes_seen: int, deadline: _Deadline, stopped_early: bool
) -> ForecastRunSummary:
    states: Final = MappingProxyType(
        {
            state: sum(1 for outcome in outcomes if outcome.state == state)
            for state in ("forecast", "insufficient_history", "dormant", "no_history")
        }
    )
    return ForecastRunSummary(
        scopes_seen=scopes_seen,
        forecasts_written=sum(outcome.forecasts for outcome in outcomes),
        runways_written=sum(outcome.runways for outcome in outcomes),
        alerts_raised=sum(outcome.alerts for outcome in outcomes),
        states=states,
        stopped_early=stopped_early,
        duration_seconds=deadline.elapsed,
    )


class QualityJob:
    """Scores yesterday's one-day-ahead predictions and publishes the rolling result."""

    def __init__(
        self,
        executor: SqlExecutor,
        *,
        scorer: QualityScorer | None = None,
        lease_manager: JobLeaseManager | None = None,
    ) -> None:
        self._scorer: Final = scorer or QualityScorer(executor)
        self._lease_manager: Final = lease_manager or JobLeaseManager(executor)

    async def run(self, *, as_of: date | None = None) -> int:
        async with self._lease_manager.hold(QUALITY_JOB_NAME) as lease:
            if isinstance(lease, LeaseHeld):
                return 0
            target: Final = as_of or (datetime.now(timezone.utc).date() - timedelta(days=1))
            return await self._scorer.publish(await self._scorer.measure(as_of=target))
