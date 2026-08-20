"""Alert rules, events and dispatch (blueprint §1.4, §1.11).

Rule evaluation is a pure function of one scope's freshly computed forecast,
runway and drivers. That is deliberate: alerting is the part of a FinOps system
that wakes people up, so it has to be testable without a database, a clock, or a
webhook receiver, and it has to be impossible for an alert to say something the
forecast does not.

Every event body carries the numbers the decision was made from, not a sentence
about them. A receiver that wants prose can render it; a receiver that wants to
route on "probability above 0.9" can read the field.

Cooldown is enforced against ``last_triggered_at`` on the rule before anything is
written or sent, because the failure mode of a budget alert is not silence, it is
two hundred identical messages the night a workload steps up.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Final, Literal, TypeAlias, TypedDict

from pydantic import TypeAdapter
from typing_extensions import ReadOnly

from litellm._logging import verbose_proxy_logger
from litellm.proxy.witos.finops.drivers import DriverDecomposition
from litellm.proxy.witos.finops.history import ScopeRef
from litellm.proxy.witos.finops.runway import RunwayResult, runway_statement
from litellm.proxy.witos.shared.sql import SqlExecutor
from litellm.proxy.witos.shared.webhooks import (
    Delivered,
    HttpPoster,
    HttpxPoster,
    WebhookPayload,
    deliver,
)

ConditionType: TypeAlias = Literal[
    "forecast_exceeds_budget",
    "budget_exhaustion_within",
    "token_exhaustion_within",
    "spend_growth_above",
    "usage_growth_above",
    "forecast_changed_above",
    "anomaly_detected",
    "forecast_quality_below",
]
Severity: TypeAlias = Literal["info", "warning", "critical"]

WILDCARD: Final = "*"
DELIVERABLE_CHANNELS: Final = frozenset({"webhook", "slack"})

_TOKEN_METRICS: Final = frozenset({"total_tokens", "prompt_tokens", "completion_tokens", "requests"})
_MONEY_METRICS: Final = frozenset({"usd", "spend_usd"})


@dataclass(frozen=True, slots=True)
class AlertRule:
    alert_id: str
    name: str
    scope_type: str
    scope_id: str
    condition_type: ConditionType
    threshold: Decimal
    lookahead_days: int | None
    channels: Sequence[Mapping[str, str]]
    cooldown_min: int
    enabled: bool
    last_triggered_at: datetime | None

    def covers(self, scope: ScopeRef) -> bool:
        if self.scope_type not in (scope.scope_type, WILDCARD):
            return False
        return self.scope_id in (scope.scope_id, WILDCARD)

    def is_cool(self, now: datetime) -> bool:
        if self.last_triggered_at is None:
            return True
        return now - self.last_triggered_at >= timedelta(minutes=self.cooldown_min)


class DriverShare(TypedDict):
    factor: ReadOnly[str]
    delta_pct: ReadOnly[float]


class BudgetBody(TypedDict):
    projected_eom: ReadOnly[str]
    limit_value: ReadOnly[str]
    threshold_pct: ReadOnly[float]
    probability: ReadOnly[float]
    statement: ReadOnly[str]


class ExhaustionBody(TypedDict):
    quota_id: ReadOnly[str]
    metric: ReadOnly[str]
    exhaustion_p50: ReadOnly[str]
    exhaustion_p90: ReadOnly[str]
    probability: ReadOnly[float]
    remaining: ReadOnly[str]
    statement: ReadOnly[str]


class GrowthBody(TypedDict):
    observed_pct: ReadOnly[float]
    threshold_pct: ReadOnly[float]
    drivers: ReadOnly[tuple[DriverShare, ...]]


class ForecastChangeBody(TypedDict):
    previous_projected_eom: ReadOnly[str]
    projected_eom: ReadOnly[str]
    change_pct: ReadOnly[float]


class QualityBody(TypedDict):
    quality_score: ReadOnly[int]
    threshold: ReadOnly[float]


class AnomalyBody(TypedDict):
    observed_spend_usd: ReadOnly[str]
    observed_requests: ReadOnly[int]
    z: ReadOnly[float]
    consecutive_buckets: ReadOnly[int]
    bucket_start_utc: ReadOnly[str]
    top_model: ReadOnly[str | None]


class NewModelBody(TypedDict):
    model: ReadOnly[str]
    hourly_spend_usd: ReadOnly[str]
    bucket_start_utc: ReadOnly[str]


@dataclass(frozen=True, slots=True)
class AlertDraft:
    """An event that has not been written yet, so tests can assert on it directly."""

    alert_id: str | None
    alert_type: ConditionType | Literal["anomaly_detected", "new_model_cost"]
    severity: Severity
    scope: ScopeRef
    title: str
    body: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ScopeSignals:
    """Everything a rule is allowed to look at, gathered once per scope."""

    scope: ScopeRef
    projected_eom: Decimal
    previous_projected_eom: Decimal | None
    quality_score: int | None
    runway: Sequence[RunwayResult]
    drivers: DriverDecomposition | None
    usage_growth_pct: float | None


def _budget_limit(signals: ScopeSignals) -> RunwayResult | None:
    money: Final = tuple(item for item in signals.runway if item.metric in _MONEY_METRICS)
    return money[0] if money else None


def _exhaustion_hits(
    signals: ScopeSignals, metrics: frozenset[str], lookahead_days: int | None, today: date
) -> tuple[RunwayResult, ...]:
    horizon: Final = today + timedelta(days=lookahead_days or 0)
    return tuple(
        item
        for item in signals.runway
        if item.metric in metrics and item.exhaustion_p50 is not None and item.exhaustion_p50 <= horizon
    )


def _forecast_exceeds_budget(rule: AlertRule, signals: ScopeSignals) -> AlertDraft | None:
    budget: Final = _budget_limit(signals)
    if budget is None or budget.limit_value <= 0:
        return None
    ceiling: Final = budget.limit_value * rule.threshold
    if signals.projected_eom <= ceiling:
        return None
    body: Final[BudgetBody] = {
        "projected_eom": str(signals.projected_eom),
        "limit_value": str(budget.limit_value),
        "threshold_pct": float(rule.threshold * 100),
        "probability": budget.prob_overrun_this_period,
        "statement": runway_statement(budget),
    }
    return AlertDraft(
        alert_id=rule.alert_id,
        alert_type=rule.condition_type,
        severity="critical",
        scope=signals.scope,
        title=f"{signals.scope.scope_type} {signals.scope.scope_id} is projected to exceed its budget",
        body=body,
    )


def _exhaustion_alert(
    rule: AlertRule, signals: ScopeSignals, metrics: frozenset[str], today: date
) -> AlertDraft | None:
    hits: Final = _exhaustion_hits(signals, metrics, rule.lookahead_days, today)
    if not hits:
        return None
    soonest: Final = min(hits, key=lambda item: item.exhaustion_p50 or today)
    body: Final[ExhaustionBody] = {
        "quota_id": soonest.quota_id,
        "metric": soonest.metric,
        "exhaustion_p50": str(soonest.exhaustion_p50),
        "exhaustion_p90": str(soonest.exhaustion_p90),
        "probability": soonest.prob_exhaust_before_reset,
        "remaining": str(soonest.remaining),
        "statement": runway_statement(soonest),
    }
    return AlertDraft(
        alert_id=rule.alert_id,
        alert_type=rule.condition_type,
        severity="critical" if soonest.prob_exhaust_before_reset >= 0.5 else "warning",
        scope=signals.scope,
        title=f"{soonest.metric} quota exhausts on {soonest.exhaustion_p50}",
        body=body,
    )


def _growth_alert(rule: AlertRule, signals: ScopeSignals, observed: float | None) -> AlertDraft | None:
    if observed is None or observed <= float(rule.threshold):
        return None
    body: Final[GrowthBody] = {
        "observed_pct": round(observed, 4),
        "threshold_pct": float(rule.threshold),
        "drivers": _driver_shares(signals),
    }
    return AlertDraft(
        alert_id=rule.alert_id,
        alert_type=rule.condition_type,
        severity="warning",
        scope=signals.scope,
        title=f"{signals.scope.scope_type} {signals.scope.scope_id} grew {observed:.1f}%",
        body=body,
    )


def _driver_shares(signals: ScopeSignals) -> tuple[DriverShare, ...]:
    if signals.drivers is None:
        return ()
    return tuple(_driver_share(item.factor, round(item.delta_pct, 4)) for item in signals.drivers.contributions)


def _driver_share(factor: str, delta_pct: float) -> DriverShare:
    share: Final[DriverShare] = {"factor": factor, "delta_pct": delta_pct}
    return share


def _forecast_changed(rule: AlertRule, signals: ScopeSignals) -> AlertDraft | None:
    previous: Final = signals.previous_projected_eom
    if previous is None or previous <= 0:
        return None
    change: Final = float(abs(signals.projected_eom - previous) / previous * 100)
    if change <= float(rule.threshold):
        return None
    body: Final[ForecastChangeBody] = {
        "previous_projected_eom": str(previous),
        "projected_eom": str(signals.projected_eom),
        "change_pct": round(change, 4),
    }
    return AlertDraft(
        alert_id=rule.alert_id,
        alert_type=rule.condition_type,
        severity="warning",
        scope=signals.scope,
        title=f"Projected month-end moved {change:.1f}%",
        body=body,
    )


def _quality_below(rule: AlertRule, signals: ScopeSignals) -> AlertDraft | None:
    if signals.quality_score is None or signals.quality_score >= rule.threshold:
        return None
    body: Final[QualityBody] = {"quality_score": signals.quality_score, "threshold": float(rule.threshold)}
    return AlertDraft(
        alert_id=rule.alert_id,
        alert_type=rule.condition_type,
        severity="info",
        scope=signals.scope,
        title=f"Forecast quality for {signals.scope} is {signals.quality_score}",
        body=body,
    )


def evaluate_rule(rule: AlertRule, signals: ScopeSignals, *, today: date) -> AlertDraft | None:
    """Apply one rule to one scope's freshly computed numbers."""
    match rule.condition_type:
        case "forecast_exceeds_budget":
            return _forecast_exceeds_budget(rule, signals)
        case "budget_exhaustion_within":
            return _exhaustion_alert(rule, signals, _MONEY_METRICS, today)
        case "token_exhaustion_within":
            return _exhaustion_alert(rule, signals, _TOKEN_METRICS, today)
        case "spend_growth_above":
            return _growth_alert(rule, signals, signals.drivers.delta_pct if signals.drivers else None)
        case "usage_growth_above":
            return _growth_alert(rule, signals, signals.usage_growth_pct)
        case "forecast_changed_above":
            return _forecast_changed(rule, signals)
        case "forecast_quality_below":
            return _quality_below(rule, signals)
        case "anomaly_detected":
            return None


def evaluate_rules(
    rules: Sequence[AlertRule], signals: ScopeSignals, *, today: date, now: datetime
) -> tuple[AlertDraft, ...]:
    """Every enabled, in-scope, off-cooldown rule that fires for this scope."""
    return tuple(
        draft
        for draft in (
            evaluate_rule(rule, signals, today=today)
            for rule in rules
            if rule.enabled and rule.covers(signals.scope) and rule.is_cool(now)
        )
        if draft is not None
    )


class NotificationResult(TypedDict):
    type: ReadOnly[str | None]
    delivered: ReadOnly[bool]
    detail: ReadOnly[str | int | None]


class _RuleRow(TypedDict):
    alert_id: ReadOnly[str]
    name: ReadOnly[str]
    scope_type: ReadOnly[str]
    scope_id: ReadOnly[str]
    condition_type: ReadOnly[str]
    threshold: ReadOnly[Decimal]
    lookahead_days: ReadOnly[int | None]
    channels: ReadOnly[Sequence[Mapping[str, str]]]
    cooldown_min: ReadOnly[int]
    enabled: ReadOnly[bool]
    last_triggered_at: ReadOnly[datetime | None]


class EventRow(TypedDict):
    id: ReadOnly[str]
    created_at: ReadOnly[datetime]
    alert_id: ReadOnly[str | None]
    alert_type: ReadOnly[str]
    severity: ReadOnly[str]
    scope_type: ReadOnly[str]
    scope_id: ReadOnly[str]
    title: ReadOnly[str]
    body: ReadOnly[Mapping[str, object]]
    status: ReadOnly[str]
    acked_by: ReadOnly[str | None]


_RULE_ROWS: Final = TypeAdapter(tuple[_RuleRow, ...])
_EVENT_ROWS: Final = TypeAdapter(tuple[EventRow, ...])

_RULE_COLUMNS: Final = (
    "alert_id, name, scope_type, scope_id, condition_type, threshold, lookahead_days, "
    "channels, cooldown_min, enabled, last_triggered_at"
)

_RULES_SQL: Final = f'SELECT {_RULE_COLUMNS} FROM "WITOS_FinOpsAlertRule" ORDER BY name'

_INSERT_EVENT_SQL: Final = """
INSERT INTO "WITOS_FinOpsAlertEvent" (
    id, created_at, alert_id, alert_type, severity, scope_type, scope_id, title, body, status, notified_via
)
VALUES (
    gen_random_uuid()::text, (now() AT TIME ZONE 'UTC'), $1, $2, $3, $4, $5, $6, $7::jsonb, 'open', $8::jsonb
)
RETURNING id
"""

_TOUCH_RULE_SQL: Final = """
UPDATE "WITOS_FinOpsAlertRule" SET last_triggered_at = (now() AT TIME ZONE 'UTC'), updated_at = (now() AT TIME ZONE 'UTC')
WHERE alert_id = $1
"""

_EVENTS_SQL: Final = """
SELECT id, created_at, alert_id, alert_type, severity, scope_type, scope_id, title, body, status, acked_by
FROM "WITOS_FinOpsAlertEvent"
WHERE ($1::text IS NULL OR status = $1)
  AND ($2::text IS NULL OR scope_type = $2)
  AND ($3::text IS NULL OR scope_id = $3)
ORDER BY created_at DESC
LIMIT $4
"""

_ACK_SQL: Final = """
UPDATE "WITOS_FinOpsAlertEvent" SET status = $2, acked_by = $3 WHERE id = $1
RETURNING id, created_at, alert_id, alert_type, severity, scope_type, scope_id, title, body, status, acked_by
"""


class AlertStore:
    """Rules in, events out, plus the channel fan-out for anything that fires."""

    def __init__(self, executor: SqlExecutor, *, poster: HttpPoster | None = None) -> None:
        self._executor: Final = executor
        self._poster: Final = poster or HttpxPoster()

    async def rules(self) -> tuple[AlertRule, ...]:
        rows: Final = _RULE_ROWS.validate_python(await self._executor.query(_RULES_SQL))
        return tuple(_as_rule(row) for row in rows)

    async def raise_event(self, draft: AlertDraft, channels: Sequence[Mapping[str, str]] = ()) -> str:
        """Write the receipt first, then notify: an alert nobody could deliver still happened."""
        notified: Final = tuple([await self._notify(channel, draft) for channel in channels])
        rows: Final = await self._executor.query(
            _INSERT_EVENT_SQL,
            draft.alert_id,
            draft.alert_type,
            draft.severity,
            draft.scope.scope_type,
            draft.scope.scope_id,
            draft.title,
            _as_json(draft.body),
            _as_json_list(notified),
        )
        if draft.alert_id:
            await self._executor.execute(_TOUCH_RULE_SQL, draft.alert_id)
        return str(rows[0]["id"]) if rows else ""

    async def _notify(self, channel: Mapping[str, str], draft: AlertDraft) -> NotificationResult:
        url: Final = channel.get("url")
        kind: Final = channel.get("type")
        if kind not in DELIVERABLE_CHANNELS or not url:
            verbose_proxy_logger.info("WIT OS FinOps alert channel %s is not deliverable from the proxy", kind)
            unsupported: Final[NotificationResult] = {
                "type": kind,
                "delivered": False,
                "detail": "unsupported_channel",
            }
            return unsupported
        payload: Final[WebhookPayload] = {
            "alert_type": draft.alert_type,
            "severity": draft.severity,
            "scope_type": draft.scope.scope_type,
            "scope_id": draft.scope.scope_id,
            "title": draft.title,
            "body": draft.body,
        }
        result: Final = await deliver(
            url,
            payload,
            secret_name=channel.get("secret_name"),
            event_id=draft.alert_id or draft.alert_type,
            poster=self._poster,
        )
        delivered: Final[NotificationResult] = {
            "type": kind,
            "delivered": isinstance(result, Delivered),
            "detail": result.status_code if isinstance(result, Delivered) else result.reason,
        }
        return delivered

    async def events(
        self, *, status: str | None = None, scope: ScopeRef | None = None, limit: int = 100
    ) -> tuple[EventRow, ...]:
        return _EVENT_ROWS.validate_python(
            await self._executor.query(
                _EVENTS_SQL,
                status,
                scope.scope_type if scope else None,
                scope.scope_id if scope else None,
                limit,
            )
        )

    async def set_status(self, event_id: str, *, status: str, acked_by: str | None) -> EventRow | None:
        rows: Final = _EVENT_ROWS.validate_python(await self._executor.query(_ACK_SQL, event_id, status, acked_by))
        return rows[0] if rows else None

    async def create_rule(
        self,
        *,
        name: str,
        scope_type: str,
        scope_id: str,
        condition_type: str,
        threshold: Decimal,
        lookahead_days: int | None,
        channels: Sequence[Mapping[str, str]],
        cooldown_min: int,
        enabled: bool,
    ) -> AlertRule:
        rows: Final = _RULE_ROWS.validate_python(
            await self._executor.query(
                _INSERT_RULE_SQL,
                name,
                scope_type,
                scope_id,
                condition_type,
                threshold,
                lookahead_days,
                _as_json_list(channels),
                cooldown_min,
                enabled,
            )
        )
        return _as_rule(rows[0])

    async def update_rule(
        self,
        alert_id: str,
        *,
        threshold: Decimal | None = None,
        lookahead_days: int | None = None,
        cooldown_min: int | None = None,
        enabled: bool | None = None,
    ) -> AlertRule | None:
        rows: Final = _RULE_ROWS.validate_python(
            await self._executor.query(_UPDATE_RULE_SQL, alert_id, threshold, lookahead_days, cooldown_min, enabled)
        )
        return _as_rule(rows[0]) if rows else None

    async def open_event_count(self) -> int:
        rows: Final = _COUNT_ROWS.validate_python(await self._executor.query(_OPEN_EVENTS_SQL))
        return rows[0]["open_events"] if rows else 0


_KNOWN_CONDITIONS: Final = frozenset(
    (
        "forecast_exceeds_budget",
        "budget_exhaustion_within",
        "token_exhaustion_within",
        "spend_growth_above",
        "usage_growth_above",
        "forecast_changed_above",
        "anomaly_detected",
        "forecast_quality_below",
    )
)


def _as_json(document: Mapping[str, object]) -> str:
    serialisable: Final = dict(document)  # mutable-ok: json.dumps needs a real dict
    return json.dumps(serialisable, default=str)


def _as_json_list(documents: Sequence[Mapping[str, object]]) -> str:
    serialisable: Final = [  # mutable-ok: json.dumps needs a real list
        dict(document)  # mutable-ok: json.dumps needs a real dict
        for document in documents
    ]
    return json.dumps(serialisable, default=str)


def _condition(value: str) -> ConditionType:
    """Narrow a stored condition string, defaulting to the one rule that never fires by itself."""
    return value if value in _KNOWN_CONDITIONS else "anomaly_detected"  # pyright: ignore[reportReturnType]  # membership in the frozenset is the narrowing


_INSERT_RULE_SQL: Final = """
INSERT INTO "WITOS_FinOpsAlertRule" (
    alert_id, name, scope_type, scope_id, condition_type, threshold, lookahead_days,
    channels, cooldown_min, enabled, created_at, updated_at
)
VALUES (
    gen_random_uuid()::text, $1, $2, $3, $4, $5::numeric, $6::int, $7::jsonb, $8::int, $9::boolean,
    (now() AT TIME ZONE 'UTC'), (now() AT TIME ZONE 'UTC')
)
RETURNING alert_id, name, scope_type, scope_id, condition_type, threshold, lookahead_days,
          channels, cooldown_min, enabled, last_triggered_at
"""

_UPDATE_RULE_SQL: Final = """
UPDATE "WITOS_FinOpsAlertRule"
SET threshold = COALESCE($2::numeric, threshold),
    lookahead_days = COALESCE($3::int, lookahead_days),
    cooldown_min = COALESCE($4::int, cooldown_min),
    enabled = COALESCE($5::boolean, enabled),
    updated_at = (now() AT TIME ZONE 'UTC')
WHERE alert_id = $1
RETURNING alert_id, name, scope_type, scope_id, condition_type, threshold, lookahead_days,
          channels, cooldown_min, enabled, last_triggered_at
"""

_OPEN_EVENTS_SQL: Final = """SELECT COUNT(*)::int AS open_events FROM "WITOS_FinOpsAlertEvent" WHERE status = 'open'"""


class _CountRow(TypedDict):
    open_events: ReadOnly[int]


_COUNT_ROWS: Final = TypeAdapter(tuple[_CountRow, ...])


def _as_rule(row: _RuleRow) -> AlertRule:
    return AlertRule(
        alert_id=row["alert_id"],
        name=row["name"],
        scope_type=row["scope_type"],
        scope_id=row["scope_id"],
        condition_type=_condition(row["condition_type"]),
        threshold=row["threshold"],
        lookahead_days=row["lookahead_days"],
        channels=row["channels"],
        cooldown_min=row["cooldown_min"],
        enabled=row["enabled"],
        last_triggered_at=row["last_triggered_at"],
    )
