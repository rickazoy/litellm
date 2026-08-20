"""Intraday anomaly detection on the hourly facts (blueprint §1.8).

Forecasts answer "what will this month cost". This answers "something is wrong
right now", which is a different question with a different tolerance for delay:
the run loop is fifteen minutes, and the input is the hourly fact table rather
than the daily one the forecast engine reads.

The detector is an exponentially weighted mean and variance per scope, which
adapts to a workload's own level instead of comparing every team against a fleet
constant, and it demands two consecutive buckets above the threshold before it
says anything. One hour above four sigma is a batch job; two in a row is a
change.

The second detector has nothing to do with statistics. A model group appearing in
a scope for the first time and immediately billing real money is worth an
``info`` event even when the volume is unremarkable, because it is usually
somebody pointing production at a model nobody costed.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from itertools import accumulate, groupby
from math import sqrt
from types import MappingProxyType
from typing import Final, TypedDict

from pydantic import TypeAdapter
from typing_extensions import ReadOnly

from litellm.proxy.witos.finops.alerts import AlertDraft, AlertStore, AnomalyBody, NewModelBody
from litellm.proxy.witos.finops.history import ScopeRef
from litellm.proxy.witos.shared.sql import SqlExecutor

EWMA_ALPHA: Final = 0.3
Z_THRESHOLD: Final = 4.0
CONSECUTIVE_BUCKETS: Final = 2
LOOKBACK_HOURS: Final = 72
NEW_MODEL_SPEND_USD: Final = Decimal(5)

# Buckets used to estimate the scale before any bucket can be scored against it.
WARMUP_BUCKETS: Final = 8
RELATIVE_NOISE_FLOOR: Final = 0.05

# Anomaly detection runs across every scope of these types in one query. Key and
# end-user scopes are excluded by default: they are the highest-cardinality
# dimensions in the fact tables and the alert a human acts on is nearly always at
# team level or above.
DEFAULT_ANOMALY_SCOPE_TYPES: Final = ("global", "organization", "team")

_HOURLY_SQL: Final = """
SELECT scope_type, scope_id, bucket_start_utc, spend_usd, request_count, breakdown
FROM "WITOS_FinOpsUsageHourly"
WHERE bucket_start_utc >= ($1::timestamptz AT TIME ZONE 'UTC')
  AND scope_type = ANY($2::text[])
ORDER BY scope_type, scope_id, bucket_start_utc
"""


class _HourlyRow(TypedDict):
    scope_type: ReadOnly[str]
    scope_id: ReadOnly[str]
    bucket_start_utc: ReadOnly[datetime]
    spend_usd: ReadOnly[Decimal]
    request_count: ReadOnly[int]
    breakdown: ReadOnly[Mapping[str, Mapping[str, Decimal]] | None]


_HOURLY_ROWS: Final = TypeAdapter(tuple[_HourlyRow, ...])
_NO_BREAKDOWN: Final[Mapping[str, Mapping[str, Decimal]]] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class ScopeBuckets:
    scope: ScopeRef
    buckets: tuple[_HourlyRow, ...]


def ewma_z_scores(
    values: Sequence[float], *, alpha: float = EWMA_ALPHA, warmup: int = WARMUP_BUCKETS
) -> tuple[float, ...]:
    """Each bucket's deviation from the exponentially weighted level before it, in baseline sigmas.

    The level adapts and the scale does not. An exponentially weighted variance
    would be inflated by the first anomalous bucket badly enough that the second
    one, which is the whole point of requiring two in a row, could no longer
    clear the threshold. Estimating the scale once from a quiet warm-up window is
    the standard EWMA control chart, and it is what makes a *sustained* excursion
    detectable rather than only its first hour.

    The scale carries a floor of 5% of the baseline level. A scope whose hourly
    spend has been literally constant does exist (a fixed nightly batch), and
    without a floor its first ordinary hour of variation would divide by zero and
    page somebody.
    """
    if len(values) <= warmup:
        return (0.0,) * len(values)
    head: Final = values[:warmup]
    baseline: Final = sum(head) / len(head)
    spread: Final = sqrt(sum((value - baseline) ** 2 for value in head) / len(head))
    scale: Final = max(spread, RELATIVE_NOISE_FLOOR * abs(baseline))

    def step(state: tuple[float, float], value: float) -> tuple[float, float]:
        level, _ = state
        deviation: Final = value - level
        return (level + alpha * deviation, deviation / scale if scale > 0 else 0.0)

    scored: Final = tuple(accumulate(values[warmup:], step, initial=(baseline, 0.0)))
    return (0.0,) * warmup + tuple(state[1] for state in scored[1:])


def _sustained(z_scores: Sequence[float], *, threshold: float, run_length: int) -> int | None:
    """Index of the last bucket of the first run of ``run_length`` scores over the threshold."""
    hits: Final = tuple(
        index
        for index in range(run_length - 1, len(z_scores))
        if all(z_scores[index - offset] >= threshold for offset in range(run_length))
    )
    return hits[0] if hits else None


def _top_contributor(breakdown: Mapping[str, Mapping[str, Decimal]] | None) -> str | None:
    if not breakdown:
        return None
    return max(breakdown, key=lambda model: Decimal(str(breakdown[model].get("spend", 0))))


def _spend_anomaly(scope: ScopeRef, buckets: Sequence[_HourlyRow]) -> AlertDraft | None:
    z_scores: Final = ewma_z_scores(tuple(float(row["spend_usd"]) for row in buckets))
    index: Final = _sustained(z_scores, threshold=Z_THRESHOLD, run_length=CONSECUTIVE_BUCKETS)
    if index is None:
        return None
    row: Final = buckets[index]
    body: Final[AnomalyBody] = {
        "observed_spend_usd": str(row["spend_usd"]),
        "observed_requests": row["request_count"],
        "z": round(z_scores[index], 3),
        "consecutive_buckets": CONSECUTIVE_BUCKETS,
        "bucket_start_utc": row["bucket_start_utc"].isoformat(),
        "top_model": _top_contributor(row["breakdown"]),
    }
    return AlertDraft(
        alert_id=None,
        alert_type="anomaly_detected",
        severity="warning",
        scope=scope,
        title=f"Spend anomaly for {scope} at {row['bucket_start_utc']:%Y-%m-%d %H:%M} UTC",
        body=body,
    )


def _new_model_alerts(
    scope: ScopeRef, buckets: Sequence[_HourlyRow], *, spend_floor: Decimal
) -> tuple[AlertDraft, ...]:
    if len(buckets) < 2:
        return ()
    latest: Final = buckets[-1]
    seen_before: Final = frozenset(model for row in buckets[:-1] for model in (row["breakdown"] or _NO_BREAKDOWN))
    return tuple(
        AlertDraft(
            alert_id=None,
            alert_type="new_model_cost",
            severity="info",
            scope=scope,
            title=f"New model {model} is billing in {scope}",
            body=_new_model_body(model, values.get("spend", Decimal(0)), latest["bucket_start_utc"]),
        )
        for model, values in (latest["breakdown"] or _NO_BREAKDOWN).items()
        if model not in seen_before and Decimal(str(values.get("spend", 0))) > spend_floor
    )


def _new_model_body(model: str, spend: Decimal, bucket_start: datetime) -> NewModelBody:
    body: Final[NewModelBody] = {
        "model": model,
        "hourly_spend_usd": str(spend),
        "bucket_start_utc": bucket_start.isoformat(),
    }
    return body


def detect(scopes: Sequence[ScopeBuckets], *, spend_floor: Decimal = NEW_MODEL_SPEND_USD) -> tuple[AlertDraft, ...]:
    """Every anomaly in the window, as drafts. Pure: no clock, no database, no delivery."""
    return tuple(
        draft
        for entry in scopes
        for draft in (
            (_spend_anomaly(entry.scope, entry.buckets),)
            + _new_model_alerts(entry.scope, entry.buckets, spend_floor=spend_floor)
        )
        if draft is not None
    )


class AnomalyDetector:
    """Reads the hourly facts, runs both detectors, writes the events."""

    def __init__(
        self,
        executor: SqlExecutor,
        store: AlertStore,
        *,
        scope_types: Sequence[str] = DEFAULT_ANOMALY_SCOPE_TYPES,
        lookback_hours: int = LOOKBACK_HOURS,
    ) -> None:
        self._executor: Final = executor
        self._store: Final = store
        self._scope_types: Final = tuple(scope_types)
        self._lookback: Final = timedelta(hours=lookback_hours)

    async def buckets(self, *, now: datetime) -> tuple[ScopeBuckets, ...]:
        scopes: Final = list(self._scope_types)  # mutable-ok: an array bind needs a list, not a tuple
        rows: Final = _HOURLY_ROWS.validate_python(
            await self._executor.query(_HOURLY_SQL, now - self._lookback, scopes)
        )
        return tuple(
            ScopeBuckets(scope=ScopeRef(scope_type=scope_type, scope_id=scope_id), buckets=tuple(group))
            for (scope_type, scope_id), group in groupby(rows, key=lambda row: (row["scope_type"], row["scope_id"]))
        )

    async def run(self, *, now: datetime | None = None) -> tuple[str, ...]:
        moment: Final = now or datetime.now(timezone.utc)
        drafts: Final = detect(await self.buckets(now=moment))
        return tuple([await self._store.raise_event(draft) for draft in drafts])
