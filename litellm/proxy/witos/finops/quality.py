"""Forecast-quality telemetry (blueprint §1.6).

Every night this scores what the forecast said about yesterday against what
yesterday actually did, and keeps a rolling 30-day WAPE and bias per scope. The
number is published in the API and in the UI footer for one reason: a forecast
that nobody can check is a claim, and a CFO is entitled to know that this
engine's last month of one-day-ahead predictions ran 8% low before deciding what
to do about next month's.

The measurement is deliberately the *one-day-ahead* prediction rather than the
whole horizon. It is the only lead time that has a fresh actual every day, so the
rolling window refills daily instead of thirty days after the fact, and it is the
lead time least contaminated by a scope's history changing shape underneath it.

Two WAPEs exist and they are not the same number. The backtest WAPE stored in
``forecast_json`` is how the winning model scored on held-out history at
selection time. The ``wape`` column is this: how the published forecast has
actually performed since. The second is the one that gets to be called accuracy.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from itertools import groupby
from types import MappingProxyType
from typing import Final, TypedDict

from pydantic import TypeAdapter
from typing_extensions import ReadOnly

from litellm.proxy.witos.finops.history import ScopeRef
from litellm.proxy.witos.finops.regime import Regime
from litellm.proxy.witos.shared.sql import SqlExecutor

ROLLING_WINDOW_DAYS: Final = 30

# Weights for the composite score. Accuracy dominates, but a model that has been
# accurate for five days has not earned the same confidence as one that has been
# accurate for a quarter, and a scope that just step-changed has not earned
# either.
_ACCURACY_WEIGHT: Final = 0.60
_COVERAGE_WEIGHT: Final = 0.25
_STABILITY_WEIGHT: Final = 0.15
_FULL_COVERAGE_DAYS: Final = 90

_STABILITY: Final[Mapping[str, float]] = MappingProxyType(
    {
        "stable": 1.0,
        "accelerating": 0.9,
        "decelerating": 0.9,
        "volatile": 0.7,
        "step_change": 0.6,
        "insufficient_history": 0.5,
    }
)

# The metric names a forecast row can carry, mapped onto the fact column that
# settles them. Interpolated into SQL from this table only, never from a caller.
_ACTUAL_COLUMNS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "spend_usd": "spend_usd",
        "total_tokens": "total_tokens",
        "prompt_tokens": "prompt_tokens",
        "completion_tokens": "completion_tokens",
        "requests": "request_count",
    }
)

_ACTUAL_CASE: Final = " ".join(
    f"WHEN '{metric}' THEN COALESCE(d.{column}, 0)::numeric" for metric, column in _ACTUAL_COLUMNS.items()
)

_ACCURACY_SQL: Final = f"""
WITH predictions AS (
    SELECT f.scope_type,
           f.scope_id,
           f.metric,
           (f.forecast_json #>> '{{days,0,date}}')::date AS target_date,
           (f.forecast_json #>> '{{days,0,p50}}')::numeric AS predicted
    FROM "WITOS_FinOpsForecast" f
    WHERE f.generated_at >= ($1::timestamptz AT TIME ZONE 'UTC')
      AND jsonb_typeof(f.forecast_json #> '{{days,0}}') = 'object'
      AND ($3::text IS NULL OR f.scope_type = $3)
      AND ($4::text IS NULL OR f.scope_id = $4)
)
SELECT p.scope_type,
       p.scope_id,
       p.metric,
       p.target_date,
       p.predicted,
       (CASE p.metric {_ACTUAL_CASE} ELSE 0 END) AS actual
FROM predictions p
LEFT JOIN "WITOS_FinOpsUsageDaily" d
       ON d.scope_type = p.scope_type
      AND d.scope_id = p.scope_id
      AND d.bucket_date = p.target_date
WHERE p.target_date <= ($2::timestamptz AT TIME ZONE 'UTC')::date
ORDER BY p.scope_type, p.scope_id, p.metric, p.target_date
"""


class _AccuracyRow(TypedDict):
    scope_type: ReadOnly[str]
    scope_id: ReadOnly[str]
    metric: ReadOnly[str]
    target_date: ReadOnly[date]
    predicted: ReadOnly[Decimal]
    actual: ReadOnly[Decimal]


_ACCURACY_ROWS: Final = TypeAdapter(tuple[_AccuracyRow, ...])


def quality_score(*, wape: float | None, history_days: int, regime: Regime) -> int:
    """A 0-100 composite of accuracy, how much history backs it, and how settled the scope is."""
    accuracy: Final = 0.5 if wape is None else max(0.0, 1.0 - min(wape, 1.0))
    coverage: Final = min(1.0, history_days / _FULL_COVERAGE_DAYS)
    stability: Final = _STABILITY.get(regime, 0.5)
    return round(100 * (_ACCURACY_WEIGHT * accuracy + _COVERAGE_WEIGHT * coverage + _STABILITY_WEIGHT * stability))


@dataclass(frozen=True, slots=True)
class AccuracyPoint:
    day: date
    predicted: Decimal
    actual: Decimal

    @property
    def error(self) -> Decimal:
        return self.predicted - self.actual


@dataclass(frozen=True, slots=True)
class ForecastAccuracy:
    """Rolling one-day-ahead accuracy for one scope and metric."""

    scope: ScopeRef
    metric: str
    points: tuple[AccuracyPoint, ...]

    @property
    def sample_days(self) -> int:
        return len(self.points)

    @property
    def wape(self) -> float | None:
        actual: Final = sum(abs(point.actual) for point in self.points)
        if actual <= 0:
            return None
        return float(sum(abs(point.error) for point in self.points) / actual)

    @property
    def bias(self) -> float | None:
        actual: Final = sum(abs(point.actual) for point in self.points)
        if actual <= 0:
            return None
        return float(sum(point.error for point in self.points) / actual)


def _key(row: _AccuracyRow) -> tuple[str, str, str]:
    return row["scope_type"], row["scope_id"], row["metric"]


def _group(rows: Sequence[_AccuracyRow], *, since: date) -> tuple[ForecastAccuracy, ...]:
    """Fold the (already ordered) rows into one accuracy record per scope and metric."""
    return tuple(
        ForecastAccuracy(
            scope=ScopeRef(scope_type=scope_type, scope_id=scope_id),
            metric=metric,
            points=tuple(
                AccuracyPoint(day=row["target_date"], predicted=row["predicted"], actual=row["actual"])
                for row in group
                if row["target_date"] >= since
            ),
        )
        for (scope_type, scope_id, metric), group in groupby(rows, key=_key)
    )


class QualityScorer:
    """Scores published forecasts against what happened, and writes the result back."""

    def __init__(self, executor: SqlExecutor) -> None:
        self._executor: Final = executor

    async def measure(
        self,
        *,
        as_of: date,
        window_days: int = ROLLING_WINDOW_DAYS,
        scope: ScopeRef | None = None,
    ) -> tuple[ForecastAccuracy, ...]:
        since: Final = as_of - timedelta(days=window_days)
        rows: Final = _ACCURACY_ROWS.validate_python(
            await self._executor.query(
                _ACCURACY_SQL,
                datetime.combine(since, datetime.min.time(), tzinfo=timezone.utc),
                datetime.combine(as_of, datetime.min.time(), tzinfo=timezone.utc),
                scope.scope_type if scope else None,
                scope.scope_id if scope else None,
            )
        )
        return _group(rows, since=since)

    async def publish(self, accuracy: Sequence[ForecastAccuracy]) -> int:
        """Write each rolling score onto the scope's current forecast row."""
        written: Final = tuple(
            [
                await self._executor.execute(
                    _PUBLISH_SQL,
                    None if item.wape is None else Decimal(str(round(item.wape, 4))),
                    None if item.bias is None else Decimal(str(round(item.bias, 4))),
                    item.scope.scope_type,
                    item.scope.scope_id,
                    item.metric,
                )
                for item in accuracy
                if item.sample_days > 0
            ]
        )
        return sum(written)


_PUBLISH_SQL: Final = """
UPDATE "WITOS_FinOpsForecast"
SET wape = $1::numeric, bias = $2::numeric
WHERE scope_type = $3 AND scope_id = $4 AND metric = $5 AND is_latest = true
"""
