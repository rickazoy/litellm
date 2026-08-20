"""The CFO export (blueprint §1.11).

One row per scope, a fixed column list, and every column derived from the same
persisted forecast the dashboard is drawing. The columns are fixed because this
file ends up in a spreadsheet that somebody has already built formulas against,
so adding a column in the middle next quarter would silently break their model:
new columns go on the end, existing ones never move or change meaning.

Both a P50 and a P90 month-end figure are exported. A single number invites the
reader to treat a forecast as a commitment, and the whole point of the Monte
Carlo is that the honest answer is a range.
"""

import csv
import io
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal
from typing import Final

from litellm.proxy.witos.finops.api_schemas import ForecastPayload
from litellm.proxy.witos.finops.engine import month_end
from litellm.proxy.witos.finops.forecast_store import ForecastRow, RunwayRow
from litellm.proxy.witos.finops.runway import MONEY_METRICS

# Documented and stable. Append-only: never reorder, never repurpose.
REPORT_COLUMNS: Final = (
    "scope_type",
    "scope_id",
    "period_end",
    "actual_mtd_usd",
    "projected_eom_p50_usd",
    "projected_eom_p90_usd",
    "budget_usd",
    "projected_variance_usd",
    "projected_variance_pct",
    "budget_consumed_pct",
    "exhaustion_p50",
    "exhaustion_p90",
    "prob_overrun_this_period",
    "survives_cycle",
    "forecast_algorithm",
    "forecast_regime",
    "forecast_wape",
    "forecast_quality_score",
    "generated_at",
    "pricing_snapshot_hash",
)


def _payload_of(row: ForecastRow) -> ForecastPayload:
    return ForecastPayload.model_validate(row["forecast_json"])


def _cumulative_at(payload: ForecastPayload, boundary: date, *, quantile: str) -> Decimal:
    points: Final = tuple(point for point in payload.cumulative if point.date <= boundary)
    if not points:
        return Decimal(0)
    last: Final = points[-1]
    return Decimal(str(last.p90 if quantile == "p90" else last.p50))


def _money(value: Decimal | None) -> str:
    return "" if value is None else f"{value:.4f}"


def _row_for(
    forecast: ForecastRow,
    *,
    actual_mtd: Decimal,
    runway: RunwayRow | None,
    boundary: date,
) -> tuple[str, ...]:
    payload: Final = _payload_of(forecast)
    projected_p50: Final = actual_mtd + _cumulative_at(payload, boundary, quantile="p50")
    projected_p90: Final = actual_mtd + _cumulative_at(payload, boundary, quantile="p90")
    budget: Final = runway["limit_value"] if runway else None
    variance: Final = None if budget is None else projected_p50 - budget
    return (
        forecast["scope_type"],
        forecast["scope_id"],
        boundary.isoformat(),
        _money(actual_mtd),
        _money(projected_p50),
        _money(projected_p90),
        _money(budget),
        _money(variance),
        "" if budget is None or budget <= 0 else f"{(variance or Decimal(0)) / budget * 100:.2f}",
        ""
        if runway is None or runway["limit_value"] <= 0
        else f"{runway['consumed'] / runway['limit_value'] * 100:.2f}",
        _iso(runway["exhaustion_p50"]) if runway else "",
        _iso(runway["exhaustion_p90"]) if runway else "",
        ""
        if runway is None or runway["prob_overrun_this_period"] is None
        else f"{runway['prob_overrun_this_period']:.4f}",
        "" if runway is None else str(runway["survives_cycle"]).lower(),
        forecast["algorithm"],
        forecast["trend_regime"],
        "" if forecast["wape"] is None else f"{forecast['wape']:.4f}",
        "" if forecast["quality_score"] is None else str(forecast["quality_score"]),
        forecast["generated_at"].isoformat(),
        forecast["pricing_snapshot_hash"] or "",
    )


def _iso(value: datetime | None) -> str:
    return "" if value is None else value.date().isoformat()


def _money_runway(runway: Sequence[RunwayRow], scope: tuple[str, str]) -> RunwayRow | None:
    """The USD quota for a scope, which is the one a CFO report is about."""
    matched: Final = tuple(
        row for row in runway if (row["scope_type"], row["scope_id"]) == scope and row["metric"] in MONEY_METRICS
    )
    return matched[0] if matched else None


def build_csv(
    forecasts: Sequence[ForecastRow],
    *,
    runway: Sequence[RunwayRow],
    actuals: Mapping[tuple[str, str], Decimal],
    as_of: date,
) -> str:
    """Render the export. Pure, so the column contract is testable without a database."""
    boundary: Final = month_end(as_of)
    buffer: Final = io.StringIO()
    writer: Final = csv.writer(buffer, lineterminator="\n")
    writer.writerow(REPORT_COLUMNS)
    writer.writerows(
        _row_for(
            forecast,
            actual_mtd=actuals.get((forecast["scope_type"], forecast["scope_id"]), Decimal(0)),
            runway=_money_runway(runway, (forecast["scope_type"], forecast["scope_id"])),
            boundary=boundary,
        )
        for forecast in forecasts
    )
    return buffer.getvalue()
