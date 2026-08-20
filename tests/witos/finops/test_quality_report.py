"""Forecast-quality telemetry and the CFO CSV export."""

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from litellm.proxy.witos.finops.quality import (
    AccuracyPoint,
    ForecastAccuracy,
    QualityScorer,
    quality_score,
)
from litellm.proxy.witos.finops.history import ScopeRef
from litellm.proxy.witos.finops.report import REPORT_COLUMNS, build_csv

from tests.witos.finops.fakes import RecordingExecutor

SCOPE = ScopeRef("team", "finance-ai")
GENERATED = datetime(2026, 8, 21, 2, 10, tzinfo=timezone.utc)


def accuracy(points: tuple[tuple[str, str], ...]) -> ForecastAccuracy:
    return ForecastAccuracy(
        scope=SCOPE,
        metric="spend_usd",
        points=tuple(
            AccuracyPoint(day=date(2026, 8, 1) + timedelta(days=index), predicted=Decimal(p), actual=Decimal(a))
            for index, (p, a) in enumerate(points)
        ),
    )


def test_rolling_wape_and_bias_measure_different_failures() -> None:
    """Consistently 10% low is a bias; alternating 10% either way is not."""
    low = accuracy((("90", "100"), ("90", "100"), ("90", "100")))
    noisy = accuracy((("90", "100"), ("110", "100"), ("90", "100")))

    assert low.wape == pytest.approx(0.1)
    assert low.bias == pytest.approx(-0.1)
    assert noisy.wape == pytest.approx(0.1)
    assert noisy.bias == pytest.approx(-0.0333, abs=0.001)


def test_a_scope_that_did_nothing_has_no_wape_rather_than_a_perfect_one() -> None:
    idle = accuracy((("0", "0"), ("0", "0")))

    assert idle.wape is None
    assert idle.bias is None
    assert idle.sample_days == 2


def test_quality_score_rewards_accuracy_history_and_stability() -> None:
    best = quality_score(wape=0.02, history_days=180, regime="stable")
    thin = quality_score(wape=0.02, history_days=5, regime="stable")
    unsettled = quality_score(wape=0.02, history_days=180, regime="step_change")
    inaccurate = quality_score(wape=0.60, history_days=180, regime="stable")

    assert best > thin > 0
    assert best > unsettled
    assert best > inaccurate
    assert 0 <= inaccurate <= 100


def test_an_unmeasurable_forecast_scores_in_the_middle_not_at_the_top() -> None:
    assert quality_score(wape=None, history_days=90, regime="stable") < quality_score(
        wape=0.0, history_days=90, regime="stable"
    )


async def test_the_scorer_only_publishes_scopes_it_actually_measured() -> None:
    executor = RecordingExecutor()
    scorer = QualityScorer(executor)

    written = await scorer.publish((accuracy((("90", "100"),)), ForecastAccuracy(scope=SCOPE, metric="x", points=())))

    assert written == 1
    assert len(executor.executions) == 1
    assert "is_latest = true" in executor.executions[0][0]


async def test_the_accuracy_query_is_scoped_and_bounded_by_the_window() -> None:
    executor = RecordingExecutor()

    await QualityScorer(executor).measure(as_of=date(2026, 8, 20), scope=SCOPE)
    statement, args = executor.queries[0]

    assert "WITOS_FinOpsForecast" in statement
    assert "days,0,p50" in statement
    assert args[2:] == (SCOPE.scope_type, SCOPE.scope_id)


def forecast_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "forecast_id": "f1",
        "scope_type": "team",
        "scope_id": "finance-ai",
        "generated_at": GENERATED,
        "forecast_start": GENERATED,
        "forecast_end": GENERATED + timedelta(days=9),
        "metric": "spend_usd",
        "strategy": "TOKEN_BASED",
        "algorithm": "usage_priced",
        "algorithm_version": "2.0.0",
        "history_days": 90,
        "training_points": 90,
        "trend_regime": "accelerating",
        "wape": Decimal("0.0840"),
        "bias": Decimal("0.0190"),
        "quality_score": 91,
        "pricing_snapshot_hash": "abc123",
        "forecast_json": {
            "cumulative": [
                {"date": "2026-08-21", "p10": 90.0, "p50": 100.0, "p90": 130.0},
                {"date": "2026-08-31", "p10": 900.0, "p50": 1000.0, "p90": 1300.0},
                {"date": "2026-09-05", "p10": 1400.0, "p50": 1500.0, "p90": 1900.0},
            ]
        },
        "proj_eom": Decimal("3000"),
        "proj_eoq": Decimal("9000"),
        "burn_rate_daily": Decimal("100"),
    }
    return {**row, **overrides}


def runway_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "id": "r1",
        "computed_at": GENERATED,
        "quota_id": "q1",
        "scope_type": "team",
        "scope_id": "finance-ai",
        "consumed": Decimal("500"),
        "remaining": Decimal("1500"),
        "exhaustion_p50": GENERATED + timedelta(days=6),
        "exhaustion_p90": GENERATED + timedelta(days=3),
        "prob_exhaust_before_reset": Decimal("0.7800"),
        "prob_overrun_this_period": Decimal("0.7800"),
        "survives_cycle": False,
        "metric": "usd",
        "limit_value": Decimal("2000"),
        "period": "month",
        "reset_at": None,
        "source_type": "litellm_budget",
    }
    return {**row, **overrides}


def test_the_export_stops_at_the_month_end_and_reports_both_quantiles() -> None:
    body = build_csv(
        (forecast_row(),),  # pyright: ignore[reportArgumentType]  # a hand-built row of the queried shape
        runway=(runway_row(),),  # pyright: ignore[reportArgumentType]  # a hand-built row of the queried shape
        actuals={("team", "finance-ai"): Decimal("500")},
        as_of=date(2026, 8, 20),
    )
    header, row = body.splitlines()
    fields = dict(zip(header.split(","), row.split(","), strict=True))

    assert fields["period_end"] == "2026-08-31"
    assert fields["actual_mtd_usd"] == "500.0000"
    assert fields["projected_eom_p50_usd"] == "1500.0000", "the 2026-09-05 point is past month end"
    assert fields["projected_eom_p90_usd"] == "1800.0000"


def test_variance_against_budget_is_signed_and_reported_as_a_percentage() -> None:
    body = build_csv(
        (forecast_row(),),  # pyright: ignore[reportArgumentType]  # a hand-built row of the queried shape
        runway=(runway_row(),),  # pyright: ignore[reportArgumentType]  # a hand-built row of the queried shape
        actuals={("team", "finance-ai"): Decimal("500")},
        as_of=date(2026, 8, 20),
    )
    fields = dict(zip(body.splitlines()[0].split(","), body.splitlines()[1].split(","), strict=True))

    assert fields["budget_usd"] == "2000.0000"
    assert fields["projected_variance_usd"] == "-500.0000"
    assert fields["projected_variance_pct"] == "-25.00"
    assert fields["budget_consumed_pct"] == "25.00"
    assert fields["survives_cycle"] == "false"


def test_a_scope_with_no_quota_still_exports_its_forecast() -> None:
    body = build_csv(
        (forecast_row(),),  # pyright: ignore[reportArgumentType]  # a hand-built row of the queried shape
        runway=(),
        actuals={},
        as_of=date(2026, 8, 20),
    )
    fields = dict(zip(body.splitlines()[0].split(","), body.splitlines()[1].split(","), strict=True))

    assert fields["budget_usd"] == ""
    assert fields["exhaustion_p50"] == ""
    assert fields["actual_mtd_usd"] == "0.0000"
    assert fields["forecast_quality_score"] == "91"


def test_the_column_list_is_the_documented_contract() -> None:
    """Spreadsheets have formulas against these positions; new columns go on the end."""
    assert REPORT_COLUMNS[:6] == (
        "scope_type",
        "scope_id",
        "period_end",
        "actual_mtd_usd",
        "projected_eom_p50_usd",
        "projected_eom_p90_usd",
    )
    assert REPORT_COLUMNS[-1] == "pricing_snapshot_hash"
    assert len(set(REPORT_COLUMNS)) == len(REPORT_COLUMNS)
