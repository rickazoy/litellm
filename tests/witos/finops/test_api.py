"""The ``/witos/finops/*`` surface: RBAC on every route, and the shapes WIT OS generates from."""

import json
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.witos.finops import router as router_module
from litellm.proxy.witos.finops.report import REPORT_COLUMNS

from tests.witos.finops.fakes import RecordingExecutor

GENERATED = datetime(2026, 8, 21, 2, 10, tzinfo=timezone.utc)
TEAM = "finance-ai"


def forecast_row(*, metric: str = "spend_usd", scope_id: str = TEAM) -> dict[str, object]:
    days = tuple(
        {
            "date": (GENERATED.date() + timedelta(days=offset)).isoformat(),
            "p10": 90.0,
            "p50": 100.0,
            "p90": 130.0,
        }
        for offset in range(5)
    )
    cumulative = tuple(
        {
            "date": day["date"],
            "p10": 90.0 * (index + 1),
            "p50": 100.0 * (index + 1),
            "p90": 130.0 * (index + 1),
        }
        for index, day in enumerate(days)
    )
    return {
        "forecast_id": "f1",
        "scope_type": "team",
        "scope_id": scope_id,
        "generated_at": GENERATED,
        "forecast_start": GENERATED,
        "forecast_end": GENERATED + timedelta(days=4),
        "metric": metric,
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
            "days": days,
            "cumulative": cumulative,
            "tier": "full_ensemble",
            "confidence": "high",
            "strategy": "TOKEN_BASED",
            "regime": "accelerating",
            "backtest": {"model": "recent_weighted_seasonal_trend", "wape": 0.084, "folds": 6},
            "components": {"prompt_tokens": {"algorithm": "ewma", "wape": 0.08}},
            "model_mix": {"gpt-4o-mini": 0.7, "gpt-5": 0.3},
            "unknown_models": ["gpt-5"],
            "unknown_model_share": 0.3,
            "pricing_fidelity": 1.002,
            "path_count": 500,
            "residual_count": 42,
            "has_band": True,
            "seed": 99,
            "drivers": [
                {"factor": "request_volume", "label": "request volume", "delta_usd": 9.1, "delta_pct": 9.1},
                {"factor": "residual", "label": "residual", "delta_usd": 0.0, "delta_pct": 0.0},
            ],
        },
        "proj_eom": Decimal("3000"),
        "proj_eoq": Decimal("9000"),
        "burn_rate_daily": Decimal("100"),
    }


def runway_row() -> dict[str, object]:
    return {
        "id": "r1",
        "computed_at": GENERATED,
        "quota_id": "q1",
        "scope_type": "team",
        "scope_id": TEAM,
        "consumed": Decimal("50000"),
        "remaining": Decimal("50000"),
        "exhaustion_p50": GENERATED + timedelta(days=6),
        "exhaustion_p90": GENERATED + timedelta(days=3),
        "prob_exhaust_before_reset": Decimal("0.7800"),
        "prob_overrun_this_period": Decimal("0.7800"),
        "survives_cycle": False,
        "metric": "usd",
        "limit_value": Decimal("100000"),
        "period": "month",
        "reset_at": GENERATED + timedelta(days=10),
        "source_type": "litellm_budget",
    }


def executor() -> RecordingExecutor:
    return RecordingExecutor(
        responses=(
            ("ORDER BY proj_eom DESC", (forecast_row(),)),
            ("AND metric = $3", (forecast_row(),)),
            ("r.survives_cycle = false", (runway_row(),)),
            ("r.scope_type = $1 AND r.scope_id = $2", (runway_row(),)),
            ("r.scope_type = ANY($1::text[])", (runway_row(),)),
            ("FROM \"WITOS_FinOpsForecast\"\nWHERE metric = 'spend_usd'", (forecast_row(),)),
            ("open_events", ({"open_events": 3},)),
            ("MAX(generated_at)", ({"generated_at": GENERATED, "scopes": 12},)),
            (
                "GROUP BY scope_type, scope_id",
                (
                    {
                        "scope_type": "team",
                        "scope_id": TEAM,
                        "scope_label": "Finance AI",
                        "spend_usd": Decimal("1200"),
                    },
                    {
                        "scope_type": "global",
                        "scope_id": "global",
                        "scope_label": None,
                        "spend_usd": Decimal("9000"),
                    },
                ),
            ),
            ('FROM "WITOS_FinOpsQuota"', ()),
            ('FROM "WITOS_FinOpsAlertRule"', ()),
            ('FROM "WITOS_FinOpsAlertEvent"', ()),
            ('FROM "WITOS_FinOpsScenario"', ()),
            ("WITOS_FinOpsConfig", ()),
            ("bucket_date AS bucket", ()),
        )
    )


def caller(*, admin: bool = False, scopes: tuple[str, ...] = ("finops:view",)) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="sk-hashed",
        user_id="u1",
        team_id=TEAM,
        org_id="org1",
        user_role=LitellmUserRoles.PROXY_ADMIN if admin else LitellmUserRoles.INTERNAL_USER,
        metadata={"witos_scopes": list(scopes)},
    )


@pytest.fixture
def recorded() -> RecordingExecutor:
    return executor()


@pytest.fixture
def client(recorded: RecordingExecutor) -> Iterator[TestClient]:
    app = FastAPI()
    app.include_router(router_module.router)
    app.dependency_overrides[router_module.finops_executor] = lambda: recorded
    app.dependency_overrides[user_api_key_auth] = lambda: caller(admin=True, scopes=())
    with TestClient(app) as test_client:
        yield test_client


def as_user(client: TestClient, user: UserAPIKeyAuth) -> TestClient:
    client.app.dependency_overrides[user_api_key_auth] = lambda: user  # pyright: ignore[reportAttributeAccessIssue]  # TestClient exposes the app it wraps
    return client


def test_forecast_returns_the_persisted_document_not_a_recomputation(
    client: TestClient, recorded: RecordingExecutor
) -> None:
    response = client.get("/witos/finops/forecast", params={"scope_type": "team", "scope_id": TEAM})
    body = response.json()

    assert response.status_code == 200
    assert body["state"] == "forecast"
    assert body["proj_eom"] == "3000"
    assert body["payload"]["days"][0]["p90"] == 130.0
    assert body["payload"]["drivers"][-1]["factor"] == "residual"
    assert all("INSERT" not in sql for sql in recorded.statements())


def test_a_scope_with_no_forecast_gets_a_state_rather_than_zeros(client: TestClient) -> None:
    empty = RecordingExecutor()
    client.app.dependency_overrides[router_module.finops_executor] = lambda: empty  # pyright: ignore[reportAttributeAccessIssue]  # TestClient exposes the app it wraps

    body = client.get("/witos/finops/forecast", params={"scope_type": "team", "scope_id": TEAM}).json()

    assert body["state"] == "no_forecast"
    assert body["payload"] is None
    assert body["proj_eom"] is None


def test_runway_carries_the_cfo_sentence(client: TestClient) -> None:
    body = client.get("/witos/finops/runway", params={"scope_type": "team", "scope_id": TEAM}).json()

    assert body["quotas"][0]["prob_exhaust_before_reset"] == pytest.approx(0.78)
    assert body["quotas"][0]["survives_cycle"] is False
    assert "78% probability of exceeding the $100,000 monthly budget" in body["quotas"][0]["statement"]


def test_overview_reports_the_fleet_headline_and_open_alerts(client: TestClient) -> None:
    body = client.get("/witos/finops/overview").json()

    assert body["open_alerts"] == 3
    assert body["mtd_spend"] == "9000"
    assert body["top_scopes"][0]["scope_id"] == TEAM
    assert body["runway_risks"][0]["quota_id"] == "q1"


def test_the_csv_export_has_the_documented_columns_in_order(client: TestClient) -> None:
    response = client.get("/witos/finops/report", params={"format": "csv", "scope_type": "team"})
    header = response.text.splitlines()[0]

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert header == ",".join(REPORT_COLUMNS)
    assert "attachment" in response.headers["content-disposition"]


def test_model_mix_marks_the_models_litellm_cannot_price(client: TestClient) -> None:
    body = client.get("/witos/finops/model-mix", params={"scope_type": "team", "scope_id": TEAM}).json()
    priced = {entry["model_group"]: entry["priced"] for entry in body["forecast_shares"]}

    assert priced == {"gpt-4o-mini": True, "gpt-5": False}
    assert body["unknown_model_share"] == 0.3


def test_admin_status_reports_freshness_and_the_algorithm_version(client: TestClient) -> None:
    body = client.get("/witos/finops/admin/status").json()

    assert body["forecast_scopes"] == 12
    assert body["algorithm_version"] == "2.0.0"
    assert body["open_alerts"] == 3


def test_a_key_without_the_view_scope_is_refused(client: TestClient) -> None:
    as_user(client, caller(scopes=()))

    response = client.get("/witos/finops/forecast", params={"scope_type": "team", "scope_id": TEAM})

    assert response.status_code == 403
    assert "finops:view" in json.dumps(response.json())


def test_a_key_may_not_read_another_teams_forecast(client: TestClient) -> None:
    as_user(client, caller())

    response = client.get("/witos/finops/forecast", params={"scope_type": "team", "scope_id": "other-team"})

    assert response.status_code == 403


def test_a_key_with_no_scope_id_is_clamped_to_its_own_team(client: TestClient) -> None:
    as_user(client, caller())

    body = client.get("/witos/finops/forecast", params={"scope_type": "team"}).json()

    assert body["scope_id"] == TEAM


def test_a_non_admin_may_not_read_a_fleet_wide_scope(client: TestClient) -> None:
    as_user(client, caller())

    assert (
        client.get("/witos/finops/forecast", params={"scope_type": "global", "scope_id": "global"}).status_code == 403
    )


def test_viewing_does_not_grant_managing(client: TestClient) -> None:
    as_user(client, caller(scopes=("finops:view",)))

    quota = client.post(
        "/witos/finops/quotas",
        json={
            "scope_type": "team",
            "scope_id": TEAM,
            "metric": "usd",
            "limit_value": "100",
            "period": "month",
            "period_start": GENERATED.isoformat(),
        },
    )
    rebuild = client.post("/witos/finops/admin/rebuild", json={})

    assert quota.status_code == 403
    assert rebuild.status_code == 403


def test_running_scenarios_is_its_own_scope(client: TestClient) -> None:
    as_user(client, caller(scopes=("finops:view",)))

    response = client.post(
        "/witos/finops/scenarios/evaluate",
        json={"scope_type": "team", "scope_id": TEAM, "assumptions": {"usage_growth_pct": 10}},
    )

    assert response.status_code == 403


def test_a_scenario_needing_baseline_users_is_refused_with_the_reason(client: TestClient) -> None:
    as_user(client, caller(admin=True, scopes=("finops:run_scenarios",)))

    response = client.post(
        "/witos/finops/scenarios/evaluate",
        json={"scope_type": "team", "scope_id": TEAM, "assumptions": {"new_users": 150}},
    )

    assert response.status_code == 400
    assert "baseline_users" in json.dumps(response.json())


def test_editing_a_mirrored_quota_is_refused_with_an_explanation(client: TestClient) -> None:
    mirrored = RecordingExecutor(
        responses=(
            ("UPDATE", ()),
            (
                'FROM "WITOS_FinOpsQuota" WHERE quota_id',
                (
                    {
                        "quota_id": "q1",
                        "scope_type": "team",
                        "scope_id": TEAM,
                        "metric": "usd",
                        "limit_value": Decimal("100"),
                        "period": "month",
                        "period_start": GENERATED,
                        "period_end": None,
                        "reset_at": None,
                        "source_type": "litellm_budget",
                        "soft_threshold_pct": None,
                    },
                ),
            ),
        )
    )
    client.app.dependency_overrides[router_module.finops_executor] = lambda: mirrored  # pyright: ignore[reportAttributeAccessIssue]  # TestClient exposes the app it wraps
    as_user(client, caller(admin=True, scopes=("finops:manage_budget",)))

    response = client.patch("/witos/finops/quotas/q1", json={"limit_value": "200"})

    assert response.status_code == 409
    assert "read-only" in json.dumps(response.json())


def test_every_route_is_documented_in_the_openapi_schema(client: TestClient) -> None:
    schema = client.app.openapi()  # pyright: ignore[reportAttributeAccessIssue]  # TestClient exposes the app it wraps
    paths = set(schema["paths"])

    assert "/witos/finops/forecast" in paths
    assert "/witos/finops/runway" in paths
    assert "/witos/finops/scenarios/evaluate" in paths
    assert "/witos/finops/report" in paths
    assert "/witos/finops/stream" in paths
    assert "/witos/finops/admin/rebuild" in paths
