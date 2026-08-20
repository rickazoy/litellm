"""Alert rules, cooldown, anomaly detection and HMAC-signed delivery."""

import json
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from litellm.proxy.witos.finops.alerts import (
    AlertDraft,
    AlertRule,
    AlertStore,
    ScopeSignals,
    evaluate_rule,
    evaluate_rules,
)
from litellm.proxy.witos.finops.anomaly import (
    ScopeBuckets,
    detect,
    ewma_z_scores,
)
from litellm.proxy.witos.finops.history import ScopeRef
from litellm.proxy.witos.finops.runway import RunwayResult
from litellm.proxy.witos.shared.webhooks import (
    SIGNATURE_HEADER,
    Delivered,
    DeliveryFailed,
    deliver,
    sign,
    verify,
)

from tests.witos.finops.fakes import RecordingExecutor

SCOPE = ScopeRef("team", "finance-ai")
NOW = datetime(2026, 8, 21, 3, 0, tzinfo=timezone.utc)
TODAY = date(2026, 8, 21)


def rule(condition: str, *, threshold: str = "1.0", lookahead: int | None = None, **kwargs: object) -> AlertRule:
    return AlertRule(
        alert_id=kwargs.get("alert_id", "a1"),  # pyright: ignore[reportArgumentType]  # test-local kwargs
        name="rule",
        scope_type=kwargs.get("scope_type", "team"),  # pyright: ignore[reportArgumentType]  # test-local kwargs
        scope_id=kwargs.get("scope_id", "*"),  # pyright: ignore[reportArgumentType]  # test-local kwargs
        condition_type=condition,  # pyright: ignore[reportArgumentType]  # test-local kwargs
        threshold=Decimal(threshold),
        lookahead_days=lookahead,
        channels=(),
        cooldown_min=kwargs.get("cooldown_min", 240),  # pyright: ignore[reportArgumentType]  # test-local kwargs
        enabled=kwargs.get("enabled", True),  # pyright: ignore[reportArgumentType]  # test-local kwargs
        last_triggered_at=kwargs.get("last_triggered_at"),  # pyright: ignore[reportArgumentType]  # test-local kwargs
    )


def runway(*, metric: str = "usd", exhaustion: date | None = None, probability: float = 0.9) -> RunwayResult:
    return RunwayResult(
        quota_id="q1",
        scope=SCOPE,
        metric=metric,
        limit_value=Decimal("100000"),
        consumed=Decimal("50000"),
        remaining=Decimal("50000"),
        exhaustion_p50=exhaustion,
        exhaustion_p90=exhaustion,
        prob_exhaust_before_reset=probability,
        prob_overrun_this_period=probability,
        survives_cycle=exhaustion is None,
        projected_consumption_pct=90.0,
        burn_per_day=Decimal("1000"),
        cycle_end=TODAY + timedelta(days=10),
        period="month",
    )


def signals(**kwargs: object) -> ScopeSignals:
    return ScopeSignals(
        scope=SCOPE,
        projected_eom=kwargs.get("projected_eom", Decimal("120000")),  # pyright: ignore[reportArgumentType]  # test-local kwargs
        previous_projected_eom=kwargs.get("previous_projected_eom"),  # pyright: ignore[reportArgumentType]  # test-local kwargs
        quality_score=kwargs.get("quality_score", 80),  # pyright: ignore[reportArgumentType]  # test-local kwargs
        runway=kwargs.get("runway", (runway(),)),  # pyright: ignore[reportArgumentType]  # test-local kwargs
        drivers=kwargs.get("drivers"),  # pyright: ignore[reportArgumentType]  # test-local kwargs
        usage_growth_pct=kwargs.get("usage_growth_pct"),  # pyright: ignore[reportArgumentType]  # test-local kwargs
    )


def test_a_projection_over_budget_raises_a_critical_alert_with_the_numbers() -> None:
    draft = evaluate_rule(rule("forecast_exceeds_budget"), signals(), today=TODAY)

    assert draft is not None
    assert draft.severity == "critical"
    assert draft.body["projected_eom"] == "120000"
    assert "probability of exceeding" in str(draft.body["statement"])


def test_a_projection_under_budget_raises_nothing() -> None:
    assert evaluate_rule(rule("forecast_exceeds_budget"), signals(projected_eom=Decimal("10")), today=TODAY) is None


def test_exhaustion_alerts_only_fire_inside_the_lookahead() -> None:
    inside = signals(runway=(runway(exhaustion=TODAY + timedelta(days=3)),))
    outside = signals(runway=(runway(exhaustion=TODAY + timedelta(days=30)),))

    assert evaluate_rule(rule("budget_exhaustion_within", lookahead=7), inside, today=TODAY) is not None
    assert evaluate_rule(rule("budget_exhaustion_within", lookahead=7), outside, today=TODAY) is None


def test_a_token_exhaustion_rule_ignores_the_money_quota() -> None:
    money_only = signals(runway=(runway(exhaustion=TODAY + timedelta(days=1)),))
    tokens = signals(runway=(runway(metric="total_tokens", exhaustion=TODAY + timedelta(days=1)),))

    assert evaluate_rule(rule("token_exhaustion_within", lookahead=7), money_only, today=TODAY) is None
    assert evaluate_rule(rule("token_exhaustion_within", lookahead=7), tokens, today=TODAY) is not None


def test_growth_and_quality_rules_read_their_own_signal() -> None:
    growing = signals(usage_growth_pct=45.0)
    poor = signals(quality_score=30)

    assert evaluate_rule(rule("usage_growth_above", threshold="20"), growing, today=TODAY) is not None
    assert evaluate_rule(rule("usage_growth_above", threshold="90"), growing, today=TODAY) is None
    assert evaluate_rule(rule("forecast_quality_below", threshold="50"), poor, today=TODAY) is not None


def test_a_forecast_that_moved_a_lot_since_yesterday_is_reported() -> None:
    moved = signals(projected_eom=Decimal("150"), previous_projected_eom=Decimal("100"))
    steady = signals(projected_eom=Decimal("101"), previous_projected_eom=Decimal("100"))

    assert evaluate_rule(rule("forecast_changed_above", threshold="20"), moved, today=TODAY) is not None
    assert evaluate_rule(rule("forecast_changed_above", threshold="20"), steady, today=TODAY) is None


def test_cooldown_suppresses_a_rule_that_just_fired() -> None:
    """The failure mode of a budget alert is two hundred messages, not silence."""
    hot = rule("forecast_exceeds_budget", last_triggered_at=NOW - timedelta(minutes=5), cooldown_min=240)
    cool = rule("forecast_exceeds_budget", last_triggered_at=NOW - timedelta(hours=9), cooldown_min=240)

    assert evaluate_rules((hot,), signals(), today=TODAY, now=NOW) == ()
    assert len(evaluate_rules((cool,), signals(), today=TODAY, now=NOW)) == 1


def test_a_disabled_or_out_of_scope_rule_never_fires() -> None:
    disabled = rule("forecast_exceeds_budget", enabled=False)
    elsewhere = rule("forecast_exceeds_budget", scope_type="team", scope_id="other-team")

    assert evaluate_rules((disabled, elsewhere), signals(), today=TODAY, now=NOW) == ()


def test_a_wildcard_rule_covers_every_scope() -> None:
    assert rule("forecast_exceeds_budget", scope_type="*", scope_id="*").covers(SCOPE) is True
    assert rule("forecast_exceeds_budget", scope_type="key", scope_id="*").covers(SCOPE) is False


def test_ewma_z_scores_grow_with_the_size_of_the_departure() -> None:
    steady = ewma_z_scores((100.0,) * 20)
    spiked = ewma_z_scores((100.0,) * 19 + (900.0,))

    assert max(abs(value) for value in steady) == 0.0
    assert spiked[-1] > 4.0


def test_an_anomaly_needs_two_consecutive_buckets() -> None:
    """One hour above four sigma is a batch job; two in a row is a change."""
    single = _buckets((100.0,) * 20 + (900.0, 100.0))
    sustained = _buckets((100.0,) * 20 + (900.0, 950.0))

    assert detect((single,)) == ()
    assert any(draft.alert_type == "anomaly_detected" for draft in detect((sustained,)))


def test_an_anomaly_names_the_model_that_drove_it() -> None:
    sustained = _buckets((100.0,) * 20 + (900.0, 950.0), model="gpt-5")
    drafts = detect((sustained,))

    assert drafts[0].body["top_model"] == "gpt-5"
    assert drafts[0].severity == "warning"


def test_a_new_model_billing_real_money_is_an_info_event() -> None:
    buckets = _buckets((100.0,) * 5, model="gpt-4o-mini")
    with_new = ScopeBuckets(
        scope=buckets.scope,
        buckets=buckets.buckets[:-1]
        + (
            {
                **buckets.buckets[-1],
                "breakdown": {"brand-new-model": {"spend": Decimal("40"), "tokens": Decimal("10")}},
            },
        ),  # pyright: ignore[reportArgumentType]  # a hand-built row of the same shape
    )
    drafts = detect((with_new,))

    assert any(draft.alert_type == "new_model_cost" for draft in drafts)
    assert all(draft.severity in {"info", "warning"} for draft in drafts)


def test_a_cheap_new_model_is_not_worth_an_event() -> None:
    buckets = _buckets((100.0,) * 5)
    with_new = ScopeBuckets(
        scope=buckets.scope,
        buckets=buckets.buckets[:-1]
        + (
            {
                **buckets.buckets[-1],
                "breakdown": {"brand-new-model": {"spend": Decimal("0.01"), "tokens": Decimal("10")}},
            },
        ),  # pyright: ignore[reportArgumentType]  # a hand-built row of the same shape
    )

    assert all(draft.alert_type != "new_model_cost" for draft in detect((with_new,)))


def test_the_receipt_is_written_even_when_the_channel_is_undeliverable() -> None:
    executor = RecordingExecutor(responses=(('INSERT INTO "WITOS_FinOpsAlertEvent"', ({"id": "e1"},)),))
    store = AlertStore(executor)
    draft = AlertDraft(
        alert_id="a1",
        alert_type="forecast_exceeds_budget",
        severity="critical",
        scope=SCOPE,
        title="over budget",
        body={"projected_eom": "1"},
    )

    async def run() -> str:
        return await store.raise_event(draft, ({"type": "email", "url": ""},))

    import asyncio

    event_id = asyncio.run(run())
    written = json.loads(executor.queries[0][1][7])

    assert event_id == "e1"
    assert written[0]["delivered"] is False
    assert written[0]["detail"] == "unsupported_channel"


def test_a_webhook_is_signed_over_the_timestamp_and_the_body(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WITOS_WEBHOOK_SECRET_SOC", "s3cret")
    poster = _CapturingPoster(status=200)

    import asyncio

    result = asyncio.run(
        deliver(
            "https://example.invalid/hook",
            {"alert": "over budget"},
            secret_name="soc",
            event_id="e1",
            poster=poster,
        )
    )

    assert isinstance(result, Delivered)
    signature = poster.headers[SIGNATURE_HEADER]
    timestamp = int(poster.headers["X-WITOS-Timestamp"])
    assert verify(poster.body, "s3cret", timestamp=timestamp, signature=signature)
    assert not verify(poster.body, "wrong", timestamp=timestamp, signature=signature)
    assert not verify(poster.body, "s3cret", timestamp=timestamp + 1, signature=signature)


def test_a_missing_secret_refuses_to_send_rather_than_sending_unsigned() -> None:
    poster = _CapturingPoster(status=200)

    import asyncio

    result = asyncio.run(
        deliver("https://example.invalid/hook", {}, secret_name="absent", event_id="e1", poster=poster)
    )

    assert isinstance(result, DeliveryFailed)
    assert poster.calls == 0


def test_a_rejected_webhook_is_a_failure_not_an_exception() -> None:
    import asyncio

    result = asyncio.run(
        deliver(
            "https://example.invalid/hook", {}, secret_name=None, event_id="e1", poster=_CapturingPoster(status=500)
        )
    )

    assert isinstance(result, DeliveryFailed)
    assert "500" in result.reason


def test_signing_is_stable_for_the_same_input() -> None:
    assert sign(b"body", "k", timestamp=1) == sign(b"body", "k", timestamp=1)
    assert sign(b"body", "k", timestamp=1) != sign(b"body", "k", timestamp=2)


def _buckets(spends: tuple[float, ...], *, model: str = "gpt-4o-mini") -> ScopeBuckets:
    start = datetime(2026, 8, 20, tzinfo=timezone.utc)
    return ScopeBuckets(
        scope=SCOPE,
        buckets=tuple(
            {
                "scope_type": SCOPE.scope_type,
                "scope_id": SCOPE.scope_id,
                "bucket_start_utc": start + timedelta(hours=index),
                "spend_usd": Decimal(str(spend)),
                "request_count": 10,
                "breakdown": {model: {"spend": Decimal(str(spend)), "tokens": Decimal("1000")}},
            }
            for index, spend in enumerate(spends)
        ),  # pyright: ignore[reportArgumentType]  # hand-built rows of the queried shape
    )


class _CapturingPoster:
    def __init__(self, *, status: int) -> None:
        self.status = status
        self.calls = 0
        self.body = b""
        self.headers: dict[str, str] = {}

    async def post(self, url: str, *, content: bytes, headers, timeout: float) -> int:
        self.calls += 1
        self.body = content
        self.headers = dict(headers)
        return self.status
