"""History assembly and the §1.6 short-history ladder."""

from datetime import date, timedelta
from decimal import Decimal

from litellm.proxy.witos.finops.history import (
    ScopeRef,
    build_history,
    classify_history,
    tier_confidence,
)

from tests.witos.finops.fakes import day_usage, flat_history


def test_missing_days_are_zero_filled_not_dropped() -> None:
    """A day with no calls is a zero, and a forecaster that loses it learns a six day week."""
    observed = (day_usage(date(2026, 5, 1)), day_usage(date(2026, 5, 5)))
    history = build_history(ScopeRef("team", "t1"), observed, as_of=date(2026, 5, 5))

    assert history is not None
    assert history.history_days == 5
    assert tuple(day.day for day in history.days) == tuple(
        date(2026, 5, 1) + timedelta(days=offset) for offset in range(5)
    )
    assert history.usage_series("prompt_tokens")[1:4] == (0.0, 0.0, 0.0)
    assert history.active_days == 2


def test_history_extends_to_as_of_so_a_stopped_scope_reads_as_zero() -> None:
    observed = (day_usage(date(2026, 5, 1)),)
    history = build_history(ScopeRef("team", "t1"), observed, as_of=date(2026, 5, 10))

    assert history is not None
    assert history.history_days == 10
    assert history.end_day == date(2026, 5, 10)
    assert history.spend_series()[-1] == 0.0


def test_ladder_boundaries_are_the_contract() -> None:
    """3, 7, 28 and 90 days each change what the engine is allowed to do (§1.6)."""
    assert classify_history(0) == "insufficient_history"
    assert classify_history(2) == "insufficient_history"
    assert classify_history(3) == "burn_rate"
    assert classify_history(6) == "burn_rate"
    assert classify_history(7) == "trend_weekday"
    assert classify_history(27) == "trend_weekday"
    assert classify_history(28) == "seasonal_trend"
    assert classify_history(89) == "seasonal_trend"
    assert classify_history(90) == "full_ensemble"
    assert classify_history(400) == "full_ensemble"


def test_confidence_never_claims_more_than_the_rung_supports() -> None:
    assert tier_confidence("insufficient_history") == "none"
    assert tier_confidence("burn_rate") == "low"
    assert tier_confidence("full_ensemble") == "high"


def test_dormancy_is_measured_from_the_last_active_day() -> None:
    quiet = build_history(ScopeRef("key", "k1"), (day_usage(date(2026, 5, 1)),), as_of=date(2026, 7, 1))
    busy = flat_history(40)

    assert quiet is not None
    assert quiet.is_dormant() is True
    assert busy.is_dormant() is False


def test_mix_totals_only_cover_the_requested_window() -> None:
    observed = (
        day_usage(date(2026, 5, 1), mix={"old-model": 10.0}),
        day_usage(date(2026, 5, 2), mix={"new-model": 30.0}),
    )
    history = build_history(ScopeRef("team", "t1"), observed, as_of=date(2026, 5, 2))

    assert history is not None
    assert dict(history.mix_tokens_total(window_days=1)) == {"new-model": 30.0}
    assert dict(history.mix_tokens_total(window_days=2)) == {"new-model": 30.0, "old-model": 10.0}


def test_total_spend_stays_decimal() -> None:
    history = flat_history(3, spend_per_day="0.1")

    assert history.total_spend() == Decimal("0.3")
