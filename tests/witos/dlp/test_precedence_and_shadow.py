"""Action precedence and shadow isolation (§2.4, §2.7, §2.14)."""

from __future__ import annotations

from litellm.proxy.witos.policy_fabric.aggregator import PolicyVerdict, aggregate, resolve_indeterminate
from litellm.proxy.witos.policy_fabric.evaluator import Truth
from litellm.proxy.witos.policy_fabric.types import (
    ACTION_PRECEDENCE,
    FailMode,
    FederationMode,
    PolicyAction,
    Severity,
)


def verdict(
    policy_id: str,
    action: PolicyAction,
    priority: int = 100,
    shadow: bool = False,
) -> PolicyVerdict:
    return PolicyVerdict(
        policy_id=policy_id,
        policy_name=f"policy-{policy_id}",
        policy_version=1,
        action=action,
        severity=Severity.HIGH,
        priority=priority,
        mode=FederationMode.MIRROR,
        shadow=shadow,
    )


def test_precedence_order_matches_the_specification_exactly() -> None:
    ordered = sorted(ACTION_PRECEDENCE, key=lambda action: ACTION_PRECEDENCE[action], reverse=True)
    assert ordered == [
        PolicyAction.BLOCK,
        PolicyAction.REQUIRE_APPROVAL,
        PolicyAction.REDACT,
        PolicyAction.MASK,
        PolicyAction.WARN,
        PolicyAction.AUDIT,
        PolicyAction.ALLOW,
    ]


def test_the_strongest_action_wins_across_simultaneous_matches() -> None:
    decision = aggregate(
        (
            verdict("a", PolicyAction.AUDIT),
            verdict("b", PolicyAction.REDACT),
            verdict("c", PolicyAction.BLOCK),
            verdict("d", PolicyAction.WARN),
        )
    )
    assert decision.action is PolicyAction.BLOCK
    assert decision.deciding is not None and decision.deciding.policy_id == "c"


def test_multiple_simultaneous_block_policies_all_appear_and_priority_picks_the_reporter() -> None:
    decision = aggregate(
        (
            verdict("low", PolicyAction.BLOCK, priority=10),
            verdict("high", PolicyAction.BLOCK, priority=900),
            verdict("mid", PolicyAction.BLOCK, priority=500),
        )
    )
    assert decision.action is PolicyAction.BLOCK
    assert decision.deciding is not None and decision.deciding.policy_id == "high"
    assert {verdict_.policy_id for verdict_ in decision.blocking_verdicts} == {"low", "high", "mid"}


def test_priority_cannot_lift_a_weaker_action_over_a_stronger_one() -> None:
    """A numeric field must never become a policy bypass."""
    decision = aggregate(
        (
            verdict("allow_all", PolicyAction.ALLOW, priority=100000),
            verdict("audit_all", PolicyAction.AUDIT, priority=99999),
            verdict("blocker", PolicyAction.BLOCK, priority=1),
        )
    )
    assert decision.action is PolicyAction.BLOCK


def test_ties_resolve_deterministically_by_policy_id() -> None:
    first = aggregate((verdict("zzz", PolicyAction.MASK), verdict("aaa", PolicyAction.MASK)))
    second = aggregate((verdict("aaa", PolicyAction.MASK), verdict("zzz", PolicyAction.MASK)))
    assert first.deciding is not None and second.deciding is not None
    assert first.deciding.policy_id == second.deciding.policy_id == "zzz"


def test_a_shadow_block_can_never_become_the_enforced_action() -> None:
    decision = aggregate((verdict("shadow_blocker", PolicyAction.BLOCK, shadow=True),))
    assert decision.action is PolicyAction.ALLOW
    assert decision.deciding is None
    assert decision.enforced_verdicts == ()
    assert len(decision.shadow_verdicts) == 1


def test_shadow_analytics_still_report_what_would_have_happened() -> None:
    decision = aggregate(
        (
            verdict("audit", PolicyAction.AUDIT),
            verdict("shadow_blocker", PolicyAction.BLOCK, shadow=True),
        )
    )
    assert decision.action is PolicyAction.AUDIT
    assert decision.shadow_action is PolicyAction.BLOCK


def test_a_shadow_policy_never_contributes_findings_to_enforcement() -> None:
    decision = aggregate((verdict("s", PolicyAction.REDACT, shadow=True),))
    assert decision.findings == ()


def test_fail_open_treats_an_unreachable_vendor_as_no_match_but_flags_it() -> None:
    action, triggered = resolve_indeterminate(Truth.INDETERMINATE, PolicyAction.BLOCK, FailMode.FAIL_OPEN)
    assert action is None
    assert triggered is True


def test_fail_closed_applies_the_policy_action_when_the_vendor_is_unreachable() -> None:
    action, triggered = resolve_indeterminate(Truth.INDETERMINATE, PolicyAction.BLOCK, FailMode.FAIL_CLOSED)
    assert action is PolicyAction.BLOCK
    assert triggered is True


def test_observe_fail_mode_downgrades_to_audit() -> None:
    action, triggered = resolve_indeterminate(Truth.INDETERMINATE, PolicyAction.BLOCK, FailMode.OBSERVE)
    assert action is PolicyAction.AUDIT
    assert triggered is True


def test_a_definite_result_never_triggers_the_fail_mode() -> None:
    assert resolve_indeterminate(Truth.TRUE, PolicyAction.MASK, FailMode.FAIL_CLOSED) == (PolicyAction.MASK, False)
    assert resolve_indeterminate(Truth.FALSE, PolicyAction.MASK, FailMode.FAIL_CLOSED) == (None, False)
