"""Decision aggregation and fail-mode resolution (§2.4, §2.8).

Precedence is fixed and total: BLOCK > REQUIRE_APPROVAL > REDACT > MASK > WARN >
AUDIT > ALLOW. A policy's `priority` breaks ties **within** a rank, deciding
whose message and redaction strategy are reported. It deliberately cannot lift a
lower-ranked action over a higher-ranked one: letting an author set
`priority: 999` on an ALLOW to defeat someone else's BLOCK would turn a
numeric field into a policy bypass.

Shadow verdicts are partitioned out before any of this runs. They are visible in
`shadow_verdicts` and in `shadow_action` for the "would have blocked N requests"
review, and they are structurally incapable of reaching `action`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from litellm.proxy.witos.policy_fabric.evaluator import Finding, Truth
from litellm.proxy.witos.policy_fabric.types import (
    ACTION_PRECEDENCE,
    FailMode,
    FederationMode,
    PolicyAction,
    RedactStrategy,
    Severity,
)


@dataclass(frozen=True, slots=True)
class PolicyVerdict:
    policy_id: str
    policy_name: str
    policy_version: int
    action: PolicyAction
    severity: Severity
    priority: int
    mode: FederationMode
    shadow: bool
    findings: tuple[Finding, ...] = ()
    matched_rule_ids: tuple[str, ...] = ()
    provider: str | None = None
    external_policy_id: str | None = None
    block_message: str | None = None
    redact_strategy: RedactStrategy = RedactStrategy.MASK
    fail_mode_triggered: bool = False
    evaluation_latency_ms: int | None = None


@dataclass(frozen=True, slots=True)
class AggregateDecision:
    action: PolicyAction
    deciding: PolicyVerdict | None
    enforced_verdicts: tuple[PolicyVerdict, ...]
    shadow_verdicts: tuple[PolicyVerdict, ...]
    shadow_action: PolicyAction

    @property
    def findings(self) -> tuple[Finding, ...]:
        return tuple(finding for verdict in self.enforced_verdicts for finding in verdict.findings)

    @property
    def blocking_verdicts(self) -> tuple[PolicyVerdict, ...]:
        return tuple(verdict for verdict in self.enforced_verdicts if verdict.action is PolicyAction.BLOCK)


def aggregate(verdicts: Sequence[PolicyVerdict]) -> AggregateDecision:
    enforced: Final = tuple(verdict for verdict in verdicts if not verdict.shadow)
    shadowed: Final = tuple(verdict for verdict in verdicts if verdict.shadow)
    winner: Final = _winner(enforced)
    return AggregateDecision(
        action=PolicyAction.ALLOW if winner is None else winner.action,
        deciding=winner,
        enforced_verdicts=enforced,
        shadow_verdicts=shadowed,
        shadow_action=_shadow_action(shadowed),
    )


def _winner(verdicts: tuple[PolicyVerdict, ...]) -> PolicyVerdict | None:
    actionable: Final = tuple(verdict for verdict in verdicts if verdict.action is not PolicyAction.ALLOW)
    if not actionable:
        return None
    return max(actionable, key=_ranking_key)


def _shadow_action(shadowed: tuple[PolicyVerdict, ...]) -> PolicyAction:
    winner: Final = _winner(tuple(_as_enforced(verdict) for verdict in shadowed))
    return PolicyAction.ALLOW if winner is None else winner.action


def _as_enforced(verdict: PolicyVerdict) -> PolicyVerdict:
    """A shadow verdict viewed as if it had been enforced, for analytics only."""
    return PolicyVerdict(
        policy_id=verdict.policy_id,
        policy_name=verdict.policy_name,
        policy_version=verdict.policy_version,
        action=verdict.action,
        severity=verdict.severity,
        priority=verdict.priority,
        mode=verdict.mode,
        shadow=False,
        findings=verdict.findings,
        matched_rule_ids=verdict.matched_rule_ids,
    )


def _ranking_key(verdict: PolicyVerdict) -> tuple[int, int, str]:
    return (ACTION_PRECEDENCE[verdict.action], verdict.priority, verdict.policy_id)


def resolve_indeterminate(
    truth: Truth,
    on_match: PolicyAction,
    fail_mode: FailMode,
) -> tuple[PolicyAction | None, bool]:
    """Map an evaluation result to an action, applying the fail mode.

    Returns the action to record (None when the policy simply did not match) and
    whether the fail mode was triggered. Fail mode is always alerted on and
    always recorded; a guardrail that silently fails open is indistinguishable
    from one that is switched off.
    """
    match truth:
        case Truth.TRUE:
            return on_match, False
        case Truth.FALSE:
            return None, False
        case Truth.INDETERMINATE:
            match fail_mode:
                case FailMode.FAIL_OPEN:
                    return None, True
                case FailMode.FAIL_CLOSED:
                    return on_match, True
                case FailMode.OBSERVE:
                    return PolicyAction.AUDIT, True
