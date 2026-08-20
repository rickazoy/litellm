"""Scope resolution and tenant separation (§2.8, §2.14).

A policy is applicable to a request only if it survives four gates: tenant,
direction, declared scope, and exceptions. The tenant gate is first and is not
optional. Every DB query in this feature filters on `organization_id` for the
same reason: a DLP receipt from one customer surfacing in another customer's
console is a worse incident than the one the policy was written to stop.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from litellm.proxy.witos.policy_fabric.canonical import WitDpsPolicy
from litellm.proxy.witos.policy_fabric.evaluator import RequestScope, glob_match
from litellm.proxy.witos.policy_fabric.types import (
    EvaluationDirection,
    PolicyStatus,
    policy_covers_direction,
)

ENFORCING_STATUSES: Final[frozenset[PolicyStatus]] = frozenset({PolicyStatus.ACTIVE, PolicyStatus.STALE})
EVALUATING_STATUSES: Final[frozenset[PolicyStatus]] = frozenset(
    {PolicyStatus.ACTIVE, PolicyStatus.STALE, PolicyStatus.SHADOW}
)


@dataclass(frozen=True, slots=True)
class ScopedPolicy:
    """A policy plus the runtime facts the guardrail needs about it."""

    policy: WitDpsPolicy
    policy_id: str
    version: int
    status: PolicyStatus
    organization_id: str | None
    connection_id: str | None

    @property
    def shadow(self) -> bool:
        """Shadow policies evaluate on live traffic and never enforce (§2.7)."""
        return self.status is PolicyStatus.SHADOW


def applicable_policies(
    policies: Sequence[ScopedPolicy],
    scope: RequestScope,
    direction: EvaluationDirection,
) -> tuple[ScopedPolicy, ...]:
    return tuple(
        scoped
        for scoped in policies
        if scoped.status in EVALUATING_STATUSES
        and _tenant_matches(scoped, scope)
        and policy_covers_direction(scoped.policy.direction, direction)
        and _scope_matches(scoped.policy, scope)
        and not _excepted(scoped.policy, scope)
    )


def _tenant_matches(scoped: ScopedPolicy, scope: RequestScope) -> bool:
    """A policy with no organization is a proxy-wide policy; a scoped one is
    invisible outside its tenant, whatever its condition says."""
    if scoped.organization_id is None:
        return True
    return scoped.organization_id == scope.organization_id


def _scope_matches(policy: WitDpsPolicy, scope: RequestScope) -> bool:
    return (
        _entities_match(policy, scope)
        and _any_glob(policy.scope.models, scope.model, scope.model_group)
        and _any_glob(policy.scope.applications, scope.application)
    )


def _entities_match(policy: WitDpsPolicy, scope: RequestScope) -> bool:
    if not policy.scope.entities:
        return True
    return any(_entity_matches(entity.entity_type, entity.entity_id, scope) for entity in policy.scope.entities)


def _entity_matches(entity_type: str, entity_id: str, scope: RequestScope) -> bool:
    match entity_type:
        case "organization":
            return glob_match(entity_id, scope.organization_id)
        case "team":
            return glob_match(entity_id, scope.team_id)
        case "user":
            return glob_match(entity_id, scope.user_id)
        case "key_alias":
            return glob_match(entity_id, scope.key_alias)
        case "end_user":
            return glob_match(entity_id, scope.end_user_id)
        case _:
            return False


def _any_glob(patterns: tuple[str, ...], *values: str | None) -> bool:
    if not patterns:
        return True
    return any(glob_match(pattern, value) for pattern in patterns for value in values)


def _excepted(policy: WitDpsPolicy, scope: RequestScope) -> bool:
    return any(_exception_matches(exception.exception_type, exception.value, scope) for exception in policy.exceptions)


def _exception_matches(exception_type: str, value: str, scope: RequestScope) -> bool:
    match exception_type:
        case "identity":
            return glob_match(value, scope.user_id)
        case "group":
            return any(glob_match(value, group) for group in scope.groups)
        case "team":
            return glob_match(value, scope.team_id)
        case "organization":
            return glob_match(value, scope.organization_id)
        case "application":
            return glob_match(value, scope.application)
        case "key_alias":
            return glob_match(value, scope.key_alias)
        case "model":
            return glob_match(value, scope.model)
        case _:
            return False
