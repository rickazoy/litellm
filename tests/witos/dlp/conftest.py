"""Shared fixtures for the DLP policy fabric suite.

Everything here is in-memory. No network, no database, no proxy: the point of
putting the engine behind injectable dependencies is that its security
properties can be asserted directly rather than inferred from an integration
run.
"""

from __future__ import annotations

from typing import Any, Final, Mapping

import pytest

from litellm.proxy.witos.policy_fabric.canonical import PolicyValidationFailure, parse_policy
from litellm.proxy.witos.policy_fabric.circuit_breaker import CircuitBreaker
from litellm.proxy.witos.policy_fabric.evaluator import DelegatedEvaluator, PresidioAnalyzer
from litellm.proxy.witos.policy_fabric.policy_cache import PolicySetCache
from litellm.proxy.witos.policy_fabric.runtime import (
    CompiledPolicyStore,
    EngineConfig,
    PolicyFabricEngine,
)
from litellm.proxy.witos.policy_fabric.scope import ScopedPolicy
from litellm.proxy.witos.policy_fabric.types import PolicyStatus

SSN_PATTERN: Final = r"\d{3}-\d{2}-\d{4}"
SSN_VALUE: Final = "123-45-6789"


def policy_document(
    name: str,
    on_match: str = "BLOCK",
    condition: Mapping[str, Any] | None = None,
    priority: int = 100,
    direction: str = "both",
    severity: str = "high",
    mode: str = "mirror",
    redact_strategy: str | None = None,
    fail_mode: str | None = None,
    scope: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "wit_dps_version": "2.0",
        "name": name,
        "mode": mode,
        "severity": severity,
        "priority": priority,
        "direction": direction,
        "condition": condition or {"type": "regex", "pattern": SSN_PATTERN, "classifier": "PII.SSN"},
        "actions": {"on_match": on_match},
    }
    if redact_strategy is not None:
        document["actions"]["redact_strategy"] = redact_strategy
    if fail_mode is not None:
        document["fail_mode"] = fail_mode
    if scope is not None:
        document["scope"] = scope
    return document


def scoped(
    document: Mapping[str, Any],
    policy_id: str = "p1",
    status: PolicyStatus = PolicyStatus.ACTIVE,
    organization_id: str | None = None,
    connection_id: str | None = None,
    version: int = 1,
) -> ScopedPolicy:
    parsed = parse_policy(document)
    assert not isinstance(parsed, PolicyValidationFailure), parsed
    return ScopedPolicy(
        policy=parsed,
        policy_id=policy_id,
        version=version,
        status=status,
        organization_id=organization_id,
        connection_id=connection_id,
    )


def engine_over(
    policies: tuple[ScopedPolicy, ...],
    config: EngineConfig | None = None,
    presidio: PresidioAnalyzer | None = None,
    delegate: DelegatedEvaluator | None = None,
    breaker: CircuitBreaker | None = None,
) -> PolicyFabricEngine:
    async def loader(organization_id: str | None) -> tuple[ScopedPolicy, ...]:
        return tuple(
            policy
            for policy in policies
            if policy.organization_id is None or policy.organization_id == organization_id
        )

    return PolicyFabricEngine(
        cache=PolicySetCache(loader=loader, ttl_seconds=0.0),
        store=CompiledPolicyStore(),
        config=config if config is not None else EngineConfig(),
        presidio=presidio,
        delegate=delegate,
        breaker=breaker,
    )


@pytest.fixture
def ssn_text() -> str:
    return f"the ssn is {SSN_VALUE} and that is that"
