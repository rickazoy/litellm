"""Engine behaviour: shadow isolation end to end, tenant separation, fail modes,
the circuit breaker, lazy evaluation and the tool direction (§2.8, §2.10, §2.14).
"""

from __future__ import annotations

import time
from typing import Final

from conftest import SSN_VALUE, engine_over, policy_document, scoped

from litellm.proxy.witos.policy_fabric.circuit_breaker import BreakerConfig, BreakerState, CircuitBreaker
from litellm.proxy.witos.policy_fabric.condition_ast import ClassifierLeaf
from litellm.proxy.witos.policy_fabric.evaluator import (
    EvaluationContext,
    Finding,
    FindingSource,
    LeafOutcome,
    RequestScope,
    Truth,
)
from litellm.proxy.witos.policy_fabric.runtime import EngineConfig, EvaluationRequest
from litellm.proxy.witos.policy_fabric.types import (
    EnforcementKind,
    EvaluationDirection,
    FailMode,
    PolicyAction,
    PolicyStatus,
)

DELEGATED_CLASS: Final = "CUSTOM.cyera.LEARNED_CONTRACT"


class CountingDelegate:
    def __init__(self, truth: Truth = Truth.TRUE) -> None:
        self.calls = 0
        self._truth = truth

    async def evaluate_leaf(self, leaf: ClassifierLeaf, context: EvaluationContext) -> LeafOutcome:
        del context
        self.calls += 1
        return LeafOutcome(
            truth=self._truth,
            findings=(
                (Finding(classifier=leaf.value, start=0, end=1, confidence=1.0, source=FindingSource.VENDOR),)
                if self._truth is Truth.TRUE
                else ()
            ),
        )


class FailingDelegate:
    def __init__(self) -> None:
        self.calls = 0

    async def evaluate_leaf(self, leaf: ClassifierLeaf, context: EvaluationContext) -> LeafOutcome:
        del leaf, context
        self.calls += 1
        raise TimeoutError("vendor unreachable")


class StubPresidio:
    def __init__(self, findings: tuple[Finding, ...]) -> None:
        self.findings = findings
        self.calls = 0

    async def analyze(self, text: str, entities: tuple[str, ...]) -> tuple[Finding, ...]:
        del text, entities
        self.calls += 1
        return self.findings


def request(content: str, organization_id: str | None = None, **scope_kwargs) -> EvaluationRequest:
    return EvaluationRequest(
        request_id="req-1",
        content=content,
        direction=scope_kwargs.pop("direction", EvaluationDirection.INPUT),
        scope=RequestScope(organization_id=organization_id, **scope_kwargs),
    )


async def test_an_active_block_policy_blocks(ssn_text: str) -> None:
    engine = engine_over((scoped(policy_document("PCI")),))
    result = await engine.evaluate(request(ssn_text))
    assert result.decision.action is PolicyAction.BLOCK
    assert result.should_block


async def test_a_shadow_policy_evaluates_but_cannot_block(ssn_text: str) -> None:
    engine = engine_over((scoped(policy_document("PCI"), status=PolicyStatus.SHADOW),))
    result = await engine.evaluate(request(ssn_text))
    assert result.decision.action is PolicyAction.ALLOW
    assert not result.should_block
    assert not result.should_rewrite
    assert result.decision.shadow_action is PolicyAction.BLOCK
    assert len(result.receipts) == 1
    assert result.receipts[0].shadow is True
    assert result.receipts[0].prevented is False


async def test_a_disabled_policy_is_not_even_evaluated(ssn_text: str) -> None:
    engine = engine_over((scoped(policy_document("PCI"), status=PolicyStatus.DISABLED),))
    result = await engine.evaluate(request(ssn_text))
    assert result.receipts == ()
    assert result.decision.action is PolicyAction.ALLOW


async def test_redaction_rewrites_content_and_the_receipt_records_it(ssn_text: str) -> None:
    engine = engine_over((scoped(policy_document("PII", on_match="REDACT", redact_strategy="mask")),))
    result = await engine.evaluate(request(ssn_text))
    assert result.should_rewrite
    assert result.rewritten_content is not None
    assert SSN_VALUE not in result.rewritten_content
    assert result.receipts[0].redaction_performed is True


async def test_a_shadow_redact_policy_never_rewrites(ssn_text: str) -> None:
    engine = engine_over(
        (scoped(policy_document("PII", on_match="REDACT"), status=PolicyStatus.SHADOW),)
    )
    result = await engine.evaluate(request(ssn_text))
    assert result.rewritten_content is None
    assert result.receipts[0].redaction_performed is False


async def test_a_policy_from_another_tenant_is_never_loaded(ssn_text: str) -> None:
    engine = engine_over((scoped(policy_document("PCI"), organization_id="org-a"),))
    theirs = await engine.evaluate(request(ssn_text, organization_id="org-a"))
    mine = await engine.evaluate(request(ssn_text, organization_id="org-b"))
    assert theirs.should_block
    assert not mine.should_block
    assert mine.receipts == ()


async def test_a_proxy_wide_policy_applies_to_every_tenant(ssn_text: str) -> None:
    engine = engine_over((scoped(policy_document("PCI"), organization_id=None),))
    assert (await engine.evaluate(request(ssn_text, organization_id="org-a"))).should_block
    assert (await engine.evaluate(request(ssn_text, organization_id="org-b"))).should_block


async def test_scope_entities_restrict_which_requests_a_policy_sees(ssn_text: str) -> None:
    document = policy_document("PCI", scope={"entities": [{"type": "team", "id": "payments"}]})
    engine = engine_over((scoped(document),))
    assert (await engine.evaluate(request(ssn_text, team_id="payments"))).should_block
    assert not (await engine.evaluate(request(ssn_text, team_id="marketing"))).should_block


async def test_delegated_leaves_fail_open_when_no_vendor_is_wired(ssn_text: str) -> None:
    document = policy_document("learned", condition={"type": "data_class", "class": DELEGATED_CLASS})
    engine = engine_over((scoped(document),), config=EngineConfig(default_fail_mode=FailMode.FAIL_OPEN))
    result = await engine.evaluate(request(ssn_text))
    assert result.decision.action is PolicyAction.ALLOW
    assert result.receipts == ()


async def test_delegated_leaves_fail_closed_when_configured(ssn_text: str) -> None:
    document = policy_document(
        "learned", condition={"type": "data_class", "class": DELEGATED_CLASS}, fail_mode="fail_closed"
    )
    engine = engine_over((scoped(document),))
    result = await engine.evaluate(request(ssn_text))
    assert result.decision.action is PolicyAction.BLOCK
    assert result.receipts[0].fail_mode_triggered is True


async def test_a_critical_block_policy_defaults_to_fail_closed(ssn_text: str) -> None:
    document = policy_document(
        "learned", severity="critical", condition={"type": "data_class", "class": DELEGATED_CLASS}
    )
    engine = engine_over((scoped(document),))
    result = await engine.evaluate(request(ssn_text))
    assert result.decision.action is PolicyAction.BLOCK
    assert result.receipts[0].fail_mode_triggered is True


async def test_a_reachable_vendor_resolves_the_delegated_leaf(ssn_text: str) -> None:
    document = policy_document("learned", condition={"type": "data_class", "class": DELEGATED_CLASS})
    delegate = CountingDelegate()
    engine = engine_over((scoped(document, connection_id="conn-1"),), delegate=delegate)
    result = await engine.evaluate(request(ssn_text))
    assert delegate.calls == 1
    assert result.decision.action is PolicyAction.BLOCK


async def test_the_circuit_breaker_trips_and_stops_calling_a_failing_vendor(ssn_text: str) -> None:
    document = policy_document(
        "learned", condition={"type": "data_class", "class": DELEGATED_CLASS}, fail_mode="fail_open"
    )
    ticks = iter(range(1, 1000))
    breaker = CircuitBreaker(
        config=BreakerConfig(failure_threshold=5, window_seconds=30.0, cooldown_seconds=30.0),
        clock=lambda: float(next(ticks)),
    )
    delegate = FailingDelegate()
    engine = engine_over((scoped(document, connection_id="conn-1"),), delegate=delegate, breaker=breaker)

    for _ in range(5):
        await engine.evaluate(request(ssn_text))
    assert delegate.calls == 5
    assert breaker.state("conn-1") is BreakerState.OPEN

    for _ in range(10):
        await engine.evaluate(request(ssn_text))
    assert delegate.calls == 5, "a tripped breaker must not keep calling the vendor"


async def test_a_tripped_breaker_half_opens_after_the_cooldown() -> None:
    now = [0.0]
    breaker = CircuitBreaker(
        config=BreakerConfig(failure_threshold=2, window_seconds=30.0, cooldown_seconds=10.0),
        clock=lambda: now[0],
    )
    breaker.record_failure("conn-1")
    breaker.record_failure("conn-1")
    assert breaker.state("conn-1") is BreakerState.OPEN
    now[0] = 11.0
    assert breaker.state("conn-1") is BreakerState.HALF_OPEN
    breaker.record_failure("conn-1")
    assert breaker.state("conn-1") is BreakerState.OPEN


async def test_a_success_closes_the_breaker() -> None:
    breaker = CircuitBreaker(config=BreakerConfig(failure_threshold=2), clock=lambda: 0.0)
    breaker.record_failure("conn-1")
    breaker.record_success("conn-1")
    breaker.record_failure("conn-1")
    assert breaker.state("conn-1") is BreakerState.CLOSED


async def test_lazy_evaluation_skips_the_vendor_when_a_cheap_leaf_already_settles_it(ssn_text: str) -> None:
    """The context leaf is false, so the ALL can never match; no vendor call."""
    document = policy_document(
        "hybrid",
        condition={
            "operator": "ALL",
            "conditions": [
                {"type": "data_class", "class": DELEGATED_CLASS},
                {"type": "team", "value": "payments"},
            ],
        },
    )
    delegate = CountingDelegate()
    engine = engine_over((scoped(document, connection_id="conn-1"),), delegate=delegate)
    result = await engine.evaluate(request(ssn_text, team_id="marketing"))
    assert delegate.calls == 0
    assert result.decision.action is PolicyAction.ALLOW


async def test_the_vendor_is_called_when_the_condition_can_still_flip(ssn_text: str) -> None:
    document = policy_document(
        "hybrid",
        condition={
            "operator": "ALL",
            "conditions": [
                {"type": "data_class", "class": DELEGATED_CLASS},
                {"type": "team", "value": "payments"},
            ],
        },
    )
    delegate = CountingDelegate()
    engine = engine_over((scoped(document, connection_id="conn-1"),), delegate=delegate)
    result = await engine.evaluate(request(ssn_text, team_id="payments"))
    assert delegate.calls == 1
    assert result.decision.action is PolicyAction.BLOCK


async def test_a_data_class_leaf_that_presidio_can_answer_never_reaches_a_vendor() -> None:
    document = policy_document("pii", condition={"type": "data_class", "class": "PII.SSN"})
    presidio = StubPresidio(
        (Finding(classifier="PII.SSN", start=11, end=22, confidence=0.99, source=FindingSource.PRESIDIO),)
    )
    delegate = CountingDelegate()
    engine = engine_over((scoped(document, connection_id="conn-1"),), presidio=presidio, delegate=delegate)
    result = await engine.evaluate(request("the ssn is 123-45-6789 ok"))
    assert presidio.calls == 1
    assert delegate.calls == 0
    assert result.decision.action is PolicyAction.BLOCK


async def test_presidio_findings_below_min_confidence_do_not_match() -> None:
    document = policy_document(
        "pii", condition={"type": "data_class", "class": "PII.SSN", "min_confidence": 0.95}
    )
    presidio = StubPresidio(
        (Finding(classifier="PII.SSN", start=0, end=4, confidence=0.4, source=FindingSource.PRESIDIO),)
    )
    engine = engine_over((scoped(document),), presidio=presidio)
    result = await engine.evaluate(request("1234"))
    assert result.decision.action is PolicyAction.ALLOW


async def test_exceptions_exempt_a_scope_from_an_otherwise_matching_policy(ssn_text: str) -> None:
    document = policy_document("PCI")
    document["exceptions"] = [{"type": "key_alias", "value": "pci-approved-service"}]
    engine = engine_over((scoped(document),))
    assert not (await engine.evaluate(request(ssn_text, key_alias="pci-approved-service"))).should_block
    assert (await engine.evaluate(request(ssn_text, key_alias="some-other-key"))).should_block


async def test_a_policy_scoped_to_input_ignores_output(ssn_text: str) -> None:
    engine = engine_over((scoped(policy_document("PCI", direction="input")),))
    inbound = await engine.evaluate(request(ssn_text, direction=EvaluationDirection.INPUT))
    outbound = await engine.evaluate(request(ssn_text, direction=EvaluationDirection.OUTPUT))
    assert inbound.should_block
    assert not outbound.should_block
    assert outbound.receipts == ()


async def test_tool_direction_evaluates_and_records_but_cannot_enforce_in_v1(ssn_text: str) -> None:
    """§2.10: the interface ships in v1, enforcement on tool payloads does not."""
    engine = engine_over((scoped(policy_document("PCI", direction="tool")),))
    result = await engine.evaluate(
        EvaluationRequest(
            request_id="req-tool",
            content=ssn_text,
            direction=EvaluationDirection.TOOL_INPUT,
            scope=RequestScope(),
            tool_name="salesforce_query",
            tool_arguments={"soql": "SELECT ssn FROM Contact"},
        )
    )
    assert result.decision.action is PolicyAction.BLOCK
    assert result.enforcement_active is False
    assert result.should_block is False
    assert result.enforcement_kind is EnforcementKind.DETECTION
    assert result.receipts[0].direction is EvaluationDirection.TOOL_INPUT
    assert result.receipts[0].prevented is False


async def test_a_tool_argument_leaf_matches_on_the_named_argument() -> None:
    document = policy_document(
        "tool-args",
        direction="tool_input",
        condition={"type": "tool_argument", "tool": "salesforce_*", "argument": "soql", "value": "*ssn*"},
    )
    engine = engine_over((scoped(document),))

    def tool_request(arguments: dict[str, str], tool: str = "salesforce_query") -> EvaluationRequest:
        return EvaluationRequest(
            request_id="req-tool",
            content="",
            direction=EvaluationDirection.TOOL_INPUT,
            scope=RequestScope(),
            tool_name=tool,
            tool_arguments=arguments,
        )

    matched = await engine.evaluate(tool_request({"soql": "SELECT ssn FROM Contact"}))
    unmatched = await engine.evaluate(tool_request({"soql": "SELECT name FROM Contact"}))
    wrong_tool = await engine.evaluate(tool_request({"soql": "SELECT ssn"}, tool="jira_search"))
    assert matched.decision.action is PolicyAction.BLOCK
    assert unmatched.decision.action is PolicyAction.ALLOW
    assert wrong_tool.decision.action is PolicyAction.ALLOW


async def test_rag_context_direction_is_addressable(ssn_text: str) -> None:
    engine = engine_over((scoped(policy_document("PCI", direction="rag_context")),))
    result = await engine.evaluate(
        EvaluationRequest(
            request_id="req-rag",
            content=ssn_text,
            direction=EvaluationDirection.RAG_CONTEXT,
            scope=RequestScope(),
        )
    )
    assert result.decision.action is PolicyAction.BLOCK
    assert result.enforcement_active is False


async def test_a_policy_that_cannot_compile_is_reported_and_never_enforces(ssn_text: str) -> None:
    document = policy_document("broken", condition={"type": "regex", "pattern": r"(a+)+\1"})
    engine = engine_over((scoped(document),))
    result = await engine.evaluate(request(ssn_text))
    assert result.decision.action is PolicyAction.ALLOW
    assert len(result.compile_failures) == 1
    assert "re2 rejected" in result.compile_failures[0]


async def test_local_only_evaluation_stays_inside_the_five_millisecond_budget() -> None:
    """§2.8 NFR: local-only policies add at most 5 ms p95 to the hot path.

    The threshold is the blueprint's, not a tuned number. Measured p95 on a
    developer machine is roughly two orders of magnitude below it, so this guards
    against an algorithmic regression (a quadratic merge, a per-request recompile)
    rather than against ordinary machine noise.
    """
    policies = tuple(
        scoped(policy_document(f"p{index}", on_match="AUDIT"), policy_id=f"p{index}") for index in range(10)
    )
    engine = engine_over(policies)
    content = "lorem ipsum dolor sit amet " * 60 + "nothing sensitive"
    probe = request(content)

    for _ in range(20):
        await engine.evaluate(probe)

    samples = []
    for _ in range(200):
        started = time.perf_counter()
        await engine.evaluate(probe)
        samples.append((time.perf_counter() - started) * 1000)
    samples.sort()
    p95 = samples[int(len(samples) * 0.95)]
    assert p95 < 5.0, f"local-only evaluation p95 was {p95:.3f}ms"
