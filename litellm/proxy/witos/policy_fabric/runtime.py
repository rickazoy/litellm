"""The evaluation engine behind the `wit_dlp` guardrail (§2.8, §2.9, §2.10).

Everything the guardrail does per request lives here so it can be exercised
without a proxy, a database or a network. The guardrail itself is a thin
adapter between LiteLLM's hook signatures and this engine.

Two invariants are enforced structurally rather than by convention.

Shadow policies cannot enforce. Their verdicts are tagged at construction and
`aggregate` partitions them out before choosing an action, so there is no code
path in which a shadow policy changes a response.

Non-prompt directions cannot enforce in v1. `tool_input`, `tool_output` and
`rag_context` evaluate and write receipts, and `enforcement_active` comes back
False, so the interface is real while the enforcement ships in v1.5.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from litellm.proxy.witos.policy_fabric.aggregator import (
    AggregateDecision,
    PolicyVerdict,
    aggregate,
    resolve_indeterminate,
)
from litellm.proxy.witos.policy_fabric.circuit_breaker import CircuitBreaker
from litellm.proxy.witos.policy_fabric.compiler import (
    CompiledPolicy,
    CompileFailure,
    Re2Module,
    compile_policy,
)
from litellm.proxy.witos.policy_fabric.condition_ast import ClassifierLeaf
from litellm.proxy.witos.policy_fabric.evaluator import (
    DelegatedEvaluator,
    EvaluationContext,
    Finding,
    LeafOutcome,
    PolicyLeafResolver,
    PresidioAnalyzer,
    RequestScope,
    Truth,
    evaluate,
)
from litellm.proxy.witos.policy_fabric.metrics import DLPMetrics
from litellm.proxy.witos.policy_fabric.policy_cache import PolicySetCache
from litellm.proxy.witos.policy_fabric.receipts import (
    DecisionReceipt,
    ReceiptScope,
    correlation_hash,
    summarise_findings,
)
from litellm.proxy.witos.policy_fabric.redaction import apply_redaction
from litellm.proxy.witos.policy_fabric.scope import ScopedPolicy, applicable_policies
from litellm.proxy.witos.policy_fabric.streaming import StreamState, enforcement_for
from litellm.proxy.witos.policy_fabric.types import (
    V1_ENFORCED_DIRECTIONS,
    EnforcementKind,
    EvaluationDirection,
    FailMode,
    PolicyAction,
    StreamingMode,
)

_NO_TOOL_ARGUMENTS: Final[Mapping[str, str]] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class EngineConfig:
    guardrail_name: str = "wit_dlp"
    default_fail_mode: FailMode = FailMode.FAIL_OPEN
    critical_fail_mode: FailMode = FailMode.FAIL_CLOSED
    streaming_mode: StreamingMode = StreamingMode.CHUNK_GATE
    match_hash_salt: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    request_id: str
    content: str
    direction: EvaluationDirection
    scope: RequestScope
    tool_name: str | None = None
    tool_arguments: Mapping[str, str] = _NO_TOOL_ARGUMENTS
    stream_state: StreamState | None = None
    # §2.9 is per scope and policy set, so the mode travels with the request
    # rather than being a property of the engine that evaluates it.
    streaming_mode: StreamingMode | None = None


@dataclass(frozen=True, slots=True)
class EngineResult:
    decision: AggregateDecision
    receipts: tuple[DecisionReceipt, ...]
    rewritten_content: str | None
    enforcement_active: bool
    enforcement_kind: EnforcementKind
    evaluation_latency_ms: int
    compile_failures: tuple[str, ...]

    @property
    def should_block(self) -> bool:
        return self.enforcement_active and self.decision.action is PolicyAction.BLOCK

    @property
    def should_rewrite(self) -> bool:
        return (
            self.enforcement_active
            and self.rewritten_content is not None
            and self.decision.action in (PolicyAction.REDACT, PolicyAction.MASK)
        )


class BreakerGuardedDelegate:
    """Wraps a vendor evaluator so a tripped connection is skipped, not retried."""

    def __init__(self, delegate: DelegatedEvaluator, connection_id: str, breaker: CircuitBreaker) -> None:
        self._delegate: Final = delegate
        self._connection_id: Final = connection_id
        self._breaker: Final = breaker

    async def evaluate_leaf(self, leaf: ClassifierLeaf, context: EvaluationContext) -> LeafOutcome:
        if not self._breaker.allows(self._connection_id):
            return LeafOutcome(truth=Truth.INDETERMINATE)
        try:
            outcome: Final = await self._delegate.evaluate_leaf(leaf, context)
        except Exception:  # noqa: BLE001  # any vendor failure is a breaker failure, whatever its type
            self._breaker.record_failure(self._connection_id)
            DLPMetrics.record_provider_error(provider="delegated", connection_id=self._connection_id)
            return LeafOutcome(truth=Truth.INDETERMINATE)
        self._breaker.record_success(self._connection_id)
        return outcome


class CompiledPolicyStore:
    """Compile once per canonical hash. Failures are cached too, so a broken
    policy is not recompiled on every request just to fail the same way."""

    def __init__(self, re2_module: Re2Module | None = None) -> None:
        self._re2: Final = re2_module
        self._compiled: Final[  # mutable-ok: memo table keyed by canonical hash, never handed out
            dict[str, CompiledPolicy | CompileFailure]
        ] = {}  # mutable-ok: memo table keyed by canonical hash, never handed out

    def get(self, scoped: ScopedPolicy) -> CompiledPolicy | CompileFailure:
        key: Final = scoped.policy.canonical_hash
        cached: Final = self._compiled.get(key)
        if cached is not None:
            return cached
        result: Final = compile_policy(scoped.policy, re2_module=self._re2)
        self._compiled[key] = result
        return result

    def clear(self) -> None:
        self._compiled.clear()


class PolicyFabricEngine:
    def __init__(
        self,
        cache: PolicySetCache,
        store: CompiledPolicyStore,
        config: EngineConfig,
        presidio: PresidioAnalyzer | None = None,
        delegate: DelegatedEvaluator | None = None,
        breaker: CircuitBreaker | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._cache: Final = cache
        self._store: Final = store
        self._config: Final = config
        self._presidio: Final = presidio
        self._delegate: Final = delegate
        self._breaker: Final = breaker if breaker is not None else CircuitBreaker()
        self._clock: Final = clock

    async def evaluate(self, request: EvaluationRequest) -> EngineResult:
        started: Final = self._clock()
        key: Final = self._cache.key_for(
            organization_id=request.scope.organization_id,
            team_id=request.scope.team_id,
            key_alias=request.scope.key_alias,
        )
        policies: Final = applicable_policies(await self._cache.get(key), request.scope, request.direction)
        context: Final = EvaluationContext(
            content=request.content,
            direction=request.direction,
            scope=request.scope,
            tool_name=request.tool_name,
            tool_arguments=request.tool_arguments,
        )
        outcomes: Final = tuple([await self._evaluate_one(scoped, context) for scoped in policies])
        verdicts: Final = tuple(outcome for outcome in outcomes if isinstance(outcome, PolicyVerdict))
        compile_failures: Final = tuple(outcome for outcome in outcomes if isinstance(outcome, str))
        decision: Final = aggregate(verdicts)
        latency_ms: Final = int((self._clock() - started) * 1000)
        enforcement_active: Final = request.direction in V1_ENFORCED_DIRECTIONS
        enforcement_kind: Final = self._enforcement_kind(request, decision)
        rewritten: Final = _rewrite(request.content, decision)
        receipts: Final = tuple(
            self._build_receipt(request, verdict, latency_ms, enforcement_kind, rewritten is not None)
            for verdict in verdicts
        )
        for receipt in receipts:
            DLPMetrics.record_decision(
                guardrail_name=self._config.guardrail_name,
                policy_id=receipt.policy_id,
                decision=receipt.decision.value,
                direction=receipt.direction.value,
                shadow=receipt.shadow,
            )
        return EngineResult(
            decision=decision,
            receipts=receipts,
            rewritten_content=rewritten,
            enforcement_active=enforcement_active,
            enforcement_kind=enforcement_kind,
            evaluation_latency_ms=latency_ms,
            compile_failures=compile_failures,
        )

    async def _evaluate_one(self, scoped: ScopedPolicy, context: EvaluationContext) -> PolicyVerdict | str | None:
        compiled: Final = self._store.get(scoped)
        if isinstance(compiled, CompileFailure):
            return f"{scoped.policy_id}: {compiled.summary}"
        resolver: Final = PolicyLeafResolver(
            compiled=compiled,
            presidio=self._presidio,
            delegate=self._guarded_delegate(scoped),
        )
        started: Final = self._clock()
        outcome: Final = await evaluate(compiled, context, resolver)
        fail_mode: Final = self._fail_mode_for(scoped)
        action, fail_triggered = resolve_indeterminate(outcome.truth, scoped.policy.actions.on_match, fail_mode)
        if fail_triggered and scoped.connection_id is not None:
            DLPMetrics.record_fail_mode(
                connection_id=scoped.connection_id,
                failed_closed=fail_mode is FailMode.FAIL_CLOSED,
            )
        if action is None:
            return None
        return PolicyVerdict(
            policy_id=scoped.policy_id,
            policy_name=scoped.policy.name,
            policy_version=scoped.version,
            action=action,
            severity=scoped.policy.severity,
            priority=scoped.policy.priority,
            mode=scoped.policy.mode,
            shadow=scoped.shadow,
            findings=outcome.findings,
            matched_rule_ids=outcome.visited_leaf_ids,
            provider=scoped.policy.source.vendor,
            external_policy_id=scoped.policy.source.external_policy_id,
            block_message=scoped.policy.actions.block_message,
            redact_strategy=scoped.policy.actions.redact_strategy,
            fail_mode_triggered=fail_triggered,
            evaluation_latency_ms=int((self._clock() - started) * 1000),
        )

    def _guarded_delegate(self, scoped: ScopedPolicy) -> DelegatedEvaluator | None:
        if self._delegate is None or scoped.connection_id is None:
            return self._delegate
        return BreakerGuardedDelegate(
            delegate=self._delegate,
            connection_id=scoped.connection_id,
            breaker=self._breaker,
        )

    def _fail_mode_for(self, scoped: ScopedPolicy) -> FailMode:
        declared: Final = scoped.policy.fail_mode
        if declared is not None:
            return declared
        if scoped.policy.actions.on_match is PolicyAction.BLOCK and scoped.policy.severity.value == "critical":
            return self._config.critical_fail_mode
        return self._config.default_fail_mode

    def _enforcement_kind(self, request: EvaluationRequest, decision: AggregateDecision) -> EnforcementKind:
        if request.direction not in V1_ENFORCED_DIRECTIONS:
            return EnforcementKind.DETECTION
        if request.stream_state is None:
            return EnforcementKind.PREVENTION
        offsets: Final = tuple((finding.start, finding.end) for finding in decision.findings)
        return enforcement_for(self._streaming_mode(request), request.stream_state, offsets)

    def _streaming_mode(self, request: EvaluationRequest) -> StreamingMode:
        return request.streaming_mode if request.streaming_mode is not None else self._config.streaming_mode

    def _build_receipt(
        self,
        request: EvaluationRequest,
        verdict: PolicyVerdict,
        latency_ms: int,
        enforcement_kind: EnforcementKind,
        redaction_performed: bool,
    ) -> DecisionReceipt:
        return DecisionReceipt(
            request_id=request.request_id,
            scope=ReceiptScope(
                organization_id=request.scope.organization_id,
                team_id=request.scope.team_id,
                user_id=request.scope.user_id,
                key_alias=request.scope.key_alias,
                model=request.scope.model,
                application=request.scope.application,
            ),
            policy_id=verdict.policy_id,
            policy_version=verdict.policy_version,
            direction=request.direction,
            decision=verdict.action,
            enforcement=enforcement_kind,
            matched_classifiers=summarise_findings(verdict.findings),
            provider=verdict.provider,
            external_policy_id=verdict.external_policy_id,
            matched_rule_ids=verdict.matched_rule_ids,
            vendor_request_id=None,
            evaluation_latency_ms=verdict.evaluation_latency_ms or latency_ms,
            fail_mode_triggered=verdict.fail_mode_triggered,
            redaction_performed=redaction_performed and not verdict.shadow,
            shadow=verdict.shadow,
            streaming_mode=self._streaming_mode(request) if request.stream_state is not None else None,
            match_hash=correlation_hash(request.content, verdict.findings, self._config.match_hash_salt),
        )


def _rewrite(content: str, decision: AggregateDecision) -> str | None:
    if decision.deciding is None or decision.action not in (PolicyAction.REDACT, PolicyAction.MASK):
        return None
    findings: Final[tuple[Finding, ...]] = decision.findings
    if not findings:
        return None
    return apply_redaction(content, findings, decision.deciding.redact_strategy).text
