"""Lazy, short-circuiting AST evaluation (§2.8).

Three properties are load-bearing.

Cheap first. Children of ALL/ANY are visited in ascending evaluation cost, so a
context check that settles the node runs before any Presidio call, and a
Presidio call runs before any vendor round trip. A delegated leaf is reached
only when the condition can still flip.

Tri-state. A delegated leaf whose vendor is unreachable is INDETERMINATE, not
False. Collapsing "we could not tell" into "no match" is a silent fail-open, and
the fail mode is a policy decision that belongs to the caller, not to the
evaluator.

No `eval`. Evaluation is a `match` over frozen dataclasses.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from types import MappingProxyType
from typing import Final, Protocol

from litellm.proxy.witos.policy_fabric.compiler import CompiledPolicy
from litellm.proxy.witos.policy_fabric.condition_ast import (
    AllNode,
    AnyNode,
    ClassifierLeaf,
    ConditionNode,
    ContextLeaf,
    LeafNode,
    NotNode,
    PatternLeaf,
    TermsLeaf,
    ToolArgumentLeaf,
)
from litellm.proxy.witos.policy_fabric.types import (
    EVALUATION_TIER_COST,
    EvaluationDirection,
    EvaluationTier,
    LeafType,
)


class Truth(str, Enum):
    TRUE = "true"
    FALSE = "false"
    INDETERMINATE = "indeterminate"


class FindingSource(str, Enum):
    LOCAL_PATTERN = "local_pattern"
    PRESIDIO = "presidio"
    VENDOR = "vendor"
    CONTEXT = "context"


@dataclass(frozen=True, slots=True)
class Finding:
    """A detection. Carries where, never what.

    `start`/`end` are Python string indices (code points), which is what the
    redactor and every downstream offset consumer expect.
    """

    classifier: str
    start: int
    end: int
    confidence: float
    source: FindingSource


@dataclass(frozen=True, slots=True)
class RequestScope:
    organization_id: str | None = None
    team_id: str | None = None
    user_id: str | None = None
    key_alias: str | None = None
    end_user_id: str | None = None
    application: str | None = None
    model: str | None = None
    model_group: str | None = None
    provider: str | None = None
    destination: str | None = None
    groups: tuple[str, ...] = ()
    file_types: tuple[str, ...] = ()


_NO_TOOL_ARGUMENTS: Final[Mapping[str, str]] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class EvaluationContext:
    content: str
    direction: EvaluationDirection
    scope: RequestScope
    tool_name: str | None = None
    tool_arguments: Mapping[str, str] = _NO_TOOL_ARGUMENTS


@dataclass(frozen=True, slots=True)
class LeafOutcome:
    truth: Truth
    findings: tuple[Finding, ...] = ()


@dataclass(frozen=True, slots=True)
class EvaluationOutcome:
    truth: Truth
    findings: tuple[Finding, ...]
    delegated_calls: int
    visited_leaf_ids: tuple[str, ...]


class LeafResolver(Protocol):
    async def resolve(self, leaf: LeafNode, context: EvaluationContext) -> LeafOutcome: ...


async def evaluate(
    compiled: CompiledPolicy,
    context: EvaluationContext,
    resolver: LeafResolver,
) -> EvaluationOutcome:
    return await _evaluate_node(compiled.policy.condition, compiled, context, resolver)


async def _evaluate_node(
    node: ConditionNode,
    compiled: CompiledPolicy,
    context: EvaluationContext,
    resolver: LeafResolver,
) -> EvaluationOutcome:
    match node:
        case AllNode():
            return await _evaluate_junction(node.conditions, compiled, context, resolver, short_circuit=Truth.FALSE)
        case AnyNode():
            return await _evaluate_junction(node.conditions, compiled, context, resolver, short_circuit=Truth.TRUE)
        case NotNode():
            inner: Final = await _evaluate_node(node.condition, compiled, context, resolver)
            return EvaluationOutcome(
                truth=_negate(inner.truth),
                findings=inner.findings,
                delegated_calls=inner.delegated_calls,
                visited_leaf_ids=inner.visited_leaf_ids,
            )
        case _:
            outcome: Final = await resolver.resolve(node, context)
            return EvaluationOutcome(
                truth=outcome.truth,
                findings=outcome.findings,
                delegated_calls=1 if compiled.tier_of(node.node_id) is EvaluationTier.DELEGATED else 0,
                visited_leaf_ids=(node.node_id,),
            )


async def _evaluate_junction(
    children: tuple[ConditionNode, ...],
    compiled: CompiledPolicy,
    context: EvaluationContext,
    resolver: LeafResolver,
    short_circuit: Truth,
) -> EvaluationOutcome:
    ordered: Final = tuple(sorted(children, key=lambda child: _node_cost(child, compiled)))
    settled: Final = Truth.TRUE if short_circuit is Truth.FALSE else Truth.FALSE
    acc: EvaluationOutcome = EvaluationOutcome(  # rebind-ok: sequential fold that must stop at the short circuit
        truth=settled, findings=(), delegated_calls=0, visited_leaf_ids=()
    )
    for child in ordered:
        result: Final = await _evaluate_node(child, compiled, context, resolver)
        acc = EvaluationOutcome(
            truth=_combine(acc.truth, result.truth, short_circuit, settled),
            findings=(*acc.findings, *result.findings),
            delegated_calls=acc.delegated_calls + result.delegated_calls,
            visited_leaf_ids=(*acc.visited_leaf_ids, *result.visited_leaf_ids),
        )
        if acc.truth is short_circuit:
            return acc
    return acc


def _combine(current: Truth, incoming: Truth, short_circuit: Truth, settled: Truth) -> Truth:
    if current is short_circuit or incoming is short_circuit:
        return short_circuit
    if current is Truth.INDETERMINATE or incoming is Truth.INDETERMINATE:
        return Truth.INDETERMINATE
    return settled


def _negate(truth: Truth) -> Truth:
    match truth:
        case Truth.TRUE:
            return Truth.FALSE
        case Truth.FALSE:
            return Truth.TRUE
        case Truth.INDETERMINATE:
            return Truth.INDETERMINATE


def _node_cost(node: ConditionNode, compiled: CompiledPolicy) -> int:
    match node:
        case AllNode() | AnyNode():
            return max(_node_cost(child, compiled) for child in node.conditions)
        case NotNode():
            return _node_cost(node.condition, compiled)
        case _:
            return EVALUATION_TIER_COST[compiled.tier_of(node.node_id)]


class PresidioAnalyzer(Protocol):
    """Narrow view of LiteLLM's Presidio integration. See `presidio_bridge`."""

    async def analyze(self, text: str, entities: tuple[str, ...]) -> tuple[Finding, ...]: ...


class DelegatedEvaluator(Protocol):
    """Vendor runtime evaluation. No adapter implements this in Phase 4."""

    async def evaluate_leaf(self, leaf: ClassifierLeaf, context: EvaluationContext) -> LeafOutcome: ...


@dataclass(frozen=True, slots=True)
class PolicyLeafResolver:
    """Routes each leaf to the cheapest resolver that can answer it."""

    compiled: CompiledPolicy
    presidio: PresidioAnalyzer | None = None
    delegate: DelegatedEvaluator | None = None

    async def resolve(self, leaf: LeafNode, context: EvaluationContext) -> LeafOutcome:
        match leaf:
            case ContextLeaf():
                return _resolve_context_leaf(leaf, context)
            case ToolArgumentLeaf():
                return _resolve_tool_argument_leaf(leaf, context)
            case PatternLeaf() | TermsLeaf():
                return self._resolve_pattern_leaf(leaf, context)
            case ClassifierLeaf():
                return await self._resolve_classifier_leaf(leaf, context)

    def _resolve_pattern_leaf(self, leaf: PatternLeaf | TermsLeaf, context: EvaluationContext) -> LeafOutcome:
        compiled_pattern: Final = self.compiled.patterns.get(leaf.node_id)
        if compiled_pattern is None:
            return LeafOutcome(truth=Truth.INDETERMINATE)
        findings: Final = tuple(
            Finding(
                classifier=compiled_pattern.classifier,
                start=match.start(),
                end=match.end(),
                confidence=1.0,
                source=FindingSource.LOCAL_PATTERN,
            )
            for match in compiled_pattern.program.finditer(context.content)
        )
        matched: Final = len(findings) >= compiled_pattern.min_count
        return LeafOutcome(truth=Truth.TRUE if matched else Truth.FALSE, findings=findings if matched else ())

    async def _resolve_classifier_leaf(self, leaf: ClassifierLeaf, context: EvaluationContext) -> LeafOutcome:
        entity: Final = self.compiled.presidio_leaves.get(leaf.node_id)
        if entity is not None and self.presidio is not None:
            analysed: Final = await self.presidio.analyze(context.content, (entity,))
            qualifying: Final = tuple(
                Finding(
                    classifier=leaf.value,
                    start=finding.start,
                    end=finding.end,
                    confidence=finding.confidence,
                    source=FindingSource.PRESIDIO,
                )
                for finding in analysed
                if finding.confidence >= leaf.min_confidence
            )
            matched: Final = len(qualifying) >= leaf.min_count
            return LeafOutcome(truth=Truth.TRUE if matched else Truth.FALSE, findings=qualifying if matched else ())
        if self.delegate is not None:
            return await self.delegate.evaluate_leaf(leaf, context)
        return LeafOutcome(truth=Truth.INDETERMINATE)


def _resolve_context_leaf(leaf: ContextLeaf, context: EvaluationContext) -> LeafOutcome:
    scope: Final = context.scope
    match leaf.leaf_type:
        case LeafType.IDENTITY:
            return _match_single(leaf.value, scope.user_id)
        case LeafType.TEAM:
            return _match_single(leaf.value, scope.team_id)
        case LeafType.ORGANIZATION:
            return _match_single(leaf.value, scope.organization_id)
        case LeafType.APPLICATION:
            return _match_single(leaf.value, scope.application)
        case LeafType.MODEL:
            return _match_single(leaf.value, scope.model)
        case LeafType.MODEL_GROUP:
            return _match_single(leaf.value, scope.model_group)
        case LeafType.PROVIDER:
            return _match_single(leaf.value, scope.provider)
        case LeafType.DESTINATION:
            return _match_single(leaf.value, scope.destination)
        case LeafType.TOOL:
            return _match_single(leaf.value, context.tool_name)
        case LeafType.GROUP:
            return _match_any(leaf.value, scope.groups)
        case LeafType.FILE_TYPE:
            return _match_any(leaf.value, scope.file_types)
        case _:
            return LeafOutcome(truth=Truth.FALSE)


def _resolve_tool_argument_leaf(leaf: ToolArgumentLeaf, context: EvaluationContext) -> LeafOutcome:
    if leaf.tool is not None and not glob_match(leaf.tool, context.tool_name):
        return LeafOutcome(truth=Truth.FALSE)
    argument_value: Final = context.tool_arguments.get(leaf.argument)
    if argument_value is None:
        return LeafOutcome(truth=Truth.FALSE)
    if leaf.value is None:
        return LeafOutcome(truth=Truth.TRUE)
    return LeafOutcome(truth=Truth.TRUE if glob_match(leaf.value, argument_value) else Truth.FALSE)


def _match_single(pattern: str, value: str | None) -> LeafOutcome:
    return LeafOutcome(truth=Truth.TRUE if glob_match(pattern, value) else Truth.FALSE)


def _match_any(pattern: str, values: tuple[str, ...]) -> LeafOutcome:
    return LeafOutcome(truth=Truth.TRUE if any(glob_match(pattern, value) for value in values) else Truth.FALSE)


def glob_match(pattern: str, value: str | None) -> bool:
    """Wildcard match for scope values, without a regex engine.

    Scope values are policy-authored and short, so `*` is all that is needed.
    Routing them through a regex would put a second pattern engine on the hot
    path, which is the exact thing the re2-only rule exists to prevent.
    """
    if value is None:
        return False
    if pattern == "*":
        return True
    if "*" not in pattern:
        return pattern == value
    segments: Final = pattern.split("*")
    if len(segments[0]) + len(segments[-1]) > len(value):
        return False
    if not value.startswith(segments[0]) or not value.endswith(segments[-1]):
        return False
    inner: Final = value[len(segments[0]) : len(value) - len(segments[-1])]
    return _contains_in_order(inner, tuple(segments[1:-1]))


def _contains_in_order(haystack: str, needles: tuple[str, ...]) -> bool:
    def step(position: int | None, needle: str) -> int | None:
        if position is None:
            return None
        found: Final = haystack.find(needle, position)
        return None if found < 0 else found + len(needle)

    return reduce(step, needles, 0) is not None
