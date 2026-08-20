"""Policy compiler (§2.8): local detectors, Presidio routing, delegated plan.

Every local pattern is compiled with **google-re2**. Python's `re` is a
backtracking engine, and an imported vendor pattern is untrusted input, so a
single crafted policy would turn the guardrail into a denial-of-service against
the proxy. re2 is linear-time by construction and simply refuses the constructs
that make backtracking explosive.

The consequence is deliberate and load-bearing: a pattern re2 rejects fails the
compile and lands in the review queue. It is never retried with `re`, and it is
never silently dropped, because a policy that quietly stops matching is worse
than one that visibly fails to install. This module does not import `re`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Protocol, runtime_checkable

from litellm.proxy.witos.policy_fabric.canonical import WitDpsPolicy
from litellm.proxy.witos.policy_fabric.classifier_registry import (
    CANONICAL_TO_PRESIDIO_ENTITY,
    resolves_locally,
)
from litellm.proxy.witos.policy_fabric.condition_ast import (
    ClassifierLeaf,
    ContextLeaf,
    LeafNode,
    PatternLeaf,
    TermsLeaf,
    ToolArgumentLeaf,
    iter_leaves,
)
from litellm.proxy.witos.policy_fabric.types import (
    CompileStatus,
    EvaluationTier,
    LeafType,
)


@runtime_checkable
class Re2Match(Protocol):
    def start(self) -> int: ...

    def end(self) -> int: ...


class Re2Pattern(Protocol):
    def finditer(self, text: str) -> Iterator[Re2Match]: ...


class Re2Module(Protocol):
    """The slice of google-re2 the compiler uses. Injectable so tests can prove
    the rejection path without depending on a specific re2 build."""

    def compile(self, pattern: str) -> Re2Pattern: ...

    def escape(self, literal: str) -> str: ...


@dataclass(frozen=True, slots=True)
class CompiledPattern:
    node_id: str
    classifier: str
    min_count: int
    program: Re2Pattern


@dataclass(frozen=True, slots=True)
class CompileError:
    node_id: str
    leaf_type: str
    reason: str
    pattern: str | None


@dataclass(frozen=True, slots=True)
class CompiledPolicy:
    policy: WitDpsPolicy
    patterns: Mapping[str, CompiledPattern]
    presidio_leaves: Mapping[str, str]
    delegated_leaves: frozenset[str]
    tiers: Mapping[str, EvaluationTier]
    compile_status: CompileStatus

    def tier_of(self, node_id: str) -> EvaluationTier:
        return self.tiers.get(node_id, EvaluationTier.CONTEXT)


@dataclass(frozen=True, slots=True)
class CompileFailure:
    policy_name: str
    errors: tuple[CompileError, ...]

    @property
    def summary(self) -> str:
        return "; ".join(f"{err.node_id} ({err.leaf_type}): {err.reason}" for err in self.errors)


class _MissingRe2:
    """Stand-in used when google-re2 is not installed.

    It fails every compile with an actionable reason instead of importing `re`.
    Degrading to a backtracking engine would silently reintroduce exactly the
    risk re2 was chosen to remove.
    """

    def compile(self, pattern: str) -> Re2Pattern:
        raise RuntimeError(
            "google-re2 is not installed; local DLP detectors cannot be compiled. "
            "Install the `witos` extra. Falling back to Python `re` is not supported."
        )

    def escape(self, literal: str) -> str:
        raise RuntimeError("google-re2 is not installed; refusing to escape patterns with another engine")


def load_re2() -> Re2Module:
    try:
        import re2
    except ImportError:
        return _MissingRe2()
    return re2


def compile_policy(policy: WitDpsPolicy, re2_module: Re2Module | None = None) -> CompiledPolicy | CompileFailure:
    engine: Final = re2_module if re2_module is not None else load_re2()
    leaves: Final = iter_leaves(policy.condition)
    outcomes: Final = tuple(_compile_leaf(leaf, engine) for leaf in leaves)
    errors: Final = tuple(outcome for outcome in outcomes if isinstance(outcome, CompileError))
    if errors:
        return CompileFailure(policy_name=policy.name, errors=errors)

    patterns: Final = MappingProxyType(
        {outcome.node_id: outcome for outcome in outcomes if isinstance(outcome, CompiledPattern)}
    )
    presidio_leaves: Final = MappingProxyType(
        {leaf.node_id: entity for leaf, entity in ((leaf, _presidio_entity_for(leaf)) for leaf in leaves) if entity}
    )
    delegated_leaves: Final = frozenset(
        leaf.node_id for leaf in leaves if isinstance(leaf, ClassifierLeaf) and leaf.node_id not in presidio_leaves
    )
    tiers: Final = MappingProxyType(
        {leaf.node_id: _tier_for(leaf, presidio_leaves, delegated_leaves) for leaf in leaves}
    )
    return CompiledPolicy(
        policy=policy,
        patterns=patterns,
        presidio_leaves=presidio_leaves,
        delegated_leaves=delegated_leaves,
        tiers=tiers,
        compile_status=_status_for(tuple(tiers.values())),
    )


def _compile_leaf(leaf: LeafNode, engine: Re2Module) -> CompiledPattern | CompileError | None:
    match leaf:
        case PatternLeaf():
            return _compile_regex(leaf, engine)
        case TermsLeaf():
            return _compile_terms(leaf, engine)
        case ClassifierLeaf() | ContextLeaf() | ToolArgumentLeaf():
            return None


def _compile_regex(leaf: PatternLeaf, engine: Re2Module) -> CompiledPattern | CompileError:
    source: Final = leaf.pattern if leaf.case_sensitive else f"(?i){leaf.pattern}"
    try:
        program: Final = engine.compile(source)
    except Exception as err:  # noqa: BLE001  # re2 signals every rejection with its own private error type
        return CompileError(
            node_id=leaf.node_id,
            leaf_type=LeafType.REGEX.value,
            reason=f"re2 rejected the pattern: {_reason(err)}",
            pattern=leaf.pattern,
        )
    return CompiledPattern(
        node_id=leaf.node_id,
        classifier=leaf.classifier,
        min_count=leaf.min_count,
        program=program,
    )


def _compile_terms(leaf: TermsLeaf, engine: Re2Module) -> CompiledPattern | CompileError:
    try:
        alternation: Final = "|".join(engine.escape(term) for term in leaf.terms)
        bounded: Final = rf"\b(?:{alternation})\b" if leaf.whole_word else f"(?:{alternation})"
        source: Final = bounded if leaf.case_sensitive else f"(?i){bounded}"
        program: Final = engine.compile(source)
    except Exception as err:  # noqa: BLE001  # re2 signals every rejection with its own private error type
        return CompileError(
            node_id=leaf.node_id,
            leaf_type=leaf.leaf_type.value,
            reason=f"re2 rejected the compiled term list: {_reason(err)}",
            pattern=None,
        )
    return CompiledPattern(
        node_id=leaf.node_id,
        classifier=leaf.classifier,
        min_count=leaf.min_count,
        program=program,
    )


def _presidio_entity_for(leaf: LeafNode) -> str | None:
    if not isinstance(leaf, ClassifierLeaf) or leaf.leaf_type is not LeafType.DATA_CLASS:
        return None
    if not resolves_locally(leaf.value):
        return None
    return CANONICAL_TO_PRESIDIO_ENTITY[leaf.value]


def _tier_for(
    leaf: LeafNode,
    presidio_leaves: Mapping[str, str],
    delegated_leaves: frozenset[str],
) -> EvaluationTier:
    if leaf.node_id in presidio_leaves:
        return EvaluationTier.PRESIDIO
    if leaf.node_id in delegated_leaves:
        return EvaluationTier.DELEGATED
    match leaf:
        case PatternLeaf() | TermsLeaf():
            return EvaluationTier.LOCAL_PATTERN
        case _:
            return EvaluationTier.CONTEXT


def _status_for(tiers: tuple[EvaluationTier, ...]) -> CompileStatus:
    observed: Final = frozenset(tiers)
    if EvaluationTier.DELEGATED not in observed:
        return CompileStatus.COMPILED_LOCAL
    if observed == {EvaluationTier.DELEGATED}:  # mutable-ok: re2 module shim
        return CompileStatus.DELEGATED
    return CompileStatus.HYBRID


def _reason(err: Exception) -> str:
    text: Final = str(err)
    return text[:400] if text else type(err).__name__
