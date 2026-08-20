"""WIT-DPS condition AST: node types and a depth- and size-capped parser (§2.4).

Recursive descent over an already-JSON-Schema-validated document. There is no
`eval`, no `exec`, no code generation and no dynamic attribute lookup anywhere
in this module or the evaluator: a policy is data, and the only thing that ever
reads it is a `match` over a closed set of frozen dataclasses.

The caps exist because an imported vendor document is untrusted input. A 10,000
level `NOT` chain is a stack overflow, and a stack overflow inside a guardrail
is a bypass.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import reduce
from typing import Final, TypeAlias

from litellm.proxy.witos.policy_fabric.types import LeafType

MAX_CONDITION_DEPTH: Final = 12
MAX_CONDITION_NODES: Final = 256
ROOT_NODE_ID: Final = "c"


@dataclass(frozen=True, slots=True)
class AllNode:
    node_id: str
    conditions: tuple[ConditionNode, ...]


@dataclass(frozen=True, slots=True)
class AnyNode:
    node_id: str
    conditions: tuple[ConditionNode, ...]


@dataclass(frozen=True, slots=True)
class NotNode:
    node_id: str
    condition: ConditionNode


@dataclass(frozen=True, slots=True)
class ClassifierLeaf:
    """`data_class`, `sensitivity`, `sensitivity_label`, `classification_source`."""

    node_id: str
    leaf_type: LeafType
    value: str
    min_confidence: float
    min_count: int


@dataclass(frozen=True, slots=True)
class PatternLeaf:
    node_id: str
    pattern: str
    case_sensitive: bool
    min_count: int
    classifier: str


@dataclass(frozen=True, slots=True)
class TermsLeaf:
    """`dictionary` and `keyword` differ only in provenance, not in matching."""

    node_id: str
    leaf_type: LeafType
    terms: tuple[str, ...]
    case_sensitive: bool
    whole_word: bool
    min_count: int
    classifier: str


@dataclass(frozen=True, slots=True)
class ContextLeaf:
    node_id: str
    leaf_type: LeafType
    value: str


@dataclass(frozen=True, slots=True)
class ToolArgumentLeaf:
    node_id: str
    tool: str | None
    argument: str
    value: str | None


LeafNode: TypeAlias = ClassifierLeaf | PatternLeaf | TermsLeaf | ContextLeaf | ToolArgumentLeaf
ConditionNode: TypeAlias = AllNode | AnyNode | NotNode | LeafNode


@dataclass(frozen=True, slots=True)
class ParseFailure:
    node_path: str
    reason: str


class _ParseError(Exception):
    """Module-internal control flow. Converted to a `ParseFailure` at the boundary."""

    def __init__(self, node_path: str, reason: str) -> None:
        super().__init__(f"{node_path}: {reason}")
        self.node_path: Final = node_path
        self.reason: Final = reason


_CLASSIFIER_LEAF_TYPES: Final[frozenset[str]] = frozenset(
    {
        LeafType.DATA_CLASS.value,
        LeafType.SENSITIVITY.value,
        LeafType.SENSITIVITY_LABEL.value,
        LeafType.CLASSIFICATION_SOURCE.value,
    }
)
_TERMS_LEAF_TYPES: Final[frozenset[str]] = frozenset({LeafType.DICTIONARY.value, LeafType.KEYWORD.value})
_CONTEXT_LEAF_TYPES: Final[frozenset[str]] = frozenset(
    {
        LeafType.IDENTITY.value,
        LeafType.GROUP.value,
        LeafType.TEAM.value,
        LeafType.ORGANIZATION.value,
        LeafType.APPLICATION.value,
        LeafType.MODEL.value,
        LeafType.MODEL_GROUP.value,
        LeafType.PROVIDER.value,
        LeafType.DESTINATION.value,
        LeafType.FILE_TYPE.value,
        LeafType.TOOL.value,
    }
)


@dataclass(frozen=True, slots=True)
class _Budget:
    """Node allowance threaded through the descent as a value, never mutated."""

    remaining: int

    def spend(self, node_path: str) -> _Budget:
        if self.remaining <= 0:
            raise _ParseError(node_path, f"condition exceeds the {MAX_CONDITION_NODES}-node cap")
        return _Budget(self.remaining - 1)


def parse_condition(raw: Mapping[str, object]) -> ConditionNode | ParseFailure:
    try:
        node, _ = _parse_node(raw, ROOT_NODE_ID, depth=1, budget=_Budget(MAX_CONDITION_NODES))
    except _ParseError as err:
        return ParseFailure(node_path=err.node_path, reason=err.reason)
    return node


def count_nodes(node: ConditionNode) -> int:
    match node:
        case AllNode() | AnyNode():
            return 1 + sum(count_nodes(child) for child in node.conditions)
        case NotNode():
            return 1 + count_nodes(node.condition)
        case _:
            return 1


def depth_of(node: ConditionNode) -> int:
    match node:
        case AllNode() | AnyNode():
            return 1 + max(depth_of(child) for child in node.conditions)
        case NotNode():
            return 1 + depth_of(node.condition)
        case _:
            return 1


def iter_leaves(node: ConditionNode) -> tuple[LeafNode, ...]:
    match node:
        case AllNode() | AnyNode():
            return tuple(leaf for child in node.conditions for leaf in iter_leaves(child))
        case NotNode():
            return iter_leaves(node.condition)
        case _:
            return (node,)


def _parse_node(
    raw: Mapping[str, object],
    node_path: str,
    depth: int,
    budget: _Budget,
) -> tuple[ConditionNode, _Budget]:
    if depth > MAX_CONDITION_DEPTH:
        raise _ParseError(node_path, f"condition exceeds the {MAX_CONDITION_DEPTH}-level depth cap")
    spent: Final = budget.spend(node_path)
    operator: Final = raw.get("operator")
    if operator is None:
        return _parse_leaf(raw, node_path), spent
    match operator:
        case "ALL" | "ANY":
            children_raw: Final = raw.get("conditions")
            if not isinstance(children_raw, Sequence) or isinstance(children_raw, (str, bytes)) or not children_raw:
                raise _ParseError(node_path, f"{operator} requires a non-empty `conditions` array")
            children, remaining = _parse_children(children_raw, node_path, depth, spent)
            node: Final[ConditionNode] = (
                AllNode(node_id=node_path, conditions=children)
                if operator == "ALL"
                else AnyNode(node_id=node_path, conditions=children)
            )
            return node, remaining
        case "NOT":
            child_raw: Final = raw.get("condition")
            if not isinstance(child_raw, Mapping):
                raise _ParseError(node_path, "NOT requires a `condition` object")
            child, remaining_after_not = _parse_node(child_raw, f"{node_path}.0", depth + 1, spent)
            return NotNode(node_id=node_path, condition=child), remaining_after_not
        case _:
            raise _ParseError(node_path, f"unknown operator {operator!r}; expected ALL, ANY or NOT")


def _parse_children(
    children_raw: Sequence[object],
    node_path: str,
    depth: int,
    budget: _Budget,
) -> tuple[tuple[ConditionNode, ...], _Budget]:
    def fold(
        acc: tuple[tuple[ConditionNode, ...], _Budget],
        indexed: tuple[int, object],
    ) -> tuple[tuple[ConditionNode, ...], _Budget]:
        parsed_so_far, remaining = acc
        index, child_raw = indexed
        child_path: Final = f"{node_path}.{index}"
        if not isinstance(child_raw, Mapping):
            raise _ParseError(child_path, "condition entries must be objects")
        child, next_remaining = _parse_node(child_raw, child_path, depth + 1, remaining)
        return (*parsed_so_far, child), next_remaining

    empty: Final[tuple[tuple[ConditionNode, ...], _Budget]] = ((), budget)
    return reduce(fold, enumerate(children_raw), empty)


def _parse_leaf(raw: Mapping[str, object], node_path: str) -> LeafNode:
    leaf_type: Final = raw.get("type")
    if not isinstance(leaf_type, str):
        raise _ParseError(node_path, "leaf requires a string `type`")
    if leaf_type in _CLASSIFIER_LEAF_TYPES:
        return _parse_classifier_leaf(raw, node_path, leaf_type)
    if leaf_type == LeafType.REGEX.value:
        return _parse_pattern_leaf(raw, node_path)
    if leaf_type in _TERMS_LEAF_TYPES:
        return _parse_terms_leaf(raw, node_path, leaf_type)
    if leaf_type in _CONTEXT_LEAF_TYPES:
        return _parse_context_leaf(raw, node_path, leaf_type)
    if leaf_type == LeafType.TOOL_ARGUMENT.value:
        return _parse_tool_argument_leaf(raw, node_path)
    raise _ParseError(node_path, f"unknown leaf type {leaf_type!r}")


def _parse_classifier_leaf(raw: Mapping[str, object], node_path: str, leaf_type: str) -> ClassifierLeaf:
    value: Final = raw.get("class") if raw.get("class") is not None else raw.get("value")
    if not isinstance(value, str) or not value:
        raise _ParseError(node_path, f"{leaf_type} requires a non-empty `class` or `value`")
    return ClassifierLeaf(
        node_id=node_path,
        leaf_type=LeafType(leaf_type),
        value=value,
        min_confidence=_float_field(raw, "min_confidence", node_path, default=0.0),
        min_count=_int_field(raw, "min_count", node_path, default=1),
    )


def _parse_pattern_leaf(raw: Mapping[str, object], node_path: str) -> PatternLeaf:
    pattern: Final = raw.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        raise _ParseError(node_path, "regex requires a non-empty `pattern`")
    return PatternLeaf(
        node_id=node_path,
        pattern=pattern,
        case_sensitive=_bool_field(raw, "case_sensitive", node_path, default=True),
        min_count=_int_field(raw, "min_count", node_path, default=1),
        classifier=_str_field(raw, "classifier", node_path, default="CUSTOM.local.regex"),
    )


def _parse_terms_leaf(raw: Mapping[str, object], node_path: str, leaf_type: str) -> TermsLeaf:
    terms_raw: Final = raw.get("terms")
    if not isinstance(terms_raw, Sequence) or isinstance(terms_raw, (str, bytes)) or not terms_raw:
        raise _ParseError(node_path, f"{leaf_type} requires a non-empty `terms` array")
    terms: Final = tuple(term for term in terms_raw if isinstance(term, str) and term)
    if len(terms) != len(terms_raw):
        raise _ParseError(node_path, f"{leaf_type} `terms` must contain only non-empty strings")
    return TermsLeaf(
        node_id=node_path,
        leaf_type=LeafType(leaf_type),
        terms=terms,
        case_sensitive=_bool_field(raw, "case_sensitive", node_path, default=False),
        whole_word=_bool_field(raw, "whole_word", node_path, default=True),
        min_count=_int_field(raw, "min_count", node_path, default=1),
        classifier=_str_field(raw, "classifier", node_path, default=f"CUSTOM.local.{leaf_type}"),
    )


def _parse_context_leaf(raw: Mapping[str, object], node_path: str, leaf_type: str) -> ContextLeaf:
    value: Final = raw.get("value")
    if not isinstance(value, str) or not value:
        raise _ParseError(node_path, f"{leaf_type} requires a non-empty `value`")
    return ContextLeaf(node_id=node_path, leaf_type=LeafType(leaf_type), value=value)


def _parse_tool_argument_leaf(raw: Mapping[str, object], node_path: str) -> ToolArgumentLeaf:
    argument: Final = raw.get("argument")
    if not isinstance(argument, str) or not argument:
        raise _ParseError(node_path, "tool_argument requires a non-empty `argument`")
    tool: Final = raw.get("tool")
    value: Final = raw.get("value")
    if tool is not None and not isinstance(tool, str):
        raise _ParseError(node_path, "tool_argument `tool` must be a string")
    if value is not None and not isinstance(value, str):
        raise _ParseError(node_path, "tool_argument `value` must be a string")
    return ToolArgumentLeaf(node_id=node_path, tool=tool, argument=argument, value=value)


def _bool_field(raw: Mapping[str, object], key: str, node_path: str, default: bool) -> bool:
    value: Final = raw.get(key)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise _ParseError(node_path, f"`{key}` must be a boolean")
    return value


def _int_field(raw: Mapping[str, object], key: str, node_path: str, default: int) -> int:
    value: Final = raw.get(key)
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise _ParseError(node_path, f"`{key}` must be a positive integer")
    return value


def _float_field(raw: Mapping[str, object], key: str, node_path: str, default: float) -> float:
    value: Final = raw.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _ParseError(node_path, f"`{key}` must be a number")
    if not 0.0 <= float(value) <= 1.0:
        raise _ParseError(node_path, f"`{key}` must be between 0 and 1")
    return float(value)


def _str_field(raw: Mapping[str, object], key: str, node_path: str, default: str) -> str:
    value: Final = raw.get(key)
    if value is None:
        return default
    if not isinstance(value, str) or not value:
        raise _ParseError(node_path, f"`{key}` must be a non-empty string")
    return value
