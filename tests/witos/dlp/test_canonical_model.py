"""WIT-DPS validation, the AST caps, and canonical hashing (§2.4, §2.14)."""

from __future__ import annotations

from typing import Any

from conftest import policy_document

from litellm.proxy.witos.policy_fabric.canonical import (
    PolicyValidationFailure,
    canonical_hash,
    parse_policy,
)
from litellm.proxy.witos.policy_fabric.condition_ast import (
    MAX_CONDITION_DEPTH,
    MAX_CONDITION_NODES,
    AllNode,
    ParseFailure,
    count_nodes,
    depth_of,
    iter_leaves,
    parse_condition,
)
from litellm.proxy.witos.policy_fabric.types import LeafType


def _nested_not(depth: int) -> dict[str, Any]:
    node: dict[str, Any] = {"type": "team", "value": "payments"}
    for _ in range(depth):
        node = {"operator": "NOT", "condition": node}
    return node


def test_depth_cap_rejects_a_condition_deeper_than_the_limit() -> None:
    inside = parse_condition(_nested_not(MAX_CONDITION_DEPTH - 2))
    assert not isinstance(inside, ParseFailure)
    assert depth_of(inside) <= MAX_CONDITION_DEPTH

    beyond = parse_condition(_nested_not(MAX_CONDITION_DEPTH + 5))
    assert isinstance(beyond, ParseFailure)
    assert "depth cap" in beyond.reason


def test_size_cap_rejects_a_condition_with_too_many_nodes() -> None:
    wide = {
        "operator": "ANY",
        "conditions": [{"type": "keyword", "terms": [f"term{index}"]} for index in range(MAX_CONDITION_NODES + 10)],
    }
    parsed = parse_condition(wide)
    assert isinstance(parsed, ParseFailure)
    assert "node cap" in parsed.reason


def test_size_cap_counts_the_whole_tree_not_just_one_level() -> None:
    """Nodes spread across branches still consume the same budget."""
    branch = {
        "operator": "ANY",
        "conditions": [{"type": "keyword", "terms": ["a"]} for _ in range(60)],
    }
    tree = {"operator": "ALL", "conditions": [branch, branch, branch, branch, branch]}
    parsed = parse_condition(tree)
    assert isinstance(parsed, ParseFailure)


def test_a_condition_just_inside_the_size_cap_parses() -> None:
    wide = {
        "operator": "ANY",
        "conditions": [{"type": "keyword", "terms": ["x"]} for _ in range(MAX_CONDITION_NODES - 1)],
    }
    parsed = parse_condition(wide)
    assert isinstance(parsed, AllNode) is False
    assert not isinstance(parsed, ParseFailure)
    assert count_nodes(parsed) == MAX_CONDITION_NODES


def test_unknown_operator_and_unknown_leaf_are_rejected() -> None:
    assert isinstance(parse_condition({"operator": "XOR", "conditions": []}), ParseFailure)
    assert isinstance(parse_condition({"type": "prompt_intent", "value": "exfiltration"}), ParseFailure)


def test_every_v1_leaf_type_parses() -> None:
    leaves: list[dict[str, Any]] = [
        {"type": "data_class", "class": "PII.SSN", "min_confidence": 0.9},
        {"type": "sensitivity", "value": "restricted"},
        {"type": "regex", "pattern": r"\d{4}"},
        {"type": "dictionary", "terms": ["alpha", "beta"]},
        {"type": "keyword", "terms": ["secret"]},
        {"type": "sensitivity_label", "value": "Confidential"},
        {"type": "identity", "value": "u1"},
        {"type": "group", "value": "secops"},
        {"type": "team", "value": "payments"},
        {"type": "organization", "value": "org1"},
        {"type": "application", "value": "app1"},
        {"type": "model", "value": "gpt-*"},
        {"type": "model_group", "value": "fast"},
        {"type": "provider", "value": "openai"},
        {"type": "destination", "value": "eu"},
        {"type": "file_type", "value": "pdf"},
        {"type": "tool", "value": "salesforce_query"},
        {"type": "tool_argument", "argument": "soql"},
        {"type": "classification_source", "value": "cyera"},
    ]
    assert len(leaves) == len(LeafType)
    parsed = parse_condition({"operator": "ANY", "conditions": leaves})
    assert not isinstance(parsed, ParseFailure)
    assert len(iter_leaves(parsed)) == len(LeafType)


def test_schema_rejects_unknown_top_level_fields() -> None:
    document = policy_document("PCI")
    document["exfiltrate_to"] = "https://example.invalid"
    parsed = parse_policy(document)
    assert isinstance(parsed, PolicyValidationFailure)


def test_canonical_hash_is_key_order_independent_and_content_sensitive() -> None:
    first = {"wit_dps_version": "2.0", "name": "a", "mode": "mirror"}
    reordered = {"mode": "mirror", "name": "a", "wit_dps_version": "2.0"}
    changed = {"wit_dps_version": "2.0", "name": "b", "mode": "mirror"}
    assert canonical_hash(first) == canonical_hash(reordered)
    assert canonical_hash(first) != canonical_hash(changed)


def test_leaf_node_ids_are_deterministic_paths() -> None:
    parsed = parse_condition(
        {
            "operator": "ALL",
            "conditions": [
                {"type": "team", "value": "payments"},
                {"operator": "NOT", "condition": {"type": "keyword", "terms": ["ok"]}},
            ],
        }
    )
    assert not isinstance(parsed, ParseFailure)
    assert [leaf.node_id for leaf in iter_leaves(parsed)] == ["c.0", "c.1.0"]
