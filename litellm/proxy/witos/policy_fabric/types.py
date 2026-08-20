"""Canonical WIT-DPS v2 vocabulary (blueprint v2 §2.2, §2.3, §2.4, §2.9).

Every enum here is a closed set. Vendor-specific strings never reach these
types directly: they pass through `classifier_registry` first, which keeps the
vendor identity alongside the canonical value.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType
from typing import Final


class FederationMode(str, Enum):
    """§2.2 — how a policy is enforced relative to its vendor."""

    MIRROR = "mirror"
    DELEGATE = "delegate"
    HYBRID = "hybrid"
    OBSERVE = "observe"


class PolicyStatus(str, Enum):
    IMPORTED = "imported"
    DRAFT = "draft"
    SHADOW = "shadow"
    ACTIVE = "active"
    DISABLED = "disabled"
    STALE = "stale"
    CONFLICT = "conflict"
    SUPERSEDED = "superseded"


class PolicyAction(str, Enum):
    BLOCK = "BLOCK"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"
    REDACT = "REDACT"
    MASK = "MASK"
    WARN = "WARN"
    AUDIT = "AUDIT"
    ALLOW = "ALLOW"


# §2.4 — BLOCK > REQUIRE_APPROVAL > REDACT > MASK > WARN > AUDIT > ALLOW.
# Higher rank wins in the aggregator.
ACTION_PRECEDENCE: Final[Mapping[PolicyAction, int]] = MappingProxyType(
    {
        PolicyAction.BLOCK: 6,
        PolicyAction.REQUIRE_APPROVAL: 5,
        PolicyAction.REDACT: 4,
        PolicyAction.MASK: 3,
        PolicyAction.WARN: 2,
        PolicyAction.AUDIT: 1,
        PolicyAction.ALLOW: 0,
    }
)

# REQUIRE_APPROVAL is v1.5 (§2.4): the approval queue does not exist yet, so a
# policy asking for it is compiled but never activated without this flag.
ENFORCEABLE_ACTIONS: Final[frozenset[PolicyAction]] = frozenset(
    {
        PolicyAction.BLOCK,
        PolicyAction.REDACT,
        PolicyAction.MASK,
        PolicyAction.WARN,
        PolicyAction.AUDIT,
        PolicyAction.ALLOW,
    }
)

# Activating either of these against live traffic needs a human (§2.7).
HUMAN_APPROVAL_ACTIONS: Final[frozenset[PolicyAction]] = frozenset({PolicyAction.BLOCK, PolicyAction.REQUIRE_APPROVAL})


class PolicyDirection(str, Enum):
    """What a policy declares it applies to."""

    INPUT = "input"
    OUTPUT = "output"
    BOTH = "both"
    TOOL = "tool"
    TOOL_INPUT = "tool_input"
    TOOL_OUTPUT = "tool_output"
    RAG_CONTEXT = "rag_context"


class EvaluationDirection(str, Enum):
    """What a concrete payload being evaluated actually is."""

    INPUT = "input"
    OUTPUT = "output"
    TOOL_INPUT = "tool_input"
    TOOL_OUTPUT = "tool_output"
    RAG_CONTEXT = "rag_context"


_DIRECTION_COVERAGE: Final[Mapping[PolicyDirection, frozenset[EvaluationDirection]]] = MappingProxyType(
    {
        PolicyDirection.INPUT: frozenset({EvaluationDirection.INPUT}),
        PolicyDirection.OUTPUT: frozenset({EvaluationDirection.OUTPUT}),
        PolicyDirection.BOTH: frozenset({EvaluationDirection.INPUT, EvaluationDirection.OUTPUT}),
        PolicyDirection.TOOL: frozenset({EvaluationDirection.TOOL_INPUT, EvaluationDirection.TOOL_OUTPUT}),
        PolicyDirection.TOOL_INPUT: frozenset({EvaluationDirection.TOOL_INPUT}),
        PolicyDirection.TOOL_OUTPUT: frozenset({EvaluationDirection.TOOL_OUTPUT}),
        PolicyDirection.RAG_CONTEXT: frozenset({EvaluationDirection.RAG_CONTEXT}),
    }
)

# §2.10 — the interface ships in v1, enforcement on these directions does not.
# A policy scoped to one of them evaluates and writes receipts; the guardrail
# refuses to let it change the response.
V1_ENFORCED_DIRECTIONS: Final[frozenset[EvaluationDirection]] = frozenset(
    {EvaluationDirection.INPUT, EvaluationDirection.OUTPUT}
)


def policy_covers_direction(declared: PolicyDirection, actual: EvaluationDirection) -> bool:
    return actual in _DIRECTION_COVERAGE[declared]


class LeafType(str, Enum):
    """§2.4 closed set. Adding a member is a deliberate spec change."""

    DATA_CLASS = "data_class"
    SENSITIVITY = "sensitivity"
    REGEX = "regex"
    DICTIONARY = "dictionary"
    KEYWORD = "keyword"
    SENSITIVITY_LABEL = "sensitivity_label"
    IDENTITY = "identity"
    GROUP = "group"
    TEAM = "team"
    ORGANIZATION = "organization"
    APPLICATION = "application"
    MODEL = "model"
    MODEL_GROUP = "model_group"
    PROVIDER = "provider"
    DESTINATION = "destination"
    FILE_TYPE = "file_type"
    TOOL = "tool"
    TOOL_ARGUMENT = "tool_argument"
    CLASSIFICATION_SOURCE = "classification_source"


CONTENT_LEAF_TYPES: Final[frozenset[LeafType]] = frozenset(
    {
        LeafType.DATA_CLASS,
        LeafType.SENSITIVITY,
        LeafType.REGEX,
        LeafType.DICTIONARY,
        LeafType.KEYWORD,
        LeafType.SENSITIVITY_LABEL,
        LeafType.CLASSIFICATION_SOURCE,
    }
)

CONTEXT_LEAF_TYPES: Final[frozenset[LeafType]] = frozenset(
    {
        LeafType.IDENTITY,
        LeafType.GROUP,
        LeafType.TEAM,
        LeafType.ORGANIZATION,
        LeafType.APPLICATION,
        LeafType.MODEL,
        LeafType.MODEL_GROUP,
        LeafType.PROVIDER,
        LeafType.DESTINATION,
        LeafType.FILE_TYPE,
        LeafType.TOOL,
    }
)


class Capability(str, Enum):
    """§2.3 closed set. Adapters advertise; the UI and sync obey."""

    POLICY_LIST = "POLICY_LIST"
    POLICY_READ = "POLICY_READ"
    POLICY_VERSION = "POLICY_VERSION"
    CLASSIFIER_LIST = "CLASSIFIER_LIST"
    REALTIME_INPUT_EVALUATION = "REALTIME_INPUT_EVALUATION"
    REALTIME_OUTPUT_EVALUATION = "REALTIME_OUTPUT_EVALUATION"
    REDACTION = "REDACTION"
    WEBHOOKS = "WEBHOOKS"
    POLICY_PUSH = "POLICY_PUSH"


class FailMode(str, Enum):
    FAIL_OPEN = "fail_open"
    FAIL_CLOSED = "fail_closed"
    OBSERVE = "observe"


class StreamingMode(str, Enum):
    """§2.9. `prevents_disclosure` is the honesty rule expressed in code."""

    BUFFER_FULL = "buffer_full"
    CHUNK_GATE = "chunk_gate"
    OBSERVE_ONLY = "observe_only"

    @property
    def prevents_disclosure(self) -> bool:
        """True only if the mode can withhold offending content from the client.

        `observe_only` streams every chunk the moment it arrives, so a violation
        it finds has already reached the caller. That is detection, and the API,
        the receipt and the docs all have to say so.
        """
        return self is not StreamingMode.OBSERVE_ONLY


class EnforcementKind(str, Enum):
    """How a decision acted on the world. Never inferred from the action alone."""

    PREVENTION = "prevention"
    DETECTION = "detection"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RedactStrategy(str, Enum):
    MASK = "mask"
    REPLACE = "replace"
    HASH = "hash"
    REMOVE = "remove"


class LogPayloadMode(str, Enum):
    METADATA_ONLY = "metadata_only"
    NONE = "none"


class CompileStatus(str, Enum):
    COMPILED_LOCAL = "compiled_local"
    DELEGATED = "delegated"
    HYBRID = "hybrid"
    FAILED = "failed"


class EvaluationTier(str, Enum):
    """Evaluation cost, used to order lazy AST evaluation cheapest-first."""

    CONTEXT = "context"
    LOCAL_PATTERN = "local_pattern"
    PRESIDIO = "presidio"
    DELEGATED = "delegated"


EVALUATION_TIER_COST: Final[Mapping[EvaluationTier, int]] = MappingProxyType(
    {
        EvaluationTier.CONTEXT: 0,
        EvaluationTier.LOCAL_PATTERN: 1,
        EvaluationTier.PRESIDIO: 2,
        EvaluationTier.DELEGATED: 3,
    }
)

WIT_DPS_VERSION: Final = "2.0"
