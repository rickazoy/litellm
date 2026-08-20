"""WIT-DPS v2 document validation and the canonical in-memory policy (§2.4).

Order matters: JSON Schema first (shape), then AST parse (depth and size caps),
then semantic checks. A document that fails any stage produces a
`PolicyValidationFailure` carrying every reason, because an operator reviewing a
rejected vendor import needs the whole list, not the first error.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from jsonschema import Draft202012Validator

from litellm.proxy.witos.policy_fabric.condition_ast import (
    ConditionNode,
    ParseFailure,
    parse_condition,
)
from litellm.proxy.witos.policy_fabric.types import (
    WIT_DPS_VERSION,
    FailMode,
    FederationMode,
    LogPayloadMode,
    PolicyAction,
    PolicyDirection,
    RedactStrategy,
    Severity,
    StreamingMode,
)
from litellm.proxy.witos.policy_fabric.wit_dps_schema import WIT_DPS_V2_SCHEMA

_VALIDATOR: Final = Draft202012Validator(WIT_DPS_V2_SCHEMA)


@dataclass(frozen=True, slots=True)
class ScopeEntity:
    entity_type: str
    entity_id: str


@dataclass(frozen=True, slots=True)
class PolicyScope:
    entities: tuple[ScopeEntity, ...]
    models: tuple[str, ...]
    applications: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PolicyException:
    exception_type: str
    value: str


@dataclass(frozen=True, slots=True)
class PolicyActions:
    on_match: PolicyAction
    redact_strategy: RedactStrategy
    block_message: str | None
    alert_channels: tuple[str, ...]
    log_payload: LogPayloadMode


@dataclass(frozen=True, slots=True)
class PolicySource:
    vendor: str | None
    connection_id: str | None
    external_policy_id: str | None
    external_policy_version: str | None
    external_url: str | None
    external_policy_hash: str | None
    unmapped: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WitDpsPolicy:
    policy_id: str
    name: str
    description: str | None
    severity: Severity
    mode: FederationMode
    priority: int
    direction: PolicyDirection
    scope: PolicyScope
    condition: ConditionNode
    actions: PolicyActions
    exceptions: tuple[PolicyException, ...]
    streaming_mode: StreamingMode | None
    fail_mode: FailMode | None
    source: PolicySource
    canonical_hash: str


@dataclass(frozen=True, slots=True)
class PolicyValidationFailure:
    errors: tuple[str, ...]


def canonical_json(document: Mapping[str, object]) -> str:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_hash(document: Mapping[str, object]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(document).encode("utf-8")).hexdigest()


def schema_errors(document: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(
        f"{'/'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
        for error in sorted(_VALIDATOR.iter_errors(document), key=str)
    )


def parse_policy(document: Mapping[str, object]) -> WitDpsPolicy | PolicyValidationFailure:
    shape_errors: Final = schema_errors(document)
    if shape_errors:
        return PolicyValidationFailure(errors=shape_errors)

    raw_condition: Final = document.get("condition")
    if not isinstance(raw_condition, Mapping):
        return PolicyValidationFailure(errors=("condition: must be an object",))
    condition: Final = parse_condition(raw_condition)
    if isinstance(condition, ParseFailure):
        return PolicyValidationFailure(errors=(f"condition/{condition.node_path}: {condition.reason}",))

    actions_raw: Final = document.get("actions")
    if not isinstance(actions_raw, Mapping):
        return PolicyValidationFailure(errors=("actions: must be an object",))

    try:
        return _build_policy(document, condition, actions_raw)
    except ValueError as err:
        return PolicyValidationFailure(errors=(str(err),))


def _build_policy(
    document: Mapping[str, object],
    condition: ConditionNode,
    actions_raw: Mapping[str, object],
) -> WitDpsPolicy:
    return WitDpsPolicy(
        policy_id=_optional_str(document, "policy_id") or "",
        name=_required_str(document, "name"),
        description=_optional_str(document, "description"),
        severity=Severity(_optional_str(document, "severity") or Severity.MEDIUM.value),
        mode=FederationMode(_required_str(document, "mode")),
        priority=_optional_int(document, "priority", default=100),
        direction=PolicyDirection(_optional_str(document, "direction") or PolicyDirection.BOTH.value),
        scope=_parse_scope(document.get("scope")),
        condition=condition,
        actions=_parse_actions(actions_raw),
        exceptions=_parse_exceptions(document.get("exceptions")),
        streaming_mode=_parse_streaming_mode(document.get("streaming")),
        fail_mode=_parse_fail_mode(document.get("fail_mode")),
        source=_parse_source(document.get("source")),
        canonical_hash=canonical_hash(document),
    )


def _parse_scope(raw: object) -> PolicyScope:
    if not isinstance(raw, Mapping):
        return PolicyScope(entities=(), models=(), applications=())
    entities_raw: Final = raw.get("entities")
    entities: Final = (
        tuple(
            ScopeEntity(entity_type=str(entry.get("type")), entity_id=str(entry.get("id")))
            for entry in entities_raw
            if isinstance(entry, Mapping)
        )
        if isinstance(entities_raw, Sequence) and not isinstance(entities_raw, (str, bytes))
        else ()
    )
    return PolicyScope(
        entities=entities,
        models=_str_tuple(raw.get("models")),
        applications=_str_tuple(raw.get("applications")),
    )


def _parse_actions(raw: Mapping[str, object]) -> PolicyActions:
    return PolicyActions(
        on_match=PolicyAction(_required_str(raw, "on_match")),
        redact_strategy=RedactStrategy(_optional_str(raw, "redact_strategy") or RedactStrategy.MASK.value),
        block_message=_optional_str(raw, "block_message"),
        alert_channels=_str_tuple(raw.get("alert_channels")),
        log_payload=LogPayloadMode(_optional_str(raw, "log_payload") or LogPayloadMode.METADATA_ONLY.value),
    )


def _parse_exceptions(raw: object) -> tuple[PolicyException, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    return tuple(
        PolicyException(exception_type=str(entry.get("type")), value=str(entry.get("value")))
        for entry in raw
        if isinstance(entry, Mapping)
    )


def _parse_streaming_mode(raw: object) -> StreamingMode | None:
    if not isinstance(raw, Mapping):
        return None
    mode: Final = raw.get("mode")
    return StreamingMode(mode) if isinstance(mode, str) else None


def _parse_fail_mode(raw: object) -> FailMode | None:
    return FailMode(raw) if isinstance(raw, str) else None


def _parse_source(raw: object) -> PolicySource:
    if not isinstance(raw, Mapping):
        return PolicySource(
            vendor=None,
            connection_id=None,
            external_policy_id=None,
            external_policy_version=None,
            external_url=None,
            external_policy_hash=None,
            unmapped=(),
        )
    return PolicySource(
        vendor=_optional_str(raw, "vendor"),
        connection_id=_optional_str(raw, "connection_id"),
        external_policy_id=_optional_str(raw, "external_policy_id"),
        external_policy_version=_optional_str(raw, "external_policy_version"),
        external_url=_optional_str(raw, "external_url"),
        external_policy_hash=_optional_str(raw, "external_policy_hash"),
        unmapped=_str_tuple(raw.get("unmapped")),
    )


def _str_tuple(raw: object) -> tuple[str, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    return tuple(entry for entry in raw if isinstance(entry, str))


def _required_str(raw: Mapping[str, object], key: str) -> str:
    value: Final = raw.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _optional_str(raw: Mapping[str, object], key: str) -> str | None:
    value: Final = raw.get(key)
    return value if isinstance(value, str) else None


def _optional_int(raw: Mapping[str, object], key: str, default: int) -> int:
    value: Final = raw.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def wit_dps_envelope(name: str, mode: FederationMode, condition: Mapping[str, object], on_match: PolicyAction) -> str:
    """Smallest valid WIT-DPS document, used by tests and the API examples."""
    return canonical_json(
        {  # mutable-ok: canonical document literal
            "wit_dps_version": WIT_DPS_VERSION,
            "name": name,
            "mode": mode.value,
            "condition": condition,
            "actions": {"on_match": on_match.value},  # mutable-ok: canonical document literal
        }
    )
