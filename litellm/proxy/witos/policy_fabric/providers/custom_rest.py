"""Generic REST adapter (§2.6, adapter 1).

For a vendor that exposes policies over HTTP in its own shape. A declarative
field map says which keys mean what; everything the map does not consume is
reported in `MappingResult.unmapped` so a reviewer sees precisely what was lost
in translation.

An unrecognised vendor action is an error, not a guess. Mapping an unknown
verb to "audit" would install a policy the customer believes is blocking, and
mapping it to "block" would install one nobody asked for. Refusing is the only
answer that cannot silently be wrong.

Transport is injected. The adapter itself makes no assumption about how bytes
are fetched, which is what lets the whole thing be tested without a network.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Final, Protocol

from litellm.proxy.witos.policy_fabric.canonical import PolicyValidationFailure, parse_policy
from litellm.proxy.witos.policy_fabric.providers.base import (
    ConnectionSettings,
    ConnectorHealth,
    DLPProviderAdapter,
    MappingResult,
    UnsupportedCapability,
    VendorClassifier,
    VendorPolicyRef,
)
from litellm.proxy.witos.policy_fabric.providers.custom_rest_endpoints import RestEndpoints
from litellm.proxy.witos.policy_fabric.types import (
    WIT_DPS_VERSION,
    Capability,
    FederationMode,
    PolicyAction,
)

PROVIDER: Final = "custom_rest"

DEFAULT_ACTION_VALUES: Final[Mapping[str, PolicyAction]] = MappingProxyType(
    {
        "block": PolicyAction.BLOCK,
        "deny": PolicyAction.BLOCK,
        "redact": PolicyAction.REDACT,
        "mask": PolicyAction.MASK,
        "warn": PolicyAction.WARN,
        "alert": PolicyAction.WARN,
        "audit": PolicyAction.AUDIT,
        "log": PolicyAction.AUDIT,
        "allow": PolicyAction.ALLOW,
    }
)


class RestTransport(Protocol):
    async def get_json(self, url: str) -> object: ...


@dataclass(frozen=True, slots=True)
class RestFieldMap:
    id_field: str = "id"
    name_field: str = "name"
    severity_field: str | None = "severity"
    action_field: str | None = "action"
    version_field: str | None = "version"
    pattern_field: str | None = "pattern"
    keywords_field: str | None = "keywords"
    data_classes_field: str | None = "data_classes"
    action_values: Mapping[str, PolicyAction] = field(default_factory=lambda: DEFAULT_ACTION_VALUES)

    @property
    def consumed_fields(self) -> frozenset[str]:
        return frozenset(
            name
            for name in (
                self.id_field,
                self.name_field,
                self.severity_field,
                self.action_field,
                self.version_field,
                self.pattern_field,
                self.keywords_field,
                self.data_classes_field,
            )
            if name is not None
        )


class CustomRestAdapter(DLPProviderAdapter):
    provider = PROVIDER

    def __init__(
        self,
        settings: ConnectionSettings,
        transport: RestTransport,
        endpoints: RestEndpoints | None = None,
        field_map: RestFieldMap | None = None,
        capabilities: frozenset[Capability] | None = None,
    ) -> None:
        self._settings: Final = settings
        self._transport: Final = transport
        self._endpoints: Final = endpoints if endpoints is not None else RestEndpoints()
        self._field_map: Final = field_map if field_map is not None else RestFieldMap()
        self._capabilities: Final = (
            capabilities if capabilities is not None else frozenset({Capability.POLICY_LIST, Capability.POLICY_READ})
        )

    async def test_connection(self) -> ConnectorHealth:
        try:
            await self._transport.get_json(self._url(self._endpoints.health))
        except Exception as err:  # noqa: BLE001  # a transport failure is a health answer, not a crash
            return ConnectorHealth(healthy=False, detail=str(err)[:400])
        return ConnectorHealth(healthy=True, detail="reachable")

    async def get_capabilities(self) -> frozenset[Capability]:
        return self._capabilities

    async def list_policies(self, since: datetime | None = None) -> tuple[VendorPolicyRef, ...]:
        self._require(Capability.POLICY_LIST)
        payload: Final = await self._transport.get_json(self._url(self._endpoints.policy_list))
        entries: Final = _as_entries(payload)
        return tuple(
            VendorPolicyRef(
                external_policy_id=_string(entry, self._field_map.id_field) or "",
                name=_string(entry, self._field_map.name_field) or "",
                external_policy_version=(
                    _string(entry, self._field_map.version_field) if self._field_map.version_field else None
                ),
            )
            for entry in entries
            if _string(entry, self._field_map.id_field)
        )

    async def get_policy(self, external_policy_id: str) -> Mapping[str, object]:
        self._require(Capability.POLICY_READ)
        payload: Final = await self._transport.get_json(self._url(self._endpoints.policy_read_for(external_policy_id)))
        if not isinstance(payload, Mapping):
            raise ValueError(f"{PROVIDER}: policy read returned a non-object payload")
        return payload

    async def list_classifiers(self) -> tuple[VendorClassifier, ...]:
        self._require(Capability.CLASSIFIER_LIST)
        payload: Final = await self._transport.get_json(self._url(self._endpoints.classifier_list))
        return tuple(
            VendorClassifier(
                vendor_classifier_id=_string(entry, "id") or "",
                name=_string(entry, "name") or "",
                description=_string(entry, "description"),
            )
            for entry in _as_entries(payload)
            if _string(entry, "id")
        )

    def map_to_witdps(self, raw: Mapping[str, object]) -> MappingResult:
        field_map: Final = self._field_map
        name: Final = _string(raw, field_map.name_field) or _string(raw, field_map.id_field)
        if not name:
            return MappingResult(canonical=None, unmapped=(), errors=("vendor policy has neither name nor id",))
        action, action_error = self._map_action(raw)
        conditions: Final = self._map_conditions(raw)
        if not conditions:
            return MappingResult(
                canonical=None,
                unmapped=self._unmapped(raw),
                errors=(*((action_error,) if action_error else ()), "vendor policy yielded no usable condition"),
            )
        if action_error is not None:
            return MappingResult(canonical=None, unmapped=self._unmapped(raw), errors=(action_error,))
        document: Final[Mapping[str, object]] = {  # mutable-ok: WIT-DPS document, validated before use
            "wit_dps_version": WIT_DPS_VERSION,
            "name": name,
            "mode": FederationMode.MIRROR.value,
            "severity": _string(raw, field_map.severity_field or "") or "medium",
            "condition": {  # mutable-ok: WIT-DPS document, validated before use
                "operator": "ANY",
                "conditions": list(conditions),  # mutable-ok: WIT-DPS document, validated before use
            },  # mutable-ok: WIT-DPS document, validated before use
            "actions": {"on_match": action.value},  # mutable-ok: WIT-DPS document, validated before use
            "source": {  # mutable-ok: WIT-DPS document, validated before use
                "vendor": PROVIDER,
                "connection_id": self._settings.connection_id,
                "external_policy_id": _string(raw, field_map.id_field) or name,
                "unmapped": list(self._unmapped(raw)),  # mutable-ok: WIT-DPS document, validated before use
            },
        }
        parsed: Final = parse_policy(document)
        if isinstance(parsed, PolicyValidationFailure):
            return MappingResult(canonical=None, unmapped=self._unmapped(raw), errors=parsed.errors)
        return MappingResult(canonical=document, unmapped=self._unmapped(raw), errors=())

    def _map_action(self, raw: Mapping[str, object]) -> tuple[PolicyAction, str | None]:
        field_name: Final = self._field_map.action_field
        if field_name is None:
            return PolicyAction.AUDIT, None
        vendor_action: Final = _string(raw, field_name)
        if vendor_action is None:
            return PolicyAction.AUDIT, None
        mapped: Final = self._field_map.action_values.get(vendor_action.strip().lower())
        if mapped is None:
            return PolicyAction.AUDIT, f"unrecognised vendor action {vendor_action!r}; refusing to guess an action"
        return mapped, None

    def _map_conditions(self, raw: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
        field_map: Final = self._field_map
        pattern: Final = _string(raw, field_map.pattern_field or "")
        keywords: Final = _string_list(raw, field_map.keywords_field or "")
        data_classes: Final = _string_list(raw, field_map.data_classes_field or "")
        return (
            *(
                ({"type": "regex", "pattern": pattern},)  # mutable-ok: WIT-DPS document, validated before use
                if pattern
                else ()  # mutable-ok: WIT-DPS document, validated before use
            ),  # mutable-ok: WIT-DPS document, validated before use
            *(
                ({"type": "keyword", "terms": list(keywords)},)  # mutable-ok: WIT-DPS document, validated before use
                if keywords
                else ()  # mutable-ok: WIT-DPS document, validated before use
            ),  # mutable-ok: WIT-DPS document, validated before use
            *tuple(
                {"type": "data_class", "class": data_class}  # mutable-ok: WIT-DPS document, validated before use
                for data_class in data_classes  # mutable-ok: WIT-DPS document, validated before use
            ),  # mutable-ok: WIT-DPS document, validated before use
        )

    def _unmapped(self, raw: Mapping[str, object]) -> tuple[str, ...]:
        return tuple(sorted(key for key in raw if key not in self._field_map.consumed_fields))

    def _require(self, capability: Capability) -> None:
        if capability not in self._capabilities:
            raise UnsupportedCapability(PROVIDER, capability)

    def _url(self, path: str) -> str:
        return f"{self._settings.base_url.rstrip('/')}{path}"


def _as_entries(payload: object) -> tuple[Mapping[str, object], ...]:
    if isinstance(payload, Mapping):
        inner: Final = payload.get("policies") or payload.get("items") or payload.get("data")
        return _as_entries(inner) if inner is not None else ()
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        return tuple(entry for entry in payload if isinstance(entry, Mapping))
    return ()


def _string(raw: Mapping[str, object], key: str) -> str | None:
    if not key:
        return None
    value: Final = raw.get(key)
    return value if isinstance(value, str) and value else None


def _string_list(raw: Mapping[str, object], key: str) -> tuple[str, ...]:
    if not key:
        return ()
    value: Final = raw.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(entry for entry in value if isinstance(entry, str) and entry)
