"""Direct WIT-DPS adapter (§2.6, adapter 1).

The universal escape hatch: a customer, an n8n flow or a script hands us WIT-DPS
JSON, either posted to the import endpoint or pulled from a URL. There is no
translation step, so the only work is validation, and validation is not a
formality here. A document arriving on this path is exactly as untrusted as one
arriving from a vendor API.

Capabilities are POLICY_LIST and POLICY_READ, and nothing else. No runtime
evaluation is claimed, because none exists.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Protocol

from litellm.proxy.witos.policy_fabric.canonical import PolicyValidationFailure, parse_policy
from litellm.proxy.witos.policy_fabric.providers.base import (
    ConnectionSettings,
    ConnectorHealth,
    DLPProviderAdapter,
    MappingResult,
    VendorPolicyRef,
)
from litellm.proxy.witos.policy_fabric.types import Capability

PROVIDER: Final = "custom_witdps"
_CAPABILITIES: Final[frozenset[Capability]] = frozenset({Capability.POLICY_LIST, Capability.POLICY_READ})


class PolicyDocumentSource(Protocol):
    """Where documents come from. Injected so the adapter is testable offline."""

    async def fetch_documents(self) -> tuple[Mapping[str, object], ...]: ...


@dataclass(frozen=True, slots=True)
class InMemoryDocumentSource:
    documents: tuple[Mapping[str, object], ...]

    async def fetch_documents(self) -> tuple[Mapping[str, object], ...]:
        return self.documents


class CustomWitDpsAdapter(DLPProviderAdapter):
    provider = PROVIDER

    def __init__(self, settings: ConnectionSettings, source: PolicyDocumentSource) -> None:
        self._settings: Final = settings
        self._source: Final = source

    async def test_connection(self) -> ConnectorHealth:
        try:
            documents: Final = await self._source.fetch_documents()
        except Exception as err:  # noqa: BLE001  # a source failure is a health answer, not a crash
            return ConnectorHealth(healthy=False, detail=f"document source unavailable: {err}")
        return ConnectorHealth(healthy=True, detail=f"{len(documents)} WIT-DPS document(s) available")

    async def get_capabilities(self) -> frozenset[Capability]:
        return _CAPABILITIES

    async def list_policies(self, since: datetime | None = None) -> tuple[VendorPolicyRef, ...]:
        documents: Final = await self._source.fetch_documents()
        return tuple(_to_ref(document, index) for index, document in enumerate(documents))

    async def get_policy(self, external_policy_id: str) -> Mapping[str, object]:
        documents: Final = await self._source.fetch_documents()
        for index, document in enumerate(documents):
            if _external_id(document, index) == external_policy_id:
                return document
        raise KeyError(external_policy_id)

    def map_to_witdps(self, raw: Mapping[str, object]) -> MappingResult:
        parsed: Final = parse_policy(raw)
        if isinstance(parsed, PolicyValidationFailure):
            return MappingResult(canonical=None, unmapped=(), errors=parsed.errors)
        return MappingResult(canonical=raw, unmapped=_declared_unmapped(raw), errors=())


def _declared_unmapped(raw: Mapping[str, object]) -> tuple[str, ...]:
    source: Final = raw.get("source")
    if not isinstance(source, Mapping):
        return ()
    unmapped: Final = source.get("unmapped")
    if not isinstance(unmapped, Sequence) or isinstance(unmapped, (str, bytes)):
        return ()
    return tuple(entry for entry in unmapped if isinstance(entry, str))


def _external_id(document: Mapping[str, object], index: int) -> str:
    candidate: Final = document.get("policy_id")
    if isinstance(candidate, str) and candidate:
        return candidate
    name: Final = document.get("name")
    return name if isinstance(name, str) and name else f"witdps-{index}"


def _to_ref(document: Mapping[str, object], index: int) -> VendorPolicyRef:
    name: Final = document.get("name")
    return VendorPolicyRef(
        external_policy_id=_external_id(document, index),
        name=name if isinstance(name, str) else f"witdps-{index}",
        external_policy_version=str(document.get("wit_dps_version", "2.0")),
    )
