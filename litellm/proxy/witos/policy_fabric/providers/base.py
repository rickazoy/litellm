"""Vendor adapter contract (§2.3, §2.6).

Capability discovery is the whole design. An adapter declares what its tenant's
API actually supports and the rest of the system obeys that answer: unsupported
features are hidden in the UI and raise `UnsupportedCapability` in code. They are
never stubbed with a plausible-looking fake, because a fake capability is a
silent gap in coverage that only shows up during an incident.

`map_to_witdps` is the parsing security boundary. Vendor documents are untrusted
input in full: unknown fields are reported in `MappingResult.unmapped` rather
than dropped, so an operator can see exactly what did not survive translation.

Credentials never live in this layer. An adapter holds a `secret_reference` and
resolves it through the proxy's secret manager at call time; nothing here can
return a secret to an API caller.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Final

from litellm.proxy.witos.policy_fabric.types import Capability, PolicyAction


class UnsupportedCapability(Exception):
    """Raised when a caller asks an adapter for something its tenant cannot do."""

    def __init__(self, provider: str, capability: Capability) -> None:
        super().__init__(f"{provider} does not support {capability.value}")
        self.provider: Final = provider
        self.capability: Final = capability


@dataclass(frozen=True, slots=True)
class ConnectorHealth:
    healthy: bool
    detail: str
    latency_ms: int | None = None


@dataclass(frozen=True, slots=True)
class VendorPolicyRef:
    external_policy_id: str
    name: str
    external_policy_version: str | None = None
    updated_at: datetime | None = None
    external_url: str | None = None


@dataclass(frozen=True, slots=True)
class VendorClassifier:
    vendor_classifier_id: str
    name: str
    description: str | None = None
    confidence_scale: float = 1.0


@dataclass(frozen=True, slots=True)
class VendorDecision:
    action: PolicyAction
    matched_policy_ids: tuple[str, ...] = ()
    vendor_request_id: str | None = None
    risk_score: float | None = None


@dataclass(frozen=True, slots=True)
class MappingResult:
    canonical: Mapping[str, object] | None
    unmapped: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.canonical is not None and not self.errors


@dataclass(frozen=True, slots=True)
class SyncStats:
    fetched: int = 0
    created: int = 0
    updated: int = 0
    unchanged: int = 0
    stale_marked: int = 0
    failed_mapping: int = 0
    pending_review: int = 0


@dataclass(frozen=True, slots=True)
class ConnectionSettings:
    connection_id: str
    provider: str
    base_url: str
    secret_reference: str
    timeout_ms: int = 800
    region: str | None = None
    tenant_external_id: str | None = None


class DLPProviderAdapter(ABC):
    """Immutable adapter contract.

    Collections are `frozenset`/`tuple` rather than `set`/`list`: a capability
    set a caller can mutate is a capability set a caller can grant itself.
    """

    provider: str

    @abstractmethod
    async def test_connection(self) -> ConnectorHealth: ...

    @abstractmethod
    async def get_capabilities(self) -> frozenset[Capability]: ...

    async def list_policies(self, since: datetime | None = None) -> tuple[VendorPolicyRef, ...]:
        raise UnsupportedCapability(self.provider, Capability.POLICY_LIST)

    async def get_policy(self, external_policy_id: str) -> Mapping[str, object]:
        raise UnsupportedCapability(self.provider, Capability.POLICY_READ)

    async def list_classifiers(self) -> tuple[VendorClassifier, ...]:
        raise UnsupportedCapability(self.provider, Capability.CLASSIFIER_LIST)

    async def evaluate_input(self, content: str, correlation_id: str) -> VendorDecision:
        raise UnsupportedCapability(self.provider, Capability.REALTIME_INPUT_EVALUATION)

    async def evaluate_output(self, content: str, correlation_id: str) -> VendorDecision:
        raise UnsupportedCapability(self.provider, Capability.REALTIME_OUTPUT_EVALUATION)

    @abstractmethod
    def map_to_witdps(self, raw: Mapping[str, object]) -> MappingResult: ...

    async def sync(self) -> SyncStats:
        raise UnsupportedCapability(self.provider, Capability.POLICY_LIST)


def resolve_secret(secret_reference: str) -> str | None:
    """Resolve a secret reference through the proxy's secret manager.

    The reference is what is stored and what the API returns. The value is
    resolved at call time and never leaves this process.
    """
    from litellm.secret_managers.main import get_secret_str

    return get_secret_str(secret_reference)
