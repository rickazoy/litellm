"""Vendor adapters. Phase 4 ships the custom pair only.

Cyera, Purview, Nightfall and Netskope are a later work package by design: the
abstraction is proved against the escape hatch first, so every vendor after it
is an adapter rather than an integration project.
"""

from typing import Final

from litellm.proxy.witos.policy_fabric.providers.base import (
    ConnectionSettings,
    ConnectorHealth,
    DLPProviderAdapter,
    MappingResult,
    SyncStats,
    UnsupportedCapability,
    VendorClassifier,
    VendorDecision,
    VendorPolicyRef,
)
from litellm.proxy.witos.policy_fabric.providers.custom_rest import CustomRestAdapter, RestFieldMap
from litellm.proxy.witos.policy_fabric.providers.custom_witdps import (
    CustomWitDpsAdapter,
    InMemoryDocumentSource,
)

PROVIDER_REGISTRY: Final = (CustomWitDpsAdapter.provider, CustomRestAdapter.provider)

__all__ = (
    "PROVIDER_REGISTRY",
    "ConnectionSettings",
    "ConnectorHealth",
    "CustomRestAdapter",
    "CustomWitDpsAdapter",
    "DLPProviderAdapter",
    "InMemoryDocumentSource",
    "MappingResult",
    "RestFieldMap",
    "SyncStats",
    "UnsupportedCapability",
    "VendorClassifier",
    "VendorDecision",
    "VendorPolicyRef",
)
