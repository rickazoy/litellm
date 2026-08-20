"""Adapter contract, capability discovery, RBAC and privacy toggles (§2.3, §2.6, §2.11, §0.5).

No network. The REST adapter's transport is injected, which is the whole reason
its untrusted-input handling can be tested at all.
"""

from __future__ import annotations

from typing import Mapping

import pytest

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.witos.policy_fabric.canonical import PolicyValidationFailure, parse_policy
from litellm.proxy.witos.policy_fabric.privacy import PrivacySettings, minimise_for_vendor
from litellm.proxy.witos.policy_fabric.providers import (
    ConnectionSettings,
    CustomRestAdapter,
    CustomWitDpsAdapter,
    InMemoryDocumentSource,
    UnsupportedCapability,
)
from litellm.proxy.witos.policy_fabric.providers.custom_rest import RestFieldMap
from litellm.proxy.witos.policy_fabric.rbac import DLPPermission, has_permission, require_permission, tenant_filter
from litellm.proxy.witos.policy_fabric.types import Capability

SETTINGS = ConnectionSettings(
    connection_id="conn-1",
    provider="custom_rest",
    base_url="https://vendor.invalid/api",
    secret_reference="WITOS_DLP_SECRET_VENDOR",
)


class StubTransport:
    def __init__(self, responses: Mapping[str, object]) -> None:
        self.responses = responses
        self.requested: list[str] = []

    async def get_json(self, url: str) -> object:
        self.requested.append(url)
        if url not in self.responses:
            raise ConnectionError(f"no stub for {url}")
        return self.responses[url]


def witdps_document() -> dict:
    return {
        "wit_dps_version": "2.0",
        "name": "PCI - Cardholder Data",
        "mode": "mirror",
        "condition": {"type": "regex", "pattern": r"\d{16}"},
        "actions": {"on_match": "BLOCK"},
        "source": {"vendor": "custom_witdps", "unmapped": ["vendor_only_field"]},
    }


async def test_the_witdps_adapter_advertises_only_what_it_can_do() -> None:
    adapter = CustomWitDpsAdapter(SETTINGS, InMemoryDocumentSource((witdps_document(),)))
    capabilities = await adapter.get_capabilities()
    assert capabilities == frozenset({Capability.POLICY_LIST, Capability.POLICY_READ})
    assert Capability.REALTIME_INPUT_EVALUATION not in capabilities


async def test_an_unsupported_capability_raises_rather_than_returning_a_fake() -> None:
    adapter = CustomWitDpsAdapter(SETTINGS, InMemoryDocumentSource(()))
    with pytest.raises(UnsupportedCapability) as raised:
        await adapter.evaluate_input("some content", "corr-1")
    assert raised.value.capability is Capability.REALTIME_INPUT_EVALUATION
    with pytest.raises(UnsupportedCapability):
        await adapter.list_classifiers()


async def test_the_witdps_adapter_validates_and_preserves_declared_unmapped_fields() -> None:
    adapter = CustomWitDpsAdapter(SETTINGS, InMemoryDocumentSource((witdps_document(),)))
    result = adapter.map_to_witdps(witdps_document())
    assert result.ok
    assert result.unmapped == ("vendor_only_field",)


async def test_the_witdps_adapter_refuses_an_invalid_document() -> None:
    adapter = CustomWitDpsAdapter(SETTINGS, InMemoryDocumentSource(()))
    result = adapter.map_to_witdps({"wit_dps_version": "2.0", "name": "x"})
    assert not result.ok
    assert result.errors


async def test_the_rest_adapter_reports_unknown_vendor_fields_rather_than_dropping_them() -> None:
    adapter = CustomRestAdapter(SETTINGS, StubTransport({}))
    result = adapter.map_to_witdps(
        {
            "id": "v-1",
            "name": "Cardholder",
            "action": "block",
            "pattern": r"\d{16}",
            "retention_policy": "90d",
            "owner_team": "grc",
        }
    )
    assert result.ok
    assert result.unmapped == ("owner_team", "retention_policy")
    assert result.canonical is not None
    parsed = parse_policy(result.canonical)
    assert not isinstance(parsed, PolicyValidationFailure)
    assert parsed.actions.on_match.value == "BLOCK"


async def test_the_rest_adapter_refuses_to_guess_an_unrecognised_action() -> None:
    adapter = CustomRestAdapter(SETTINGS, StubTransport({}))
    result = adapter.map_to_witdps(
        {"id": "v-1", "name": "Odd", "action": "quarantine_and_page_the_ciso", "pattern": r"\d{4}"}
    )
    assert not result.ok
    assert "refusing to guess" in result.errors[0]


async def test_the_rest_adapter_refuses_a_policy_with_no_usable_condition() -> None:
    adapter = CustomRestAdapter(SETTINGS, StubTransport({}))
    result = adapter.map_to_witdps({"id": "v-1", "name": "Empty", "action": "block"})
    assert not result.ok


async def test_the_rest_adapter_only_calls_endpoints_from_its_constants_module() -> None:
    transport = StubTransport({"https://vendor.invalid/api/policies": [{"id": "a", "name": "A"}]})
    adapter = CustomRestAdapter(SETTINGS, transport)
    refs = await adapter.list_policies()
    assert transport.requested == ["https://vendor.invalid/api/policies"]
    assert [ref.external_policy_id for ref in refs] == ["a"]


async def test_the_rest_adapter_refuses_a_capability_it_was_not_granted() -> None:
    adapter = CustomRestAdapter(SETTINGS, StubTransport({}), capabilities=frozenset({Capability.POLICY_READ}))
    with pytest.raises(UnsupportedCapability):
        await adapter.list_policies()


async def test_a_custom_field_map_changes_what_is_consumed_and_what_is_unmapped() -> None:
    adapter = CustomRestAdapter(
        SETTINGS,
        StubTransport({}),
        field_map=RestFieldMap(id_field="ref", name_field="title", action_field="verdict"),
    )
    result = adapter.map_to_witdps(
        {"ref": "r1", "title": "T", "verdict": "redact", "pattern": r"\d{4}", "id": "ignored"}
    )
    assert result.ok
    assert "id" in result.unmapped


async def test_a_connection_health_failure_is_a_value_not_an_exception() -> None:
    adapter = CustomRestAdapter(SETTINGS, StubTransport({}))
    health = await adapter.test_connection()
    assert health.healthy is False
    assert health.detail


def _key(scopes: list[str] | None = None, role: LitellmUserRoles | None = None) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="sk-test",
        user_role=role,
        metadata={"scopes": scopes} if scopes is not None else {},
    )


def test_a_secops_key_gets_dlp_scopes_and_a_bare_key_gets_none() -> None:
    secops = _key(["dlp:view", "dlp:enforce"])
    assert has_permission(secops, DLPPermission.VIEW)
    assert has_permission(secops, DLPPermission.ENFORCE)
    assert not has_permission(secops, DLPPermission.APPROVE)
    assert not has_permission(_key(), DLPPermission.VIEW)


def test_a_finops_key_has_no_dlp_access() -> None:
    cfo = _key(["finops:view", "finops:run_scenarios", "finops:manage_alerts"])
    for permission in DLPPermission:
        assert not has_permission(cfo, permission), permission


def test_the_dlp_wildcard_grants_every_dlp_scope() -> None:
    for permission in DLPPermission:
        assert has_permission(_key(["dlp:*"]), permission)


def test_a_proxy_admin_bypasses_the_scope_check() -> None:
    admin = _key(role=LitellmUserRoles.PROXY_ADMIN)
    for permission in DLPPermission:
        assert has_permission(admin, permission)


def test_a_view_only_admin_can_read_but_not_enforce() -> None:
    viewer = _key(role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)
    assert has_permission(viewer, DLPPermission.VIEW)
    assert has_permission(viewer, DLPPermission.VIEW_DECISIONS)
    assert not has_permission(viewer, DLPPermission.ENFORCE)
    assert not has_permission(viewer, DLPPermission.APPROVE)


def test_require_permission_raises_a_403_naming_the_missing_scope() -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as raised:
        require_permission(_key(), DLPPermission.IMPORT)
    assert raised.value.status_code == 403
    assert "dlp:import" in str(raised.value.detail)


def test_the_tenant_filter_clamps_a_non_admin_to_its_own_organization() -> None:
    scoped_key = UserAPIKeyAuth(api_key="sk-test", org_id="org-1")
    assert tenant_filter(scoped_key) == {"organization_id": "org-1"}
    assert tenant_filter(UserAPIKeyAuth(api_key="sk-test")) == {"organization_id": "__no_tenant__"}
    assert tenant_filter(_key(role=LitellmUserRoles.PROXY_ADMIN)) is None


def test_privacy_defaults_are_the_conservative_ones() -> None:
    settings = PrivacySettings()
    assert settings.send_full_content_to_provider is False
    assert settings.send_identity is False
    assert settings.store_local_content is False


def test_privacy_toggles_shape_what_leaves_the_process() -> None:
    strict = minimise_for_vendor("x" * 5000, "corr-1", "u1", "crm", PrivacySettings())
    assert len(strict.content) == 200
    assert strict.identity is None
    assert strict.application is None
    assert strict.correlation_id == "corr-1"

    permissive = minimise_for_vendor(
        "x" * 5000,
        "corr-1",
        "u1",
        "crm",
        PrivacySettings(send_full_content_to_provider=True, send_identity=True, send_application_id=True),
    )
    assert len(permissive.content) == 5000
    assert permissive.identity == "u1"
    assert permissive.application == "crm"


def test_privacy_settings_round_trip_through_json() -> None:
    original = PrivacySettings(send_identity=True, store_vendor_findings=True)
    assert PrivacySettings.from_json(original.to_json()) == original
    assert PrivacySettings.from_json(None) == PrivacySettings()
    assert PrivacySettings.from_json({"send_identity": "yes-please"}) == PrivacySettings()
