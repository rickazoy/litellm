"""`/witos/dlp/*` API (§2.12).

Three rules shape this surface.

Secrets are references. A connection stores and returns `secret_reference`, the
name of a secret manager key. The resolved value is never a field, never a
response, and never logged; the API reports only whether the reference resolves.

Imports land in shadow. `POST /policies/import` writes `status=shadow` for every
document regardless of any `auto_apply` intent, and `POST /policies/{id}/activate`
is the only path to enforcement. A policy whose action is BLOCK or
REQUIRE_APPROVAL additionally needs `dlp:approve`, so the person who can import
a policy is not automatically the person who can start refusing traffic with it.

Receipts are masked at write time, so `GET /decisions` has nothing to redact.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Final

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.witos.policy_fabric.canonical import (
    PolicyValidationFailure,
    canonical_hash,
    parse_policy,
)
from litellm.proxy.witos.policy_fabric.classifier_registry import (
    CANONICAL_CLASSES,
    CANONICAL_TO_PRESIDIO_ENTITY,
)
from litellm.proxy.witos.policy_fabric.compiler import CompileFailure, compile_policy
from litellm.proxy.witos.policy_fabric.evaluator import RequestScope
from litellm.proxy.witos.policy_fabric.policy_cache import (
    PolicySetCache,
    coordination_redis_cache,
    publish_policy_invalidation,
)
from litellm.proxy.witos.policy_fabric.privacy import PrivacySettings
from litellm.proxy.witos.policy_fabric.providers.base import resolve_secret
from litellm.proxy.witos.policy_fabric.rbac import DLPPermission, require_permission, tenant_filter
from litellm.proxy.witos.policy_fabric.receipts import summarise_findings
from litellm.proxy.witos.policy_fabric.runtime import (
    CompiledPolicyStore,
    EngineConfig,
    EvaluationRequest,
    PolicyFabricEngine,
)
from litellm.proxy.witos.policy_fabric.scope import ScopedPolicy
from litellm.proxy.witos.policy_fabric.store import DLPTables, load_policy_set, tables_for
from litellm.proxy.witos.policy_fabric.streaming import disclaimer_for
from litellm.proxy.witos.policy_fabric.types import (
    HUMAN_APPROVAL_ACTIONS,
    Capability,
    EvaluationDirection,
    PolicyAction,
    PolicyStatus,
    StreamingMode,
)

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

router: Final = APIRouter(prefix="/witos/dlp", tags=["WIT OS DLP"])  # mutable-ok: dict-shaped JSON body

_MAX_PAGE_SIZE: Final = 500
_SUPPORTED_PROVIDERS: Final[tuple[str, ...]] = ("custom_witdps", "custom_rest")


class ConnectionCreateRequest(BaseModel):
    provider: str
    name: str
    base_url: str
    auth_type: str = "api_key"
    secret_reference: str
    description: str | None = None
    region: str | None = None
    tenant_external_id: str | None = None
    sync_mode: str = "manual_approve"
    sync_interval_min: int = 60
    fail_mode: str = "fail_open"
    timeout_ms: int = 800
    privacy: dict[str, bool] | None = None  # mutable-ok: dict-shaped JSON body
    enabled: bool = True
    organization_id: str | None = None


class ConnectionUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    base_url: str | None = None
    secret_reference: str | None = None
    sync_mode: str | None = None
    sync_interval_min: int | None = None
    fail_mode: str | None = None
    timeout_ms: int | None = None
    privacy: dict[str, bool] | None = None  # mutable-ok: dict-shaped JSON body
    enabled: bool | None = None


class PolicyImportRequest(BaseModel):
    documents: list[dict[str, object]] = Field(min_length=1, max_length=200)  # mutable-ok: dict-shaped JSON body
    connection_id: str | None = None
    organization_id: str | None = None
    change_note: str | None = None


class TestBenchRequest(BaseModel):
    content: str
    direction: EvaluationDirection = EvaluationDirection.INPUT
    organization_id: str | None = None
    team_id: str | None = None
    user_id: str | None = None
    key_alias: str | None = None
    application: str | None = None
    model: str | None = None
    model_group: str | None = None
    tool_name: str | None = None
    tool_arguments: dict[str, str] | None = None  # mutable-ok: dict-shaped JSON body


def _require_prisma() -> PrismaClient:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail=CommonProxyErrors.db_not_connected_error.value)
    return prisma_client


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _merged_where(base: Mapping[str, object], caller: UserAPIKeyAuth) -> Mapping[str, object]:
    scoped: Final = tenant_filter(caller)
    return base if scoped is None else {**base, **scoped}  # mutable-ok: dict-shaped JSON body


def _connection_view(record: object) -> Mapping[str, object]:
    """Never echoes a secret. `secret_reference` is a key name, not a credential."""
    reference: Final = getattr(record, "secret_reference", "")
    return {  # mutable-ok: dict-shaped JSON body
        "connection_id": getattr(record, "connection_id", None),
        "provider": getattr(record, "provider", None),
        "name": getattr(record, "name", None),
        "description": getattr(record, "description", None),
        "base_url": getattr(record, "base_url", None),
        "region": getattr(record, "region", None),
        "tenant_external_id": getattr(record, "tenant_external_id", None),
        "auth_type": getattr(record, "auth_type", None),
        "secret_reference": reference,
        "secret_resolves": bool(resolve_secret(reference)) if reference else False,
        "capabilities": _json_field(record, "capabilities_json"),
        "sync_mode": getattr(record, "sync_mode", None),
        "sync_interval_min": getattr(record, "sync_interval_min", None),
        "fail_mode": getattr(record, "fail_mode", None),
        "timeout_ms": getattr(record, "timeout_ms", None),
        "privacy": _json_field(record, "privacy_json"),
        "enabled": getattr(record, "enabled", None),
        "status": getattr(record, "status", None),
        "last_sync_at": getattr(record, "last_sync_at", None),
        "last_error": getattr(record, "last_error", None),
        "organization_id": getattr(record, "organization_id", None),
    }


def _policy_view(record: object) -> Mapping[str, object]:
    return {  # mutable-ok: dict-shaped JSON body
        "policy_id": getattr(record, "policy_id", None),
        "name": getattr(record, "name", None),
        "description": getattr(record, "description", None),
        "source": getattr(record, "source", None),
        "connection_id": getattr(record, "connection_id", None),
        "external_policy_id": getattr(record, "external_policy_id", None),
        "external_policy_version": getattr(record, "external_policy_version", None),
        "mode": getattr(record, "mode", None),
        "status": getattr(record, "status", None),
        "priority": getattr(record, "priority", None),
        "direction": getattr(record, "direction", None),
        "canonical_policy_hash": getattr(record, "canonical_policy_hash", None),
        "compile_status": getattr(record, "compile_status", None),
        "compile_error": getattr(record, "compile_error", None),
        "approved_by": getattr(record, "approved_by", None),
        "activated_at": getattr(record, "activated_at", None),
        "organization_id": getattr(record, "organization_id", None),
        "canonical_policy_json": _json_field(record, "canonical_policy_json"),
    }


def _json_field(record: object, name: str) -> object:
    raw: Final = getattr(record, name, None)
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
    return raw


@router.get("/providers")
async def list_providers(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Mapping[str, object]:
    require_permission(user_api_key_dict, DLPPermission.VIEW)
    return {  # mutable-ok: dict-shaped JSON body
        "providers": [  # mutable-ok: dict-shaped JSON body
            {  # mutable-ok: dict-shaped JSON body
                "provider": "custom_witdps",
                "capabilities": [  # mutable-ok: dict-shaped JSON body
                    Capability.POLICY_LIST.value,
                    Capability.POLICY_READ.value,
                ],  # mutable-ok: dict-shaped JSON body
                "modes": ["mirror", "observe"],  # mutable-ok: dict-shaped JSON body
            },
            {  # mutable-ok: dict-shaped JSON body
                "provider": "custom_rest",
                "capabilities": [  # mutable-ok: dict-shaped JSON body
                    Capability.POLICY_LIST.value,
                    Capability.POLICY_READ.value,
                ],  # mutable-ok: dict-shaped JSON body
                "modes": ["mirror", "observe"],  # mutable-ok: dict-shaped JSON body
            },
        ],
        "note": (
            "Capability lists are discovered per connection. Vendor adapters beyond the custom pair "
            "are a later work package and are deliberately absent rather than stubbed."
        ),
        "streaming_modes": [  # mutable-ok: dict-shaped JSON body
            {  # mutable-ok: dict-shaped JSON body
                "mode": mode.value,
                "prevents_disclosure": mode.prevents_disclosure,
                "disclaimer": disclaimer_for(mode),
            }
            for mode in StreamingMode
        ],
    }


@router.get("/classifiers")
async def list_classifiers(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Mapping[str, object]:
    require_permission(user_api_key_dict, DLPPermission.VIEW)
    return {  # mutable-ok: dict-shaped JSON body
        "canonical_classes": sorted(CANONICAL_CLASSES),
        "locally_resolvable": sorted(CANONICAL_TO_PRESIDIO_ENTITY),
        "custom_class_format": "CUSTOM.<vendor>.<name>",
    }


@router.get("/connections")
async def list_connections(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Mapping[str, object]:
    require_permission(user_api_key_dict, DLPPermission.VIEW)
    tables: Final = tables_for(_require_prisma())
    records: Final = await tables.connections.find_many(
        where=_merged_where({}, user_api_key_dict)  # mutable-ok: dict-shaped JSON body
    )  # mutable-ok: dict-shaped JSON body
    return {"connections": [_connection_view(record) for record in records]}  # mutable-ok: dict-shaped JSON body


@router.post("/connections")
async def create_connection(
    payload: ConnectionCreateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Mapping[str, object]:
    require_permission(user_api_key_dict, DLPPermission.MANAGE_CONNECTIONS)
    if payload.provider not in _SUPPORTED_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail={  # mutable-ok: dict-shaped JSON body
                "error": f"unsupported provider {payload.provider!r}",
                "supported": list(_SUPPORTED_PROVIDERS),  # mutable-ok: dict-shaped JSON body
            },  # mutable-ok: dict-shaped JSON body
        )
    tables: Final = tables_for(_require_prisma())
    privacy: Final = PrivacySettings.from_json(payload.privacy)
    record: Final = await tables.connections.create(
        data={  # mutable-ok: dict-shaped JSON body
            "connection_id": str(uuid.uuid4()),
            "provider": payload.provider,
            "name": payload.name,
            "description": payload.description,
            "base_url": payload.base_url,
            "region": payload.region,
            "tenant_external_id": payload.tenant_external_id,
            "auth_type": payload.auth_type,
            "secret_reference": payload.secret_reference,
            "sync_mode": payload.sync_mode,
            "sync_interval_min": payload.sync_interval_min,
            "fail_mode": payload.fail_mode,
            "timeout_ms": payload.timeout_ms,
            "privacy_json": json.dumps(privacy.to_json()),
            "enabled": payload.enabled,
            "organization_id": payload.organization_id or user_api_key_dict.org_id,
            "created_by": user_api_key_dict.user_id or "unknown",
            "updated_at": _now(),
        }
    )
    return _connection_view(record)


@router.get("/connections/{connection_id}")
async def get_connection(
    connection_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Mapping[str, object]:
    require_permission(user_api_key_dict, DLPPermission.VIEW)
    record: Final = await _connection_or_404(connection_id, user_api_key_dict)
    return _connection_view(record)


@router.patch("/connections/{connection_id}")
async def update_connection(
    connection_id: str,
    payload: ConnectionUpdateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Mapping[str, object]:
    require_permission(user_api_key_dict, DLPPermission.MANAGE_CONNECTIONS)
    await _connection_or_404(connection_id, user_api_key_dict)
    tables: Final = tables_for(_require_prisma())
    changes: Final = {  # mutable-ok: dict-shaped JSON body
        key: value
        for key, value in (
            ("name", payload.name),
            ("description", payload.description),
            ("base_url", payload.base_url),
            ("secret_reference", payload.secret_reference),
            ("sync_mode", payload.sync_mode),
            ("sync_interval_min", payload.sync_interval_min),
            ("fail_mode", payload.fail_mode),
            ("timeout_ms", payload.timeout_ms),
            ("enabled", payload.enabled),
            (
                "privacy_json",
                json.dumps(PrivacySettings.from_json(payload.privacy).to_json())
                if payload.privacy is not None
                else None,
            ),
        )
        if value is not None
    }
    record: Final = await tables.connections.update(
        where={"connection_id": connection_id},  # mutable-ok: dict-shaped JSON body
        data={**changes, "updated_at": _now()},  # mutable-ok: dict-shaped JSON body
    )
    return _connection_view(record)


@router.delete("/connections/{connection_id}")
async def delete_connection(
    connection_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Mapping[str, object]:
    require_permission(user_api_key_dict, DLPPermission.MANAGE_CONNECTIONS)
    await _connection_or_404(connection_id, user_api_key_dict)
    tables: Final = tables_for(_require_prisma())
    await tables.connections.delete(where={"connection_id": connection_id})  # mutable-ok: dict-shaped JSON body
    return {"deleted": connection_id}  # mutable-ok: dict-shaped JSON body


@router.post("/connections/{connection_id}/test")
async def test_connection(
    connection_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Mapping[str, object]:
    require_permission(user_api_key_dict, DLPPermission.MANAGE_CONNECTIONS)
    record: Final = await _connection_or_404(connection_id, user_api_key_dict)
    reference: Final = getattr(record, "secret_reference", "")
    return {  # mutable-ok: dict-shaped JSON body
        "connection_id": connection_id,
        "secret_resolves": bool(resolve_secret(reference)) if reference else False,
        "detail": (
            "Credential reference checked against the secret manager. Reachability testing "
            "requires a vendor adapter, which is a later work package."
        ),
    }


@router.post("/connections/{connection_id}/discover")
async def discover_capabilities(
    connection_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Mapping[str, object]:
    require_permission(user_api_key_dict, DLPPermission.MANAGE_CONNECTIONS)
    record: Final = await _connection_or_404(connection_id, user_api_key_dict)
    provider: Final = getattr(record, "provider", "")
    capabilities: Final = (Capability.POLICY_LIST.value, Capability.POLICY_READ.value)
    tables: Final = tables_for(_require_prisma())
    await tables.connections.update(
        where={"connection_id": connection_id},  # mutable-ok: dict-shaped JSON body
        data={  # mutable-ok: dict-shaped JSON body
            "capabilities_json": json.dumps(list(capabilities)),  # mutable-ok: dict-shaped JSON body
            "updated_at": _now(),
        },  # mutable-ok: dict-shaped JSON body
    )
    return {  # mutable-ok: dict-shaped JSON body
        "connection_id": connection_id,
        "provider": provider,
        "capabilities": list(capabilities),  # mutable-ok: dict-shaped JSON body
    }  # mutable-ok: dict-shaped JSON body


@router.post("/connections/{connection_id}/sync")
async def sync_connection(
    connection_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Mapping[str, object]:
    """Declared by §2.12, implemented by the federation work package.

    Returning 501 rather than a fabricated success: a sync that reports "0
    policies changed" without having talked to anything is the worst possible
    answer, because it looks like confirmation that nothing drifted.
    """
    require_permission(user_api_key_dict, DLPPermission.MANAGE_CONNECTIONS)
    await _connection_or_404(connection_id, user_api_key_dict)
    raise HTTPException(
        status_code=501,
        detail={  # mutable-ok: dict-shaped JSON body
            "error": "policy sync requires the federation sync engine, which is a later work package",
            "available_now": "POST /witos/dlp/policies/import",
        },
    )


@router.get("/sync-runs")
async def list_sync_runs(
    connection_id: str | None = None,
    limit: int = Query(default=50, le=_MAX_PAGE_SIZE),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Mapping[str, object]:
    require_permission(user_api_key_dict, DLPPermission.VIEW)
    tables: Final = tables_for(_require_prisma())
    allowed: Final = await _allowed_connection_ids(tables, user_api_key_dict)
    if allowed is not None and connection_id is not None and connection_id not in allowed:
        raise HTTPException(
            status_code=404,
            detail={"error": "connection not found"},  # mutable-ok: dict-shaped JSON body
        )  # mutable-ok: dict-shaped JSON body
    where: Final[Mapping[str, object]] = (
        {"connection_id": connection_id}  # mutable-ok: dict-shaped JSON body
        if connection_id is not None
        else ({} if allowed is None else {"connection_id": {"in": list(allowed)}})  # mutable-ok: dict-shaped JSON body
    )
    runs: Final = await tables.sync_runs.find_many(
        where=where,
        order={"started_at": "desc"},  # mutable-ok: dict-shaped JSON body
        take=limit,  # mutable-ok: dict-shaped JSON body
    )  # mutable-ok: dict-shaped JSON body
    return {"sync_runs": [_sync_run_view(run) for run in runs]}  # mutable-ok: dict-shaped JSON body


def _sync_run_view(run: object) -> Mapping[str, object]:
    return {  # mutable-ok: dict-shaped JSON body
        "id": getattr(run, "id", None),
        "connection_id": getattr(run, "connection_id", None),
        "started_at": getattr(run, "started_at", None),
        "finished_at": getattr(run, "finished_at", None),
        "status": getattr(run, "status", None),
        "stats": _json_field(run, "stats"),
    }


@router.get("/policies")
async def list_policies(
    status: str | None = None,
    provider: str | None = None,
    limit: int = Query(default=100, le=_MAX_PAGE_SIZE),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Mapping[str, object]:
    require_permission(user_api_key_dict, DLPPermission.VIEW)
    tables: Final = tables_for(_require_prisma())
    filters: Final = {  # mutable-ok: dict-shaped JSON body
        key: value for key, value in (("status", status), ("source", provider)) if value is not None
    }  # mutable-ok: dict-shaped JSON body
    records: Final = await tables.policies.find_many(
        where=_merged_where(filters, user_api_key_dict),
        order={"priority": "desc"},  # mutable-ok: dict-shaped JSON body
        take=limit,
    )
    return {"policies": [_policy_view(record) for record in records]}  # mutable-ok: dict-shaped JSON body


@router.get("/policies/{policy_id}")
async def get_policy(
    policy_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Mapping[str, object]:
    require_permission(user_api_key_dict, DLPPermission.VIEW)
    return _policy_view(await _policy_or_404(policy_id, user_api_key_dict))


@router.get("/policies/{policy_id}/versions")
async def list_policy_versions(
    policy_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Mapping[str, object]:
    require_permission(user_api_key_dict, DLPPermission.VIEW)
    await _policy_or_404(policy_id, user_api_key_dict)
    tables: Final = tables_for(_require_prisma())
    rows: Final = await tables.versions.find_many(
        where={"policy_id": policy_id},  # mutable-ok: dict-shaped JSON body
        order={"version": "desc"},  # mutable-ok: dict-shaped JSON body
    )  # mutable-ok: dict-shaped JSON body
    return {"versions": [_version_view(row) for row in rows]}  # mutable-ok: dict-shaped JSON body


@router.get("/policies/{policy_id}/diff")
async def diff_policy_versions(
    policy_id: str,
    from_version: int = Query(alias="from"),
    to_version: int = Query(alias="to"),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Mapping[str, object]:
    require_permission(user_api_key_dict, DLPPermission.VIEW)
    await _policy_or_404(policy_id, user_api_key_dict)
    tables: Final = tables_for(_require_prisma())
    rows: Final = await tables.versions.find_many(
        where={  # mutable-ok: dict-shaped JSON body
            "policy_id": policy_id,
            "version": {"in": [from_version, to_version]},  # mutable-ok: dict-shaped JSON body
        }  # mutable-ok: dict-shaped JSON body
    )
    indexed: Final = {getattr(row, "version", None): row for row in rows}  # mutable-ok: dict-shaped JSON body
    if from_version not in indexed or to_version not in indexed:
        raise HTTPException(
            status_code=404,
            detail={"error": "one or both versions do not exist"},  # mutable-ok: dict-shaped JSON body
        )  # mutable-ok: dict-shaped JSON body
    return {  # mutable-ok: dict-shaped JSON body
        "policy_id": policy_id,
        "from": _version_view(indexed[from_version]),
        "to": _version_view(indexed[to_version]),
        "hash_changed": getattr(indexed[from_version], "hash", None) != getattr(indexed[to_version], "hash", None),
    }


def _version_view(row: object) -> Mapping[str, object]:
    return {  # mutable-ok: dict-shaped JSON body
        "version": getattr(row, "version", None),
        "hash": getattr(row, "hash", None),
        "change_note": getattr(row, "change_note", None),
        "created_by": getattr(row, "created_by", None),
        "created_at": getattr(row, "created_at", None),
        "canonical_json": _json_field(row, "canonical_json"),
    }


@router.post("/policies/import")
async def import_policies(
    payload: PolicyImportRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Mapping[str, object]:
    require_permission(user_api_key_dict, DLPPermission.IMPORT)
    tables: Final = tables_for(_require_prisma())
    organization_id: Final = payload.organization_id or user_api_key_dict.org_id
    results: Final = tuple(
        [
            await _import_one(tables, document, payload, organization_id, user_api_key_dict)
            for document in payload.documents
        ]
    )
    await _invalidate_policy_caches()
    return {  # mutable-ok: dict-shaped JSON body
        "imported": [  # mutable-ok: dict-shaped JSON body
            result for result in results if result.get("status") == PolicyStatus.SHADOW.value
        ],  # mutable-ok: dict-shaped JSON body
        "rejected": [  # mutable-ok: dict-shaped JSON body
            result for result in results if result.get("status") != PolicyStatus.SHADOW.value
        ],  # mutable-ok: dict-shaped JSON body
        "note": (
            "Every accepted policy is created in shadow. Shadow policies evaluate live traffic and "
            "write receipts with shadow=true; they never enforce. Activation is a separate, "
            "audited step that requires dlp:enforce, and dlp:approve as well for BLOCK policies."
        ),
    }


async def _import_one(
    tables: DLPTables,
    document: Mapping[str, object],
    payload: PolicyImportRequest,
    organization_id: str | None,
    caller: UserAPIKeyAuth,
) -> Mapping[str, object]:
    parsed: Final = parse_policy(document)
    if isinstance(parsed, PolicyValidationFailure):
        return {  # mutable-ok: dict-shaped JSON body
            "name": document.get("name"),
            "status": "rejected",
            "errors": list(parsed.errors),  # mutable-ok: dict-shaped JSON body
        }  # mutable-ok: dict-shaped JSON body
    compiled: Final = compile_policy(parsed)
    compile_error: Final = compiled.summary if isinstance(compiled, CompileFailure) else None
    if compile_error is not None:
        return {  # mutable-ok: dict-shaped JSON body
            "name": parsed.name,
            "status": "rejected",
            "errors": [compile_error],  # mutable-ok: dict-shaped JSON body
            "review_required": True,
        }
    policy_id: Final = str(uuid.uuid4())
    document_hash: Final = canonical_hash(document)
    await tables.policies.create(
        data={  # mutable-ok: dict-shaped JSON body
            "policy_id": policy_id,
            "name": parsed.name,
            "description": parsed.description,
            "source": parsed.source.vendor or "local",
            "connection_id": payload.connection_id,
            "external_policy_id": parsed.source.external_policy_id,
            "external_policy_version": parsed.source.external_policy_version,
            "mode": parsed.mode.value,
            "status": PolicyStatus.SHADOW.value,
            "priority": parsed.priority,
            "direction": parsed.direction.value,
            "canonical_policy_json": json.dumps(document),
            "external_policy_hash": parsed.source.external_policy_hash,
            "canonical_policy_hash": document_hash,
            "compile_status": compiled.compile_status.value,
            "organization_id": organization_id,
            "updated_at": _now(),
        }
    )
    await tables.versions.create(
        data={  # mutable-ok: dict-shaped JSON body
            "policy_version_id": str(uuid.uuid4()),
            "policy_id": policy_id,
            "version": 1,
            "source_version": parsed.source.external_policy_version,
            "canonical_json": json.dumps(document),
            "hash": document_hash,
            "change_note": payload.change_note or "initial import",
            "created_by": caller.user_id or "unknown",
        }
    )
    return {  # mutable-ok: dict-shaped JSON body
        "policy_id": policy_id,
        "name": parsed.name,
        "status": PolicyStatus.SHADOW.value,
        "compile_status": compiled.compile_status.value,
        "unmapped": list(parsed.source.unmapped),  # mutable-ok: dict-shaped JSON body
    }


@router.post("/policies/{policy_id}/shadow")
async def move_policy_to_shadow(
    policy_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Mapping[str, object]:
    require_permission(user_api_key_dict, DLPPermission.APPROVE)
    await _policy_or_404(policy_id, user_api_key_dict)
    return await _set_status(policy_id, PolicyStatus.SHADOW, user_api_key_dict, activated=False)


@router.post("/policies/{policy_id}/activate")
async def activate_policy(
    policy_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Mapping[str, object]:
    require_permission(user_api_key_dict, DLPPermission.ENFORCE)
    record: Final = await _policy_or_404(policy_id, user_api_key_dict)
    current: Final = PolicyStatus(getattr(record, "status", PolicyStatus.IMPORTED.value))
    if current is not PolicyStatus.SHADOW:
        raise HTTPException(
            status_code=409,
            detail={  # mutable-ok: dict-shaped JSON body
                "error": "a policy must run in shadow before it can enforce",
                "current_status": current.value,
            },
        )
    action: Final = _action_of(record)
    if action in HUMAN_APPROVAL_ACTIONS:
        require_permission(user_api_key_dict, DLPPermission.APPROVE)
    if action is PolicyAction.REQUIRE_APPROVAL:
        raise HTTPException(
            status_code=409,
            detail={  # mutable-ok: dict-shaped JSON body
                "error": "REQUIRE_APPROVAL has no approval queue yet and cannot be activated"
            },  # mutable-ok: dict-shaped JSON body
        )
    verbose_proxy_logger.info(
        "WIT OS DLP: policy %s activated by %s (action=%s)",
        policy_id,
        user_api_key_dict.user_id,
        action.value,
    )
    return await _set_status(policy_id, PolicyStatus.ACTIVE, user_api_key_dict, activated=True)


@router.post("/policies/{policy_id}/disable")
async def disable_policy(
    policy_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Mapping[str, object]:
    require_permission(user_api_key_dict, DLPPermission.ENFORCE)
    await _policy_or_404(policy_id, user_api_key_dict)
    return await _set_status(policy_id, PolicyStatus.DISABLED, user_api_key_dict, activated=False)


def _action_of(record: object) -> PolicyAction:
    document: Final = _json_field(record, "canonical_policy_json")
    if not isinstance(document, Mapping):
        return PolicyAction.AUDIT
    actions: Final = document.get("actions")
    if not isinstance(actions, Mapping):
        return PolicyAction.AUDIT
    on_match: Final = actions.get("on_match")
    return PolicyAction(on_match) if isinstance(on_match, str) else PolicyAction.AUDIT


async def _set_status(
    policy_id: str,
    status: PolicyStatus,
    caller: UserAPIKeyAuth,
    activated: bool,
) -> Mapping[str, object]:
    tables: Final = tables_for(_require_prisma())
    record: Final = await tables.policies.update(
        where={"policy_id": policy_id},  # mutable-ok: dict-shaped JSON body
        data={  # mutable-ok: dict-shaped JSON body
            "status": status.value,
            "approved_by": caller.user_id if activated else None,
            "activated_at": _now() if activated else None,
            "updated_at": _now(),
        },
    )
    await _invalidate_policy_caches()
    return _policy_view(record)


@router.post("/test")
async def test_bench(
    payload: TestBenchRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Mapping[str, object]:
    require_permission(user_api_key_dict, DLPPermission.TEST)
    prisma_client: Final = _require_prisma()
    organization_id: Final = payload.organization_id or user_api_key_dict.org_id
    scope: Final = RequestScope(
        organization_id=organization_id,
        team_id=payload.team_id,
        user_id=payload.user_id,
        key_alias=payload.key_alias,
        application=payload.application,
        model=payload.model,
        model_group=payload.model_group,
    )

    async def loader(org: str | None) -> tuple[ScopedPolicy, ...]:
        return await load_policy_set(prisma_client, org)

    engine: Final = PolicyFabricEngine(
        cache=PolicySetCache(loader=loader, ttl_seconds=0.0),
        store=CompiledPolicyStore(),
        config=EngineConfig(guardrail_name="wit_dlp_test_bench"),
    )
    result: Final = await engine.evaluate(
        EvaluationRequest(
            request_id=f"testbench-{uuid.uuid4()}",
            content=payload.content,
            direction=payload.direction,
            scope=scope,
            tool_name=payload.tool_name,
            tool_arguments=payload.tool_arguments or {},  # mutable-ok: dict-shaped JSON body
        )
    )
    return {  # mutable-ok: dict-shaped JSON body
        "decision": result.decision.action.value,
        "shadow_decision": result.decision.shadow_action.value,
        "enforcement_active": result.enforcement_active,
        "enforcement": result.enforcement_kind.value,
        "evaluation_latency_ms": result.evaluation_latency_ms,
        "compile_failures": list(result.compile_failures),  # mutable-ok: dict-shaped JSON body
        "verdicts": [  # mutable-ok: dict-shaped JSON body
            {  # mutable-ok: dict-shaped JSON body
                "policy_id": verdict.policy_id,
                "policy_name": verdict.policy_name,
                "action": verdict.action.value,
                "shadow": verdict.shadow,
                "source_vendor": verdict.provider,
                "matched_classifiers": [  # mutable-ok: dict-shaped JSON body
                    {"class": matched.classifier, "count": matched.count}  # mutable-ok: dict-shaped JSON body
                    for matched in summarise_findings(verdict.findings)
                ],
                "matched_rule_ids": list(verdict.matched_rule_ids),  # mutable-ok: dict-shaped JSON body
                "fail_mode_triggered": verdict.fail_mode_triggered,
                "evaluation_latency_ms": verdict.evaluation_latency_ms,
            }
            for verdict in (*result.decision.enforced_verdicts, *result.decision.shadow_verdicts)
        ],
        "note": "The test bench never writes decision receipts and never enforces.",
    }


@router.get("/decisions")
async def list_decisions(
    policy_id: str | None = None,
    shadow: bool | None = None,
    limit: int = Query(default=100, le=_MAX_PAGE_SIZE),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Mapping[str, object]:
    require_permission(user_api_key_dict, DLPPermission.VIEW_DECISIONS)
    tables: Final = tables_for(_require_prisma())
    filters: Final = {  # mutable-ok: dict-shaped JSON body
        key: value for key, value in (("policy_id", policy_id), ("shadow", shadow)) if value is not None
    }  # mutable-ok: dict-shaped JSON body
    rows: Final = await tables.decisions.find_many(
        where=_merged_where(filters, user_api_key_dict),
        order={"created_at": "desc"},  # mutable-ok: dict-shaped JSON body
        take=limit,
    )
    return {"decisions": [_decision_view(row) for row in rows]}  # mutable-ok: dict-shaped JSON body


@router.get("/decisions/summary")
async def decisions_summary(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    limit: int = Query(default=_MAX_PAGE_SIZE, le=_MAX_PAGE_SIZE),
) -> Mapping[str, object]:
    require_permission(user_api_key_dict, DLPPermission.VIEW_DECISIONS)
    tables: Final = tables_for(_require_prisma())
    rows: Final = await tables.decisions.find_many(
        where=_merged_where({}, user_api_key_dict),  # mutable-ok: dict-shaped JSON body
        order={"created_at": "desc"},  # mutable-ok: dict-shaped JSON body
        take=limit,  # mutable-ok: dict-shaped JSON body
    )
    decisions: Final = tuple(str(getattr(row, "decision", "")) for row in rows)
    return {  # mutable-ok: dict-shaped JSON body
        "total": len(rows),
        "by_action": {  # mutable-ok: dict-shaped JSON body
            action: decisions.count(action)
            for action in sorted(set(decisions))  # mutable-ok: dict-shaped JSON body
        },  # mutable-ok: dict-shaped JSON body
        "shadow": sum(1 for row in rows if getattr(row, "shadow", False) is True),
        "prevented": sum(1 for row in rows if getattr(row, "prevented", False) is True),
        "detected_only": sum(1 for row in rows if getattr(row, "prevented", False) is not True),
    }


@router.get("/decisions/{decision_id}")
async def get_decision(
    decision_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Mapping[str, object]:
    require_permission(user_api_key_dict, DLPPermission.VIEW_DECISIONS)
    tables: Final = tables_for(_require_prisma())
    row: Final = await tables.decisions.find_first(
        where=_merged_where({"decision_id": decision_id}, user_api_key_dict)  # mutable-ok: dict-shaped JSON body
    )  # mutable-ok: dict-shaped JSON body
    if row is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "decision not found"},  # mutable-ok: dict-shaped JSON body
        )  # mutable-ok: dict-shaped JSON body
    return _decision_view(row)


def _decision_view(row: object) -> Mapping[str, object]:
    streaming_mode: Final = getattr(row, "streaming_mode", None)
    prevented: Final = getattr(row, "prevented", False) is True
    return {  # mutable-ok: dict-shaped JSON body
        "decision_id": getattr(row, "decision_id", None),
        "created_at": getattr(row, "created_at", None),
        "request_id": getattr(row, "request_id", None),
        "policy_id": getattr(row, "policy_id", None),
        "policy_version": getattr(row, "policy_version", None),
        "provider": getattr(row, "provider", None),
        "direction": getattr(row, "direction", None),
        "decision": getattr(row, "decision", None),
        "matched_classifiers": _json_field(row, "matched_classifiers"),
        "scope": _json_field(row, "scope_json"),
        "shadow": getattr(row, "shadow", None),
        "fail_mode_triggered": getattr(row, "fail_mode_triggered", None),
        "redaction_performed": getattr(row, "redaction_performed", None),
        "evaluation_latency_ms": getattr(row, "evaluation_latency_ms", None),
        "streaming_mode": streaming_mode,
        "enforcement": "prevention" if prevented else "detection",
        "enforcement_note": (None if streaming_mode is None else disclaimer_for(StreamingMode(streaming_mode))),
    }


async def _connection_or_404(connection_id: str, caller: UserAPIKeyAuth) -> object:
    tables: Final = tables_for(_require_prisma())
    record: Final = await tables.connections.find_first(
        where=_merged_where({"connection_id": connection_id}, caller)  # mutable-ok: dict-shaped JSON body
    )  # mutable-ok: dict-shaped JSON body
    if record is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "connection not found"},  # mutable-ok: dict-shaped JSON body
        )  # mutable-ok: dict-shaped JSON body
    return record


async def _policy_or_404(policy_id: str, caller: UserAPIKeyAuth) -> object:
    tables: Final = tables_for(_require_prisma())
    record: Final = await tables.policies.find_first(
        where=_merged_where({"policy_id": policy_id}, caller)  # mutable-ok: dict-shaped JSON body
    )  # mutable-ok: dict-shaped JSON body
    if record is None:
        raise HTTPException(status_code=404, detail={"error": "policy not found"})  # mutable-ok: dict-shaped JSON body
    return record


async def _allowed_connection_ids(tables: DLPTables, caller: UserAPIKeyAuth) -> frozenset[str] | None:
    scoped: Final = tenant_filter(caller)
    if scoped is None:
        return None
    records: Final = await tables.connections.find_many(where=scoped)
    return frozenset(str(getattr(record, "connection_id", "")) for record in records)


async def _invalidate_policy_caches() -> None:
    """Hot reload: bump this pod, then tell every other pod. No restart."""
    from litellm.proxy.guardrails.guardrail_hooks.wit_dlp import invalidate_local_policy_caches

    invalidate_local_policy_caches()
    await publish_policy_invalidation(coordination_redis_cache())
