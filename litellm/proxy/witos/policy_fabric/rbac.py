"""DLP permission scopes (§0.5).

A SecOps key gets `dlp:*` and no financial visibility; a CFO key gets the FinOps
scopes and no DLP access at all. Scopes are read from key metadata, and the
separation is the point: whoever can read decision receipts is not automatically
whoever can turn a blocking policy on.

`dlp:approve` and `dlp:enforce` are deliberately distinct from `dlp:import`.
Importing a policy is safe because imports land in shadow. Activating one is the
step that can start refusing a customer's traffic, so it needs its own grant.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Final

from fastapi import HTTPException

from litellm.proxy._types import (
    LitellmUserRoles,
    UserAPIKeyAuth,
    user_api_key_has_admin_view,
)


class DLPPermission(str, Enum):
    VIEW = "dlp:view"
    MANAGE_CONNECTIONS = "dlp:manage_connections"
    IMPORT = "dlp:import"
    APPROVE = "dlp:approve"
    ENFORCE = "dlp:enforce"
    VIEW_DECISIONS = "dlp:view_decisions"
    TEST = "dlp:test"


DLP_WILDCARD: Final = "dlp:*"
_READ_ONLY_PERMISSIONS: Final[frozenset[DLPPermission]] = frozenset({DLPPermission.VIEW, DLPPermission.VIEW_DECISIONS})


def granted_scopes(user_api_key_dict: UserAPIKeyAuth) -> frozenset[str]:
    return frozenset(
        scope
        for source in (user_api_key_dict.permissions, user_api_key_dict.metadata)
        for scope in _scopes_from(source)
    )


def _scopes_from(source: Mapping[str, object] | None) -> tuple[str, ...]:
    if not isinstance(source, Mapping):
        return ()
    raw: Final = source.get("scopes") or source.get("permissions")
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return tuple(entry for entry in raw if isinstance(entry, str))
    return ()


def has_permission(user_api_key_dict: UserAPIKeyAuth, permission: DLPPermission) -> bool:
    if user_api_key_dict.user_role is LitellmUserRoles.PROXY_ADMIN:
        return True
    if user_api_key_has_admin_view(user_api_key_dict) and permission in _READ_ONLY_PERMISSIONS:
        return True
    scopes: Final = granted_scopes(user_api_key_dict)
    return permission.value in scopes or DLP_WILDCARD in scopes


def require_permission(user_api_key_dict: UserAPIKeyAuth, permission: DLPPermission) -> None:
    if has_permission(user_api_key_dict, permission):
        return
    raise HTTPException(
        status_code=403,
        detail={  # mutable-ok: prisma where fragment is dict-shaped
            "error": f"missing required scope {permission.value}"
        },  # mutable-ok: prisma where fragment is dict-shaped
    )


def tenant_filter(user_api_key_dict: UserAPIKeyAuth) -> Mapping[str, object] | None:
    """Prisma `where` fragment restricting rows to the caller's organization.

    Returns None only for proxy admins. Every DLP query passes its `where`
    through this: strict tenant separation is not something to remember at each
    call site.
    """
    if user_api_key_has_admin_view(user_api_key_dict):
        return None
    organization_id: Final = user_api_key_dict.org_id
    if organization_id is None:
        return {"organization_id": "__no_tenant__"}  # mutable-ok: prisma where fragment is dict-shaped
    return {"organization_id": organization_id}  # mutable-ok: prisma where fragment is dict-shaped
