"""FinOps permission scopes and entity clamping (blueprint §0.5).

Two independent checks, and both have to pass.

*What may this key do* is a scope string carried in the key's metadata or
permissions (``witos_scopes``). A CFO key holds ``finops:view``,
``finops:run_scenarios`` and ``finops:manage_alerts``, and cannot touch model
management or DLP. A SecOps key holds neither.

*Whose numbers may it see* is the clamp copied from the fork's own spend
endpoints (``_resolve_spend_report_scope``): a non-admin caller asking for
another team's forecast gets a 403, and a non-admin caller asking for nothing in
particular gets its own entities rather than the fleet. Forecast rows are
attributed to a scope exactly the way spend rows are, so the same rule applies
without reinterpretation.

Proxy admins pass both checks. That is the fork's existing model, and inventing a
second admin concept for FinOps would create a permission a customer's SSO group
mapping does not know about.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from fastapi import HTTPException, status
from pydantic import TypeAdapter, ValidationError

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.witos.finops.history import ScopeRef

FinOpsPermission: TypeAlias = Literal[
    "finops:view", "finops:manage", "finops:manage_budget", "finops:manage_alerts", "finops:run_scenarios"
]

SCOPES_METADATA_KEY: Final = "witos_scopes"

_SCOPE_NAMES: Final = TypeAdapter(tuple[str, ...])
_CLAIMS: Final = TypeAdapter(Mapping[str, object])

# Scope types a caller can be clamped to, mapped onto the field of its own key
# that has to match. `global`, `model`, `model_group` and `tag` are fleet-wide
# views with no per-caller identity, so they are admin-only.
_IDENTITY_FIELDS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "organization": "org_id",
        "team": "team_id",
        "user": "user_id",
        "key": "api_key",
        "end_user": "end_user_id",
    }
)

_ADMIN_ROLES: Final = (LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)


def forbidden(message: str) -> HTTPException:
    """The one place a FinOps refusal is shaped, so every route refuses identically."""
    return HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=_detail(message))


def bad_request(message: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=_detail(message))


def _detail(message: str) -> Mapping[str, str]:
    body: Final = {"error": message}  # mutable-ok: FastAPI serialises the detail body it is given
    return body


def is_admin(user: UserAPIKeyAuth) -> bool:
    return user.user_role in _ADMIN_ROLES


def granted_permissions(user: UserAPIKeyAuth) -> frozenset[str]:
    """Scopes on the key, from ``permissions`` first and ``metadata`` second.

    Both are untyped JSON columns on a model that lives in the type-checker's
    excluded ``_types.py``, so every value is parsed here rather than trusted.
    """
    return frozenset(
        _string_list(_scopes_claim(user.permissions))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # untyped JSON column on a model excluded from type checking
        + _string_list(_scopes_claim(user.metadata))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # untyped JSON column on a model excluded from type checking
    )


def _scopes_claim(container: object) -> object:
    """Pull the scopes claim out of an untyped JSON column without trusting its shape."""
    try:
        claims: Final = _CLAIMS.validate_python(container)
    except ValidationError:
        return None
    return claims.get(SCOPES_METADATA_KEY)


def _string_list(raw: object) -> tuple[str, ...]:
    """A comma-separated string or a JSON array of names, and nothing else."""
    if isinstance(raw, str):
        return tuple(part.strip() for part in raw.split(",") if part.strip())
    try:
        return _SCOPE_NAMES.validate_python(raw)
    except ValidationError:
        return ()


def require_permission(user: UserAPIKeyAuth, permission: FinOpsPermission) -> None:
    """403 unless the key carries the scope, or is a proxy admin."""
    if is_admin(user) or permission in granted_permissions(user):
        return
    raise forbidden(f"This key is missing the '{permission}' scope required for /witos/finops")


def _identity_of(user: UserAPIKeyAuth, scope_type: str) -> str | None:
    match _IDENTITY_FIELDS.get(scope_type):
        case "org_id":
            return user.org_id
        case "team_id":
            return user.team_id
        case "user_id":
            return user.user_id
        case "api_key":
            return user.api_key
        case "end_user_id":
            return user.end_user_id
        case _:
            return None


def resolve_scope(user: UserAPIKeyAuth, *, scope_type: str, scope_id: str | None) -> ScopeRef:
    """The scope this caller is allowed to read, or a 403 explaining why not.

    A caller that asks for its own team gets it. A caller that asks for another
    team, or for a fleet-wide scope type, is refused rather than silently
    downgraded to its own data, so a dashboard built against the wrong scope
    fails loudly instead of showing plausible numbers for somebody else.
    """
    if is_admin(user):
        if scope_id is None:
            raise bad_request(f"scope_id is required for scope_type={scope_type}")
        return ScopeRef(scope_type=scope_type, scope_id=scope_id)

    own: Final = _identity_of(user, scope_type)
    if own is None:
        raise forbidden(f"Not authorized to view a {scope_type} scope")
    if scope_id is not None and scope_id != own:
        raise forbidden(f"Not authorized to view a {scope_type} other than your own")
    return ScopeRef(scope_type=scope_type, scope_id=own)


def visible_scope_types(user: UserAPIKeyAuth) -> tuple[str, ...]:
    """Scope types a fleet-level listing may include for this caller."""
    if is_admin(user):
        return ("global", "organization", "team", "user", "key", "end_user")
    return tuple(scope_type for scope_type in _IDENTITY_FIELDS if _identity_of(user, scope_type) is not None)
