"""Database access for the policy fabric.

Prisma's generated client is untyped, so every table is reached through a
narrow `Protocol` view. That keeps the untyped surface at exactly one boundary
instead of letting it leak into the runtime.

Every read takes an `organization_id` and every query filters on it. Tenant
separation is a property of this module, not a discipline expected of callers.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Final, Protocol

from litellm._logging import verbose_proxy_logger
from litellm.proxy.witos.policy_fabric.canonical import PolicyValidationFailure, parse_policy
from litellm.proxy.witos.policy_fabric.scope import ScopedPolicy
from litellm.proxy.witos.policy_fabric.types import PolicyStatus

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


class PolicyRecord(Protocol):
    policy_id: str
    name: str
    source: str
    connection_id: str | None
    external_policy_id: str | None
    external_policy_version: str | None
    mode: str
    status: str
    priority: int
    direction: str
    canonical_policy_json: object
    canonical_policy_hash: str
    external_policy_hash: str | None
    compile_status: str | None
    compile_error: str | None
    approved_by: str | None
    activated_at: datetime | None
    organization_id: str | None
    created_at: datetime | None
    updated_at: datetime | None


class PolicyTableActions(Protocol):
    async def find_many(
        self,
        where: Mapping[str, object] | None = ...,
        order: Mapping[str, str] | None = ...,
        skip: int = ...,
        take: int = ...,
    ) -> Sequence[PolicyRecord]: ...

    async def find_first(self, where: Mapping[str, object]) -> PolicyRecord | None: ...

    async def create(self, data: Mapping[str, object]) -> PolicyRecord: ...

    async def update(self, where: Mapping[str, object], data: Mapping[str, object]) -> PolicyRecord | None: ...

    async def delete(self, where: Mapping[str, object]) -> PolicyRecord | None: ...

    async def count(self, where: Mapping[str, object] | None = ...) -> int: ...


class GenericTableActions(Protocol):
    async def find_many(
        self,
        where: Mapping[str, object] | None = ...,
        order: Mapping[str, str] | None = ...,
        skip: int = ...,
        take: int = ...,
    ) -> Sequence[object]: ...

    async def find_first(self, where: Mapping[str, object]) -> object | None: ...

    async def create(self, data: Mapping[str, object]) -> object: ...

    async def create_many(self, data: Sequence[Mapping[str, object]]) -> int: ...

    async def update(self, where: Mapping[str, object], data: Mapping[str, object]) -> object | None: ...

    async def delete(self, where: Mapping[str, object]) -> object | None: ...

    async def count(self, where: Mapping[str, object] | None = ...) -> int: ...


@dataclass(frozen=True, slots=True)
class DLPTables:
    connections: GenericTableActions
    policies: PolicyTableActions
    versions: GenericTableActions
    sync_runs: GenericTableActions
    decisions: GenericTableActions
    retro_runs: GenericTableActions


def tables_for(prisma_client: PrismaClient) -> DLPTables:
    database: Final = prisma_client.db
    return DLPTables(
        connections=database.witos_dlpconnection,
        policies=database.witos_dlppolicy,
        versions=database.witos_dlppolicyversion,
        sync_runs=database.witos_dlpsyncrun,
        decisions=database.witos_dlpdecision,
        retro_runs=database.witos_dlpretrorun,
    )


def _document_of(record: PolicyRecord) -> Mapping[str, object] | None:
    raw: Final = record.canonical_policy_json
    decoded: Final = json.loads(raw) if isinstance(raw, str) else raw
    return decoded if isinstance(decoded, Mapping) else None


def to_scoped_policy(record: PolicyRecord, version: int) -> ScopedPolicy | None:
    document: Final = _document_of(record)
    if document is None:
        verbose_proxy_logger.warning("WIT OS DLP: policy %s has an unreadable canonical document", record.policy_id)
        return None
    parsed: Final = parse_policy(document)
    if isinstance(parsed, PolicyValidationFailure):
        verbose_proxy_logger.warning(
            "WIT OS DLP: policy %s failed revalidation on load: %s", record.policy_id, parsed.errors
        )
        return None
    return ScopedPolicy(
        policy=parsed,
        policy_id=record.policy_id,
        version=version,
        status=PolicyStatus(record.status),
        organization_id=record.organization_id,
        connection_id=record.connection_id,
    )


EVALUATED_STATUS_VALUES: Final[tuple[str, ...]] = (
    PolicyStatus.ACTIVE.value,
    PolicyStatus.SHADOW.value,
    PolicyStatus.STALE.value,
)


async def load_policy_set(
    prisma_client: PrismaClient,
    organization_id: str | None,
) -> tuple[ScopedPolicy, ...]:
    """Load every policy that may evaluate for this tenant.

    Proxy-wide policies (no organization) and the caller's own tenant policies,
    and nothing else. A policy belonging to another organization is not filtered
    out later, it is never loaded.
    """
    tables: Final = tables_for(prisma_client)
    where: Final[Mapping[str, object]] = {  # mutable-ok: prisma input is dict-shaped
        "status": {"in": list(EVALUATED_STATUS_VALUES)},  # mutable-ok: prisma input is dict-shaped
        "OR": [  # mutable-ok: prisma input is dict-shaped
            {"organization_id": None},  # mutable-ok: prisma input is dict-shaped
            {"organization_id": organization_id},  # mutable-ok: prisma input is dict-shaped
        ],  # mutable-ok: prisma input is dict-shaped
    }
    records: Final = await tables.policies.find_many(
        where=where,
        order={"priority": "desc"},  # mutable-ok: prisma input is dict-shaped
    )  # mutable-ok: prisma input is dict-shaped
    versions: Final = await _latest_versions(tables, tuple(record.policy_id for record in records))
    return tuple(
        scoped
        for scoped in (to_scoped_policy(record, versions.get(record.policy_id, 1)) for record in records)
        if scoped is not None
    )


async def _latest_versions(tables: DLPTables, policy_ids: tuple[str, ...]) -> Mapping[str, int]:
    if not policy_ids:
        return {}  # mutable-ok: prisma input is dict-shaped
    rows: Final = await tables.versions.find_many(
        where={"policy_id": {"in": list(policy_ids)}},  # mutable-ok: prisma input is dict-shaped
        order={"version": "desc"},  # mutable-ok: prisma input is dict-shaped
    )
    return {  # mutable-ok: prisma input is dict-shaped
        policy_id: max(
            (version for row in rows for version in (_version_of(row, policy_id),) if version is not None),
            default=1,
        )
        for policy_id in policy_ids
    }


def _version_of(row: object, policy_id: str) -> int | None:
    row_policy_id: Final = getattr(row, "policy_id", None)
    version: Final = getattr(row, "version", None)
    if row_policy_id != policy_id or not isinstance(version, int):
        return None
    return version
