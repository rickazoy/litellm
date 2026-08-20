"""The narrow SQL surface every WIT OS background job runs against.

WIT OS aggregation is pure SQL by design (blueprint §1.5): the proxy must never
pull spend rows into Python to add them up. That leaves the jobs needing exactly
two operations, so they depend on this two-method protocol rather than on
``PrismaClient``. Injecting the executor is what lets the aggregation and lease
SQL be exercised against a real Postgres in tests without a running proxy.
"""

from collections.abc import Mapping, Sequence
from typing import Final, Protocol, TypeAlias, runtime_checkable

Row: TypeAlias = Mapping[str, object]


@runtime_checkable
class SqlExecutor(Protocol):
    """Postgres access for background jobs. Implementations must be safe to call concurrently."""

    async def query(self, sql: str, *args: object) -> Sequence[Row]:
        """Run a statement that returns rows."""
        ...

    async def execute(self, sql: str, *args: object) -> int:
        """Run a statement that returns a row count."""
        ...


class _RawQueryClient(Protocol):
    async def query_raw(self, query: str, *args: object) -> Sequence[Row]: ...

    async def execute_raw(self, query: str, *args: object) -> int: ...


class _ProxyPrismaClient(Protocol):
    @property
    def db(self) -> _RawQueryClient: ...


class PrismaSqlExecutor:
    """``SqlExecutor`` backed by a Prisma client."""

    def __init__(self, client: _RawQueryClient) -> None:
        self._client: Final = client

    @classmethod
    def for_proxy(cls, prisma_client: _ProxyPrismaClient) -> "PrismaSqlExecutor":
        """Wrap the proxy's ``PrismaClient``, which nests the query API under ``.db``."""
        return cls(prisma_client.db)

    async def query(self, sql: str, *args: object) -> Sequence[Row]:
        return await self._client.query_raw(sql, *args)

    async def execute(self, sql: str, *args: object) -> int:
        return await self._client.execute_raw(sql, *args)
