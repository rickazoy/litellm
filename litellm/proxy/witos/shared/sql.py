"""The narrow SQL surface every WIT OS background job runs against.

WIT OS aggregation is pure SQL by design (blueprint §1.5): the proxy must never
pull spend rows into Python to add them up. That leaves the jobs needing exactly
two operations, so they depend on this two-method protocol rather than on
``PrismaClient``. Injecting the executor is what lets the aggregation and lease
SQL be exercised against a real Postgres in tests without a running proxy.
"""

from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timezone
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


def bind(value: object) -> object:
    """Make one bind survive Prisma's raw-query encoder.

    ``query_raw`` JSON-encodes every argument through a ``singledispatch``
    registry that covers ``datetime``, ``Decimal``, ``Json`` and ``Base64`` and
    nothing else, so a bare ``date`` raises ``TypeError: Type <class
    'datetime.date'> not serializable`` at request time. Every WIT OS bucket
    column is a UTC day and every statement casts these binds to
    ``timestamptz``, so a ``date`` is promoted to UTC midnight rather than
    stringified. The value is made timezone-aware deliberately: Prisma treats a
    naive datetime as UTC but Postgres would not, and a bind that means a
    different instant depending on the server's zone is the same class of bug as
    the naive-``TIMESTAMP(3)`` lease columns.

    ``datetime`` subclasses ``date``, so the test is exclusive or every
    timestamp would be truncated to midnight.
    """
    if isinstance(value, date) and not isinstance(value, datetime):
        return datetime.combine(value, time.min, tzinfo=timezone.utc)
    return value


class PrismaSqlExecutor:
    """``SqlExecutor`` backed by a Prisma client."""

    def __init__(self, client: _RawQueryClient) -> None:
        self._client: Final = client

    @classmethod
    def for_proxy(cls, prisma_client: _ProxyPrismaClient) -> "PrismaSqlExecutor":
        """Wrap the proxy's ``PrismaClient``, which nests the query API under ``.db``."""
        return cls(prisma_client.db)

    async def query(self, sql: str, *args: object) -> Sequence[Row]:
        return await self._client.query_raw(sql, *(bind(arg) for arg in args))

    async def execute(self, sql: str, *args: object) -> int:
        return await self._client.execute_raw(sql, *(bind(arg) for arg in args))
