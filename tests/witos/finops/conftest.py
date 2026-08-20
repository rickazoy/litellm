"""Postgres-backed fixtures for the FinOps suite.

The whole point of these tests is the SQL, so they run against a real Postgres
and never against a stand-in. The aggregator's guarantees (a recompute replaces a
bucket, an ON CONFLICT arbitrates a lease, a JSONB path yields cache tokens) are
properties of Postgres; a fake that reimplemented them would only be testing
itself.

Nothing here reaches the network. Set ``WITOS_TEST_DATABASE_URL`` to a local
server, or leave it unset and the session probes the conventional local address
a CI Postgres service container listens on. With neither reachable the suite
skips rather than pretending to have covered anything.
"""

import os
import re
from collections.abc import AsyncIterator, Iterator, Sequence
from pathlib import Path

import psycopg
import pytest
from psycopg.rows import dict_row

REPO_ROOT = Path(__file__).resolve().parents[3]
MIGRATIONS_DIR = REPO_ROOT / "litellm-proxy-extras" / "litellm_proxy_extras" / "migrations"
FALLBACK_DATABASE_URL = "postgresql://postgres:postgres@127.0.0.1:5432/postgres"

# Everything the FinOps SQL reads or writes. Truncated between tests so each one
# states its own world.
MANAGED_TABLES = (
    "WITOS_FinOpsUsageHourly",
    "WITOS_FinOpsUsageDaily",
    "WITOS_FinOpsQuota",
    "WITOS_FinOpsConfig",
    "WITOS_BackgroundJobLease",
    "LiteLLM_SpendLogs",
    "LiteLLM_VerificationToken",
    "LiteLLM_TeamTable",
    "LiteLLM_UserTable",
    "LiteLLM_EndUserTable",
    "LiteLLM_TagTable",
    "LiteLLM_OrganizationTable",
    "LiteLLM_BudgetTable",
)

_TRUNCATE_SQL = "TRUNCATE " + ", ".join(f'"{table}"' for table in MANAGED_TABLES) + " CASCADE"

_PLACEHOLDER = re.compile(r"\$(\d+)")


def to_psycopg(sql: str, args: Sequence[object]) -> tuple[str, tuple[object, ...]]:
    """Rewrite Prisma's ``$n`` placeholders as psycopg's ``%s``, preserving order.

    Arguments are re-ordered to follow the textual order of the placeholders, so
    a statement that repeats or reorders ``$n`` still binds what Postgres would.
    """
    order: list[int] = []

    def collect(match: "re.Match[str]") -> str:
        order.append(int(match.group(1)) - 1)
        return "%s"

    return _PLACEHOLDER.sub(collect, sql), tuple(args[index] for index in order)


class PsycopgExecutor:
    """``SqlExecutor`` over a real connection, for tests only."""

    def __init__(self, connection: psycopg.AsyncConnection) -> None:
        self._connection = connection

    async def query(self, sql: str, *args: object) -> Sequence[dict]:
        statement, bound = to_psycopg(sql, args)
        async with self._connection.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(statement, bound or None)
            return await cursor.fetchall()

    async def execute(self, sql: str, *args: object) -> int:
        statement, bound = to_psycopg(sql, args)
        async with self._connection.cursor() as cursor:
            await cursor.execute(statement, bound or None)
            return max(cursor.rowcount, 0)


def _migration_statements() -> Iterator[str]:
    for directory in sorted(path for path in MIGRATIONS_DIR.iterdir() if path.is_dir()):
        migration = directory / "migration.sql"
        if migration.exists():
            yield migration.read_text()


@pytest.fixture(scope="session")
def database_url() -> str:
    candidate = os.environ.get("WITOS_TEST_DATABASE_URL") or FALLBACK_DATABASE_URL
    try:
        psycopg.connect(candidate, connect_timeout=3).close()
    except psycopg.Error as error:
        pytest.skip(f"No Postgres for the FinOps suite ({error}); set WITOS_TEST_DATABASE_URL")
    return candidate


@pytest.fixture(scope="session")
def migrated_database(database_url: str) -> str:
    """Apply the committed migration chain once, this branch's new migration included."""
    with psycopg.connect(database_url, autocommit=True) as connection:
        existing = connection.execute("""SELECT to_regclass('public."WITOS_FinOpsUsageHourly"')""").fetchone()
        if existing is None or existing[0] is None:
            for statement in _migration_statements():
                connection.execute(statement)
    return database_url


@pytest.fixture
async def connection(migrated_database: str) -> AsyncIterator[psycopg.AsyncConnection]:
    async with await psycopg.AsyncConnection.connect(migrated_database, autocommit=True) as conn:
        await conn.execute(_TRUNCATE_SQL)
        yield conn


@pytest.fixture
async def executor(connection: psycopg.AsyncConnection) -> PsycopgExecutor:
    return PsycopgExecutor(connection)
