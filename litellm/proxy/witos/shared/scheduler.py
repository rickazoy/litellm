"""Multi-pod job lease for WIT OS background jobs (blueprint §1.5).

Every WIT OS job runs on every pod, so exactly one pod must win each cycle. The
lease is a row in ``WITOS_BackgroundJobLease`` claimed by a single
``INSERT ... ON CONFLICT DO UPDATE ... WHERE lease_until <= now()`` statement.
Postgres takes a row lock for the duration of that statement, so the WHERE is
evaluated against the committed incumbent and only one concurrent claimant can
observe it as true: that is the compare-and-set.

A Postgres advisory lock was the other candidate the blueprint allows, and it is
the wrong primitive here. The session-scoped form (``pg_advisory_lock``) is held
by a connection, and Prisma hands out pooled connections per statement with no
session affinity, so the lock could be taken on one connection and never
released from another. The transaction-scoped form (``pg_advisory_xact_lock``)
releases at commit and therefore cannot outlive a single statement, let alone a
multi-minute aggregation cycle. The lease row also survives a pod losing power,
expiring on wall-clock time, and can be inspected with a SELECT.

Expiry is evaluated by the database clock, never by the caller's, so
pods with skewed clocks cannot both believe they hold the lease.
"""

import asyncio
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final, TypeAlias, TypedDict

from typing_extensions import ReadOnly

from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.proxy.witos.shared.sql import SqlExecutor

DEFAULT_LEASE_TTL: Final = timedelta(minutes=5)
DEFAULT_HEARTBEAT_INTERVAL: Final = timedelta(seconds=60)


class _LeaseRow(TypedDict):
    owner_id: ReadOnly[str]
    lease_until: ReadOnly[datetime]


@dataclass(frozen=True, slots=True)
class LeaseAcquired:
    """This pod owns the job for the rest of ``lease_until``."""

    job_name: str
    owner_id: str
    lease_until: datetime


@dataclass(frozen=True, slots=True)
class LeaseHeld:
    """Another pod owns the job; this cycle is a no-op."""

    job_name: str
    held_by: str
    lease_until: datetime


LeaseResult: TypeAlias = LeaseAcquired | LeaseHeld

_CLAIM_SQL: Final = """
INSERT INTO "WITOS_BackgroundJobLease" (job_name, owner_id, lease_until, heartbeat_at)
VALUES ($1, $2, (now() AT TIME ZONE 'UTC') + make_interval(secs => $3::double precision),
        (now() AT TIME ZONE 'UTC'))
ON CONFLICT (job_name) DO UPDATE
    SET owner_id = EXCLUDED.owner_id,
        lease_until = EXCLUDED.lease_until,
        heartbeat_at = (now() AT TIME ZONE 'UTC')
    WHERE "WITOS_BackgroundJobLease".lease_until <= (now() AT TIME ZONE 'UTC')
RETURNING owner_id, lease_until
"""

_INCUMBENT_SQL: Final = """
SELECT owner_id, lease_until FROM "WITOS_BackgroundJobLease" WHERE job_name = $1
"""

_HEARTBEAT_SQL: Final = """
UPDATE "WITOS_BackgroundJobLease"
SET lease_until = (now() AT TIME ZONE 'UTC') + make_interval(secs => $3::double precision),
    heartbeat_at = (now() AT TIME ZONE 'UTC')
WHERE job_name = $1 AND owner_id = $2
"""

# Expires the lease rather than deleting the row, so the next pod can claim it
# immediately while `owner_id` and `heartbeat_at` stay readable for operators.
# '-infinity' rather than the current time: the column is TIMESTAMP(3), so a
# written now() rounds to the nearest millisecond and can land microseconds
# AFTER the now() the very next claim compares it against, leaving a
# just-released lease briefly unclaimable.
_RELEASE_SQL: Final = """
UPDATE "WITOS_BackgroundJobLease"
SET lease_until = '-infinity'::timestamp
WHERE job_name = $1 AND owner_id = $2
"""


def _as_lease_rows(rows: Sequence[object]) -> tuple[_LeaseRow, ...]:
    from pydantic import TypeAdapter

    return TypeAdapter(tuple[_LeaseRow, ...]).validate_python(tuple(rows))


class JobLeaseManager:
    """Claims, renews and releases the single-owner lease for a named job."""

    def __init__(self, executor: SqlExecutor, owner_id: str | None = None) -> None:
        self._executor: Final = executor
        self.owner_id: Final = owner_id or str(uuid.uuid4())

    async def acquire(self, job_name: str, ttl: timedelta = DEFAULT_LEASE_TTL) -> LeaseResult:
        claimed: Final = _as_lease_rows(
            await self._executor.query(_CLAIM_SQL, job_name, self.owner_id, ttl.total_seconds())
        )
        if claimed:
            return LeaseAcquired(job_name=job_name, owner_id=self.owner_id, lease_until=claimed[0]["lease_until"])
        incumbent: Final = _as_lease_rows(await self._executor.query(_INCUMBENT_SQL, job_name))
        if not incumbent:
            # The row was claimed and then deleted between the two statements.
            # Reporting it as held keeps this cycle a no-op; the next one retries.
            return LeaseHeld(job_name=job_name, held_by="unknown", lease_until=datetime.min)
        return LeaseHeld(job_name=job_name, held_by=incumbent[0]["owner_id"], lease_until=incumbent[0]["lease_until"])

    async def heartbeat(self, job_name: str, ttl: timedelta = DEFAULT_LEASE_TTL) -> bool:
        """Extend this pod's lease. False means ownership was lost and the job must stop."""
        updated: Final = await self._executor.execute(_HEARTBEAT_SQL, job_name, self.owner_id, ttl.total_seconds())
        return updated > 0

    async def release(self, job_name: str) -> bool:
        return await self._executor.execute(_RELEASE_SQL, job_name, self.owner_id) > 0

    @asynccontextmanager
    async def hold(
        self,
        job_name: str,
        ttl: timedelta = DEFAULT_LEASE_TTL,
        heartbeat_interval: timedelta = DEFAULT_HEARTBEAT_INTERVAL,
    ) -> AsyncGenerator[LeaseResult]:
        """Own ``job_name`` for the body, renewing the lease in the background.

        The body always runs so callers can branch on the result; a ``LeaseHeld``
        result means another pod owns the cycle and the body must skip its work.
        """
        result: Final = await self.acquire(job_name, ttl)
        if isinstance(result, LeaseHeld):
            verbose_proxy_logger.debug("WIT OS job %s already held by %s", job_name, result.held_by)
            yield result
            return
        renewer: Final = asyncio.create_task(self._renew_until_cancelled(job_name, ttl, heartbeat_interval))
        try:
            yield result
        finally:
            renewer.cancel()
            await asyncio.gather(renewer, return_exceptions=True)
            await self.release(job_name)

    async def _renew_until_cancelled(self, job_name: str, ttl: timedelta, interval: timedelta) -> None:
        while True:
            await asyncio.sleep(interval.total_seconds())
            if not await self.heartbeat(job_name, ttl):
                verbose_proxy_logger.warning("WIT OS lost the lease for job %s mid-run", job_name)
                return
