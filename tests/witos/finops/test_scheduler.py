"""Lease arbitration for multi-pod background jobs (blueprint §1.5)."""

import asyncio
from datetime import timedelta

import pytest

from litellm.proxy.witos.shared.scheduler import JobLeaseManager, LeaseAcquired, LeaseHeld

JOB = "witos_test_job"


async def test_only_one_of_two_pods_wins_the_lease(executor):
    first = JobLeaseManager(executor, owner_id="pod-a")
    second = JobLeaseManager(executor, owner_id="pod-b")

    won = await first.acquire(JOB, ttl=timedelta(minutes=5))
    lost = await second.acquire(JOB, ttl=timedelta(minutes=5))

    assert isinstance(won, LeaseAcquired)
    assert isinstance(lost, LeaseHeld)
    assert lost.held_by == "pod-a"


async def test_concurrent_claims_produce_exactly_one_winner(migrated_database):
    """Two pods racing on the same row: the CAS must not let both through."""
    import psycopg

    from tests.witos.finops.conftest import PsycopgExecutor

    async def claim(owner: str):
        async with await psycopg.AsyncConnection.connect(migrated_database, autocommit=True) as conn:
            return await JobLeaseManager(PsycopgExecutor(conn), owner_id=owner).acquire(JOB)

    async with await psycopg.AsyncConnection.connect(migrated_database, autocommit=True) as setup:
        await setup.execute('DELETE FROM "WITOS_BackgroundJobLease" WHERE job_name = %s', (JOB,))

    results = await asyncio.gather(*(claim(f"pod-{index}") for index in range(8)))
    winners = [result for result in results if isinstance(result, LeaseAcquired)]
    assert len(winners) == 1
    assert all(result.held_by == winners[0].owner_id for result in results if isinstance(result, LeaseHeld))


async def test_crashed_owner_lease_is_taken_over_after_expiry(executor):
    """A pod that dies mid-cycle never renews, so the lease expires and the next pod resumes."""
    crashed = JobLeaseManager(executor, owner_id="pod-crashed")
    survivor = JobLeaseManager(executor, owner_id="pod-survivor")

    assert isinstance(await crashed.acquire(JOB, ttl=timedelta(seconds=1)), LeaseAcquired)
    assert isinstance(await survivor.acquire(JOB, ttl=timedelta(minutes=5)), LeaseHeld)

    await asyncio.sleep(1.2)

    assert isinstance(await survivor.acquire(JOB, ttl=timedelta(minutes=5)), LeaseAcquired)


async def test_heartbeat_extends_only_the_current_owner(executor):
    owner = JobLeaseManager(executor, owner_id="pod-a")
    other = JobLeaseManager(executor, owner_id="pod-b")
    acquired = await owner.acquire(JOB, ttl=timedelta(seconds=2))
    assert isinstance(acquired, LeaseAcquired)

    assert await owner.heartbeat(JOB, ttl=timedelta(minutes=10)) is True
    assert await other.heartbeat(JOB, ttl=timedelta(minutes=10)) is False

    # The extension is real: the short original TTL has passed and the lease still holds.
    await asyncio.sleep(2.1)
    assert isinstance(await other.acquire(JOB), LeaseHeld)


async def test_release_frees_the_lease_immediately(executor):
    owner = JobLeaseManager(executor, owner_id="pod-a")
    other = JobLeaseManager(executor, owner_id="pod-b")
    await owner.acquire(JOB, ttl=timedelta(hours=1))

    assert await owner.release(JOB) is True
    assert isinstance(await other.acquire(JOB), LeaseAcquired)


async def test_release_by_a_non_owner_is_a_no_op(executor):
    owner = JobLeaseManager(executor, owner_id="pod-a")
    imposter = JobLeaseManager(executor, owner_id="pod-b")
    await owner.acquire(JOB, ttl=timedelta(hours=1))

    assert await imposter.release(JOB) is False
    assert isinstance(await imposter.acquire(JOB), LeaseHeld)


async def test_hold_releases_even_when_the_body_raises(executor):
    owner = JobLeaseManager(executor, owner_id="pod-a")
    other = JobLeaseManager(executor, owner_id="pod-b")

    with pytest.raises(RuntimeError):
        async with owner.hold(JOB, ttl=timedelta(hours=1)) as lease:
            assert isinstance(lease, LeaseAcquired)
            raise RuntimeError("cycle blew up")

    assert isinstance(await other.acquire(JOB), LeaseAcquired)


async def test_hold_yields_lease_held_when_another_pod_owns_the_job(executor):
    owner = JobLeaseManager(executor, owner_id="pod-a")
    other = JobLeaseManager(executor, owner_id="pod-b")
    await owner.acquire(JOB, ttl=timedelta(hours=1))

    async with other.hold(JOB) as lease:
        assert isinstance(lease, LeaseHeld)
        assert lease.held_by == "pod-a"

    # The loser's hold must not have released the winner's lease on the way out.
    assert isinstance(await JobLeaseManager(executor, owner_id="pod-c").acquire(JOB), LeaseHeld)
