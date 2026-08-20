"""The flag gate in ``litellm/proxy/witos/registration.py``.

No network, no database, no running scheduler. Everything the wiring touches is
injected: a scheduler that records what it was asked to run, and a Prisma stand-in
whose only job is to hand back rows or blow up on demand.

The tests that matter most here are the negative ones. A build with both flags
unset is what the first deploy of this image runs, so "registers nothing and
schedules nothing" is not a tidiness check, it is the deploy plan's first gate.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Final

import pytest
from fastapi import FastAPI

from litellm._logging import verbose_proxy_logger
from litellm.constants import APSCHEDULER_MISFIRE_GRACE_TIME
from litellm.proxy.witos.registration import (
    AGGREGATION_INTERVAL_ENV,
    AGGREGATION_JOB_ID,
    ANOMALY_INTERVAL_ENV,
    ANOMALY_JOB_ID,
    DLP_ENABLED_ENV,
    DLP_ROUTE_PREFIX,
    FINOPS_ENABLED_ENV,
    FINOPS_ROUTE_PREFIX,
    FORECAST_HOUR_ENV,
    FORECAST_JOB_ID,
    LATENESS_WINDOW_ENV,
    QUALITY_HOUR_ENV,
    QUALITY_JOB_ID,
    QUOTA_MIRROR_HOUR_ENV,
    QUOTA_MIRROR_JOB_ID,
    include_witos_routers,
    schedule_witos_jobs,
)

ALL_JOB_IDS: Final = (
    AGGREGATION_JOB_ID,
    ANOMALY_JOB_ID,
    QUOTA_MIRROR_JOB_ID,
    FORECAST_JOB_ID,
    QUALITY_JOB_ID,
)

_ALL_ENV: Final = (
    FINOPS_ENABLED_ENV,
    DLP_ENABLED_ENV,
    LATENESS_WINDOW_ENV,
    AGGREGATION_INTERVAL_ENV,
    ANOMALY_INTERVAL_ENV,
    QUOTA_MIRROR_HOUR_ENV,
    FORECAST_HOUR_ENV,
    QUALITY_HOUR_ENV,
)


@dataclass(frozen=True)
class RecordedJob:
    func: Callable[[], Awaitable[None]]
    trigger: str
    job_id: str
    replace_existing: bool
    misfire_grace_time: int
    minutes: int | None
    hour: int | None
    minute: int | None
    timezone: str | None


class RecordingScheduler:
    """``JobScheduler`` that keeps the registration instead of running it."""

    def __init__(self) -> None:
        self.jobs: list[RecordedJob] = []

    def add_job(
        self,
        func: Callable[[], Awaitable[None]],
        trigger: str,
        *,
        id: str,
        replace_existing: bool,
        misfire_grace_time: int,
        minutes: int | None = None,
        hour: int | None = None,
        minute: int | None = None,
        timezone: str | None = None,
    ) -> object:
        self.jobs.append(
            RecordedJob(
                func=func,
                trigger=trigger,
                job_id=id,
                replace_existing=replace_existing,
                misfire_grace_time=misfire_grace_time,
                minutes=minutes,
                hour=hour,
                minute=minute,
                timezone=timezone,
            )
        )
        return None

    def by_id(self, job_id: str) -> RecordedJob:
        matched: Final = [job for job in self.jobs if job.job_id == job_id]
        assert len(matched) == 1, f"expected exactly one {job_id}, got {len(matched)}"
        return matched[0]


class FakeDb:
    """The two raw-SQL methods ``PrismaSqlExecutor`` reaches for, and nothing else."""

    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure
        self.statements: list[str] = []

    async def query_raw(self, query: str, *args: Any) -> Sequence[Any]:
        self.statements.append(query)
        if self.failure is not None:
            raise self.failure
        return []

    async def execute_raw(self, query: str, *args: Any) -> int:
        self.statements.append(query)
        if self.failure is not None:
            raise self.failure
        return 0


class FakePrisma:
    def __init__(self, failure: Exception | None = None) -> None:
        self.db = FakeDb(failure)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neither the developer's shell nor CI may decide what these tests observe."""
    for name in _ALL_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def captured_logs(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    """``verbose_proxy_logger`` output, which litellm otherwise keeps off the root handler."""
    monkeypatch.setattr(verbose_proxy_logger, "propagate", True)
    caplog.set_level(logging.INFO, logger=verbose_proxy_logger.name)
    return caplog


def witos_paths(app: FastAPI) -> tuple[str, ...]:
    """Every resolved path the app would serve under ``/witos``."""
    return tuple(sorted(route.path for route in app.routes if getattr(route, "path", "").startswith("/witos")))


def mounted() -> tuple[str, ...]:
    app: Final = FastAPI()
    include_witos_routers(app)
    return witos_paths(app)


def scheduled(prisma: FakePrisma | None) -> RecordingScheduler:
    scheduler: Final = RecordingScheduler()
    schedule_witos_jobs(scheduler, prisma)  # pyright: ignore[reportArgumentType]  # test double
    return scheduler


class TestBothFlagsOff:
    def test_mounts_no_routes(self) -> None:
        assert mounted() == ()

    def test_schedules_no_jobs(self) -> None:
        assert scheduled(FakePrisma()).jobs == []

    def test_touches_no_database(self) -> None:
        prisma: Final = FakePrisma()
        scheduled(prisma)
        assert prisma.db.statements == []

    def test_says_so_in_the_log(self, captured_logs: pytest.LogCaptureFixture) -> None:
        include_witos_routers(FastAPI())
        scheduled(FakePrisma())
        text: Final = captured_logs.text
        assert f"WIT OS FinOps API SKIPPED: {FINOPS_ENABLED_ENV} is not set to true" in text
        assert f"WIT OS DLP API SKIPPED: {DLP_ENABLED_ENV} is not set to true" in text
        assert f"WIT OS FinOps jobs SKIPPED: {FINOPS_ENABLED_ENV} is not set to true" in text


@pytest.mark.parametrize("value", ["", " ", "1", "0", "yes", "no", "on", "off", "enabled", "TRUE-ish"])
def test_only_an_explicit_true_enables_a_subsystem(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(FINOPS_ENABLED_ENV, value)
    monkeypatch.setenv(DLP_ENABLED_ENV, value)
    assert mounted() == ()
    assert scheduled(FakePrisma()).jobs == []


@pytest.mark.parametrize("value", ["true", "True", "TRUE", " true "])
def test_true_in_any_casing_enables(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(DLP_ENABLED_ENV, value)
    assert any(path.startswith(DLP_ROUTE_PREFIX) for path in mounted())


class TestFlagsAreIndependent:
    def test_finops_alone_mounts_only_finops(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(FINOPS_ENABLED_ENV, "true")
        paths: Final = mounted()
        assert paths, "the FinOps flag mounted nothing"
        assert all(path.startswith(FINOPS_ROUTE_PREFIX) for path in paths)

    def test_dlp_alone_mounts_only_dlp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(DLP_ENABLED_ENV, "true")
        paths: Final = mounted()
        assert paths, "the DLP flag mounted nothing"
        assert all(path.startswith(DLP_ROUTE_PREFIX) for path in paths)

    def test_dlp_alone_schedules_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(DLP_ENABLED_ENV, "true")
        assert scheduled(FakePrisma()).jobs == []

    def test_both_on_mounts_both(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(FINOPS_ENABLED_ENV, "true")
        monkeypatch.setenv(DLP_ENABLED_ENV, "true")
        paths: Final = mounted()
        assert any(path.startswith(FINOPS_ROUTE_PREFIX) for path in paths)
        assert any(path.startswith(DLP_ROUTE_PREFIX) for path in paths)


class TestDlpNestsRetro:
    """The retro router is mounted by the DLP router, so it must arrive with it and only once."""

    @pytest.fixture
    def paths(self, monkeypatch: pytest.MonkeyPatch) -> tuple[str, ...]:
        monkeypatch.setenv(DLP_ENABLED_ENV, "true")
        return mounted()

    @pytest.mark.parametrize(
        "path",
        [
            "/witos/dlp/retro/simulate",
            "/witos/dlp/retro/compare",
            "/witos/dlp/retro/runs",
            "/witos/dlp/retro/runs/{run_id}",
        ],
    )
    def test_retro_route_resolves_under_the_parent_prefix(self, paths: tuple[str, ...], path: str) -> None:
        assert path in paths

    def test_retro_is_not_also_mounted_at_the_top_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(DLP_ENABLED_ENV, "true")
        app: Final = FastAPI()
        include_witos_routers(app)
        assert not [route for route in app.routes if getattr(route, "path", "").startswith("/retro")]

    def test_no_retro_route_is_registered_twice(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A path may legitimately appear once per method, but never twice for the same one."""
        monkeypatch.setenv(DLP_ENABLED_ENV, "true")
        app: Final = FastAPI()
        include_witos_routers(app)
        registered: Final = [
            (route.path, method)
            for route in app.routes
            if getattr(route, "path", "").startswith("/witos/dlp/retro/")
            for method in sorted(getattr(route, "methods", ()) or ())
        ]
        assert registered
        assert len(registered) == len(set(registered))


class TestScheduling:
    @pytest.fixture
    def scheduler(self, monkeypatch: pytest.MonkeyPatch) -> RecordingScheduler:
        monkeypatch.setenv(FINOPS_ENABLED_ENV, "true")
        return scheduled(FakePrisma())

    def test_every_finops_job_is_registered_exactly_once(self, scheduler: RecordingScheduler) -> None:
        assert sorted(job.job_id for job in scheduler.jobs) == sorted(ALL_JOB_IDS)

    def test_job_ids_carry_the_lease_names_the_jobs_claim(self) -> None:
        """Drift here would leave the scheduler log and the lease table talking about different jobs."""
        from litellm.proxy.witos.finops.aggregator import AGGREGATION_JOB_NAME
        from litellm.proxy.witos.finops.jobs import ANOMALY_JOB_NAME, FORECAST_JOB_NAME, QUALITY_JOB_NAME
        from litellm.proxy.witos.finops.quotas import QUOTA_MIRROR_JOB_NAME

        assert AGGREGATION_JOB_ID == f"{AGGREGATION_JOB_NAME}_job"
        assert ANOMALY_JOB_ID == f"{ANOMALY_JOB_NAME}_job"
        assert QUOTA_MIRROR_JOB_ID == f"{QUOTA_MIRROR_JOB_NAME}_job"
        assert FORECAST_JOB_ID == f"{FORECAST_JOB_NAME}_job"
        assert QUALITY_JOB_ID == f"{QUALITY_JOB_NAME}_job"

    def test_every_job_replaces_an_existing_registration(self, scheduler: RecordingScheduler) -> None:
        """A reload that added a second copy of each job would double every write."""
        assert [job.job_id for job in scheduler.jobs if not job.replace_existing] == []

    def test_every_job_uses_the_proxy_misfire_grace(self, scheduler: RecordingScheduler) -> None:
        assert {job.misfire_grace_time for job in scheduler.jobs} == {APSCHEDULER_MISFIRE_GRACE_TIME}

    def test_default_intraday_cadences(self, scheduler: RecordingScheduler) -> None:
        aggregation: Final = scheduler.by_id(AGGREGATION_JOB_ID)
        anomaly: Final = scheduler.by_id(ANOMALY_JOB_ID)
        assert (aggregation.trigger, aggregation.minutes) == ("interval", 5)
        assert (anomaly.trigger, anomaly.minutes) == ("interval", 15)

    @pytest.mark.parametrize(
        ("job_id", "hour"),
        [(QUOTA_MIRROR_JOB_ID, 1), (FORECAST_JOB_ID, 2), (QUALITY_JOB_ID, 3)],
    )
    def test_default_nightly_cadences(self, scheduler: RecordingScheduler, job_id: str, hour: int) -> None:
        """The mirror writes the quotas runway needs, and quality scores what forecast published."""
        job: Final = scheduler.by_id(job_id)
        assert (job.trigger, job.hour, job.minute) == ("cron", hour, 0)

    def test_nightly_jobs_are_anchored_to_utc(self, scheduler: RecordingScheduler) -> None:
        """The proxy's scheduler has no timezone, and the facts these jobs read are UTC days."""
        nightly: Final = [job for job in scheduler.jobs if job.trigger == "cron"]
        assert nightly
        assert {job.timezone for job in nightly} == {"UTC"}


class TestCadencesAreConfigurable:
    def test_env_overrides_are_honoured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(FINOPS_ENABLED_ENV, "true")
        monkeypatch.setenv(AGGREGATION_INTERVAL_ENV, "3")
        monkeypatch.setenv(ANOMALY_INTERVAL_ENV, "30")
        monkeypatch.setenv(QUOTA_MIRROR_HOUR_ENV, "20")
        monkeypatch.setenv(FORECAST_HOUR_ENV, "21")
        monkeypatch.setenv(QUALITY_HOUR_ENV, "0")
        scheduler: Final = scheduled(FakePrisma())
        assert scheduler.by_id(AGGREGATION_JOB_ID).minutes == 3
        assert scheduler.by_id(ANOMALY_JOB_ID).minutes == 30
        assert scheduler.by_id(QUOTA_MIRROR_JOB_ID).hour == 20
        assert scheduler.by_id(FORECAST_JOB_ID).hour == 21
        assert scheduler.by_id(QUALITY_JOB_ID).hour == 0

    @pytest.mark.parametrize("value", ["banana", "", "  ", "-5", "24", "99", "1.5"])
    def test_an_unusable_hour_falls_back_to_the_default(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        """A typo in an env var must not stop the proxy from starting."""
        monkeypatch.setenv(FINOPS_ENABLED_ENV, "true")
        monkeypatch.setenv(FORECAST_HOUR_ENV, value)
        assert scheduled(FakePrisma()).by_id(FORECAST_JOB_ID).hour == 2

    @pytest.mark.parametrize("value", ["banana", "0", "-1", "99999"])
    def test_an_unusable_interval_falls_back_to_the_default(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv(FINOPS_ENABLED_ENV, "true")
        monkeypatch.setenv(AGGREGATION_INTERVAL_ENV, value)
        assert scheduled(FakePrisma()).by_id(AGGREGATION_JOB_ID).minutes == 5

    def test_the_lateness_window_is_reported_to_the_operator(
        self, monkeypatch: pytest.MonkeyPatch, captured_logs: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv(FINOPS_ENABLED_ENV, "true")
        monkeypatch.setenv(LATENESS_WINDOW_ENV, "6")
        scheduled(FakePrisma())
        assert "lateness window 6h" in captured_logs.text


class TestNoDatabase:
    def test_scheduling_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(FINOPS_ENABLED_ENV, "true")
        assert scheduled(None).jobs == []

    def test_the_reason_is_logged(
        self, monkeypatch: pytest.MonkeyPatch, captured_logs: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv(FINOPS_ENABLED_ENV, "true")
        scheduled(None)
        assert "no database" in captured_logs.text
        assert FINOPS_ENABLED_ENV in captured_logs.text

    def test_routes_still_mount_without_a_database(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The routes answer 503 themselves; refusing to mount them would hide the reason."""
        monkeypatch.setenv(FINOPS_ENABLED_ENV, "true")
        assert any(path.startswith(FINOPS_ROUTE_PREFIX) for path in mounted())


class TestFailureIsolation:
    """A forecasting bug must not become a gateway outage."""

    @pytest.fixture
    def broken(self, monkeypatch: pytest.MonkeyPatch) -> tuple[RecordingScheduler, FakePrisma]:
        monkeypatch.setenv(FINOPS_ENABLED_ENV, "true")
        prisma: Final = FakePrisma(RuntimeError("the lease table is gone"))
        return scheduled(prisma), prisma

    @pytest.mark.parametrize("job_id", ALL_JOB_IDS)
    async def test_a_job_that_raises_does_not_propagate(
        self, broken: tuple[RecordingScheduler, FakePrisma], job_id: str
    ) -> None:
        scheduler, prisma = broken
        assert await scheduler.by_id(job_id).func() is None
        assert prisma.db.statements, f"{job_id} never reached the database, so nothing was actually contained"

    @pytest.mark.parametrize("job_id", ALL_JOB_IDS)
    async def test_the_failure_is_logged_with_the_job_id(
        self,
        broken: tuple[RecordingScheduler, FakePrisma],
        captured_logs: pytest.LogCaptureFixture,
        job_id: str,
    ) -> None:
        await broken[0].by_id(job_id).func()
        assert job_id in captured_logs.text
        assert "the next tick will retry" in captured_logs.text

    async def test_a_second_tick_still_runs_after_a_failure(
        self, broken: tuple[RecordingScheduler, FakePrisma]
    ) -> None:
        """Containment is worthless if the job is left dead; the next tick must try again."""
        scheduler, prisma = broken
        tick: Final = scheduler.by_id(AGGREGATION_JOB_ID).func
        await tick()
        first: Final = len(prisma.db.statements)
        await tick()
        assert len(prisma.db.statements) > first

    async def test_containment_does_not_swallow_cancellation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A shutdown must still be able to stop a running job."""
        monkeypatch.setenv(FINOPS_ENABLED_ENV, "true")
        scheduler: Final = scheduled(FakePrisma(asyncio.CancelledError()))
        with pytest.raises(asyncio.CancelledError):
            await scheduler.by_id(AGGREGATION_JOB_ID).func()


class TestAgainstRealApscheduler:
    """The recording double proves intent; this proves APScheduler accepts it."""

    @pytest.fixture
    def scheduler(self, monkeypatch: pytest.MonkeyPatch) -> Any:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler

        monkeypatch.setenv(FINOPS_ENABLED_ENV, "true")
        instance: Final = AsyncIOScheduler()
        schedule_witos_jobs(instance, FakePrisma())  # pyright: ignore[reportArgumentType]  # test double
        return instance

    def test_every_job_is_accepted(self, scheduler: Any) -> None:
        assert sorted(job.id for job in scheduler.get_jobs()) == sorted(ALL_JOB_IDS)

    @pytest.mark.parametrize(
        ("job_id", "minutes"), [(AGGREGATION_JOB_ID, 5), (ANOMALY_JOB_ID, 15)]
    )
    def test_interval_triggers_resolve_to_the_cadence(self, scheduler: Any, job_id: str, minutes: int) -> None:
        trigger: Final = next(job.trigger for job in scheduler.get_jobs() if job.id == job_id)
        assert trigger.interval == timedelta(minutes=minutes)

    @pytest.mark.parametrize(
        ("job_id", "hour"), [(QUOTA_MIRROR_JOB_ID, 1), (FORECAST_JOB_ID, 2), (QUALITY_JOB_ID, 3)]
    )
    def test_cron_triggers_resolve_to_the_hour_in_utc(self, scheduler: Any, job_id: str, hour: int) -> None:
        """APScheduler resolved the timezone string, so the nightly runs are not on container-local time."""
        trigger: Final = next(job.trigger for job in scheduler.get_jobs() if job.id == job_id)
        assert str(trigger.timezone) == "UTC"
        assert repr(trigger) == f"<CronTrigger (hour='{hour}', minute='0', timezone='UTC')>"


_INERTNESS_SCRIPT: Final = """
import sys
from fastapi import FastAPI
import litellm.proxy.witos.registration as reg

reg.include_witos_routers(FastAPI())
reg.schedule_witos_jobs(None, None)

leaked = sorted(
    name
    for name in sys.modules
    if name.startswith("litellm.proxy.witos.") and name != "litellm.proxy.witos.registration"
)
print(",".join(leaked))
sys.exit(1 if leaked else 0)
"""


def test_flags_off_imports_none_of_the_witos_packages() -> None:
    """Off means inert, not idle.

    The first deploy runs this image with both flags unset, so a defect anywhere
    in `finops/` or `policy_fabric/` must be unreachable, import errors included.
    A subprocess because the rest of this suite has already imported them.
    """
    env: Final = {key: value for key, value in os.environ.items() if key not in _ALL_ENV}
    result: Final = subprocess.run(
        [sys.executable, "-c", _INERTNESS_SCRIPT], capture_output=True, text=True, env=env, check=False
    )
    assert result.returncode == 0, f"imported with the flags off: {result.stdout.strip()}{result.stderr}"
