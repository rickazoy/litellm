"""The only place ``proxy_server.py`` learns that WIT OS exists (blueprint §0.3).

Every conditional lives here, so the proxy gains two call sites and nothing else
and an upstream merge stays two trivial hunks.

Both flags default off, and off means inert rather than idle: no routes, no jobs,
and the FinOps and policy-fabric packages are never imported at all. A build with
the flags unset therefore behaves exactly like one without this fork's code on
the request path, which is the condition the first deploy is tested under. The
second deploy flips a flag, and if the new code misbehaves the operator flips it
back and restarts. No image rollback.

A job that raises is logged and dropped, never re-raised. Forecasting is a
reporting concern, and a reporting concern that can stop the gateway from serving
inference is a worse outage than having no forecast. Failure isolation here is
what makes that impossible: APScheduler never sees an exception from a WIT OS
job, so it can neither disable the job nor let one propagate out of a tick.

The lease each job takes (``witos/shared/scheduler.py``) is what makes the
cadences below safe on more than one pod. Every replica schedules every job; the
lease decides which replica actually does the work that cycle.
"""

import os
from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import TYPE_CHECKING, Final, Protocol

from litellm._logging import verbose_proxy_logger
from litellm.constants import APSCHEDULER_MISFIRE_GRACE_TIME
from litellm.secret_managers.main import str_to_bool

if TYPE_CHECKING:
    from fastapi import FastAPI

    from litellm.proxy.utils import PrismaClient
    from litellm.proxy.witos.finops.anomaly import AnomalyDetector
    from litellm.proxy.witos.shared.scheduler import JobLeaseManager

FINOPS_ENABLED_ENV: Final = "WITOS_FINOPS_ENABLED"
DLP_ENABLED_ENV: Final = "WITOS_DLP_ENABLED"

LATENESS_WINDOW_ENV: Final = "WITOS_FINOPS_LATENESS_WINDOW_H"
AGGREGATION_INTERVAL_ENV: Final = "WITOS_FINOPS_AGGREGATION_INTERVAL_MIN"
ANOMALY_INTERVAL_ENV: Final = "WITOS_FINOPS_ANOMALY_INTERVAL_MIN"
QUOTA_MIRROR_HOUR_ENV: Final = "WITOS_FINOPS_QUOTA_MIRROR_HOUR_UTC"
FORECAST_HOUR_ENV: Final = "WITOS_FINOPS_FORECAST_HOUR_UTC"
QUALITY_HOUR_ENV: Final = "WITOS_FINOPS_QUALITY_HOUR_UTC"

DEFAULT_LATENESS_WINDOW_H: Final = 2
DEFAULT_AGGREGATION_INTERVAL_MIN: Final = 5
DEFAULT_ANOMALY_INTERVAL_MIN: Final = 15

# The three nightly jobs are ordered by what they read. The quota mirror runs
# first because runway is computed against the quotas it writes; quality runs
# last because it scores the forecasts the middle job published.
DEFAULT_QUOTA_MIRROR_HOUR_UTC: Final = 1
DEFAULT_FORECAST_HOUR_UTC: Final = 2
DEFAULT_QUALITY_HOUR_UTC: Final = 3

# Scheduler ids carry the `_job` suffix the proxy's own jobs use, over the lease
# names in `finops/`, so one grep finds a job in both the scheduler log and the
# lease table. `tests/witos/test_registration.py` pins them to those lease names.
AGGREGATION_JOB_ID: Final = "witos_finops_aggregation_job"
ANOMALY_JOB_ID: Final = "witos_finops_anomaly_job"
QUOTA_MIRROR_JOB_ID: Final = "witos_finops_quota_mirror_job"
FORECAST_JOB_ID: Final = "witos_finops_forecast_job"
QUALITY_JOB_ID: Final = "witos_finops_quality_job"

FINOPS_ROUTE_PREFIX: Final = "/witos/finops"
DLP_ROUTE_PREFIX: Final = "/witos/dlp"

# The proxy's scheduler runs with `timezone=None`, so a cron trigger would
# otherwise anchor on whatever the container's local zone happens to be. The
# facts, the forecast horizon and the "as of yesterday" the nightly jobs compute
# are all UTC days, so the triggers say so rather than inheriting a surprise.
_CRON_TIMEZONE: Final = "UTC"


class JobScheduler(Protocol):
    """The slice of APScheduler's ``AsyncIOScheduler`` this module uses.

    Narrow enough that a test can record the calls without a running scheduler,
    an event loop, or a database.
    """

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
    ) -> object: ...


def _flag(name: str) -> bool:
    """True only for an explicit ``true``. Anything else, unset included, is off."""
    return str_to_bool(os.environ.get(name)) is True


def _parse_int(raw: str) -> int | None:
    try:
        return int(raw.strip())
    except ValueError:
        return None


def _int_env(name: str, *, default: int, minimum: int, maximum: int) -> int:
    """A bounded integer knob that falls back loudly rather than crashing startup."""
    raw: Final = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    parsed: Final = _parse_int(raw)
    if parsed is None or not minimum <= parsed <= maximum:
        verbose_proxy_logger.warning(
            "WIT OS %s=%r is not an integer in [%d, %d]; using the default of %d", name, raw, minimum, maximum, default
        )
        return default
    return parsed


def _announce(subsystem: str, env_name: str, *, enabled: bool, detail: str) -> None:
    """One line per subsystem, always, naming the flag that decided it."""
    if enabled:
        verbose_proxy_logger.info("WIT OS %s ENABLED by %s=true: %s", subsystem, env_name, detail)
        return
    verbose_proxy_logger.info("WIT OS %s SKIPPED: %s is not set to true, so %s", subsystem, env_name, detail)


def include_witos_routers(app: "FastAPI") -> None:
    """Mount whichever WIT OS API surfaces their flags turn on.

    Both routers already carry their own ``/witos/*`` prefix, and the DLP router
    nests the retroactive-simulation router itself, so neither is given a prefix
    here and the retro routes are not mounted a second time.
    """
    finops_enabled: Final = _flag(FINOPS_ENABLED_ENV)
    dlp_enabled: Final = _flag(DLP_ENABLED_ENV)

    if finops_enabled:
        from litellm.proxy.witos.finops.router import router as finops_router

        app.include_router(finops_router)
    if dlp_enabled:
        from litellm.proxy.witos.policy_fabric.dlp_endpoints import router as dlp_router

        app.include_router(dlp_router)

    _announce(
        "FinOps API",
        FINOPS_ENABLED_ENV,
        enabled=finops_enabled,
        detail=f"{FINOPS_ROUTE_PREFIX}/* mounted" if finops_enabled else f"no {FINOPS_ROUTE_PREFIX}/* route exists",
    )
    _announce(
        "DLP API",
        DLP_ENABLED_ENV,
        enabled=dlp_enabled,
        detail=f"{DLP_ROUTE_PREFIX}/* mounted" if dlp_enabled else f"no {DLP_ROUTE_PREFIX}/* route exists",
    )


def schedule_witos_jobs(scheduler: JobScheduler, prisma_client: "PrismaClient | None") -> None:
    """Schedule the FinOps background jobs, or log why it did not.

    DLP has no background work: enforcement is a guardrail on the request path
    and its API reads what that guardrail wrote, so ``WITOS_DLP_ENABLED`` has
    nothing to decide here.
    """
    if not _flag(FINOPS_ENABLED_ENV):
        _announce("FinOps jobs", FINOPS_ENABLED_ENV, enabled=False, detail="no aggregation, anomaly or nightly job")
        return
    if prisma_client is None:
        verbose_proxy_logger.warning(
            "WIT OS FinOps is enabled by %s but the proxy has no database, so no jobs were scheduled. "
            "FinOps reads and writes Postgres exclusively; connect a database to the proxy.",
            FINOPS_ENABLED_ENV,
        )
        return

    from litellm.proxy.witos.finops.aggregator import FinOpsAggregator
    from litellm.proxy.witos.finops.alerts import AlertStore
    from litellm.proxy.witos.finops.anomaly import AnomalyDetector
    from litellm.proxy.witos.finops.jobs import ForecastJob, QualityJob
    from litellm.proxy.witos.finops.quotas import QuotaMirror
    from litellm.proxy.witos.shared.scheduler import JobLeaseManager
    from litellm.proxy.witos.shared.sql import PrismaSqlExecutor

    executor: Final = PrismaSqlExecutor.for_proxy(prisma_client)  # pyright: ignore[reportArgumentType]  # PrismaWrapper forwards query_raw/execute_raw through __getattr__, which a Protocol cannot see
    # One lease owner per pod rather than one per job, so the lease table reads as
    # "which replica holds what" instead of five unrelated uuids.
    leases: Final = JobLeaseManager(executor)
    alerts: Final = AlertStore(executor)

    aggregation_minutes: Final = _int_env(
        AGGREGATION_INTERVAL_ENV, default=DEFAULT_AGGREGATION_INTERVAL_MIN, minimum=1, maximum=1440
    )
    anomaly_minutes: Final = _int_env(
        ANOMALY_INTERVAL_ENV, default=DEFAULT_ANOMALY_INTERVAL_MIN, minimum=1, maximum=1440
    )
    lateness_hours: Final = _int_env(LATENESS_WINDOW_ENV, default=DEFAULT_LATENESS_WINDOW_H, minimum=1, maximum=72)
    mirror_hour: Final = _int_env(QUOTA_MIRROR_HOUR_ENV, default=DEFAULT_QUOTA_MIRROR_HOUR_UTC, minimum=0, maximum=23)
    forecast_hour: Final = _int_env(FORECAST_HOUR_ENV, default=DEFAULT_FORECAST_HOUR_UTC, minimum=0, maximum=23)
    quality_hour: Final = _int_env(QUALITY_HOUR_ENV, default=DEFAULT_QUALITY_HOUR_UTC, minimum=0, maximum=23)

    aggregator: Final = FinOpsAggregator(
        executor, lateness_window=timedelta(hours=lateness_hours), lease_manager=leases
    )
    detector: Final = AnomalyDetector(executor, alerts)
    mirror: Final = QuotaMirror(executor, lease_manager=leases)
    forecast: Final = ForecastJob(executor, alerts=alerts, lease_manager=leases)
    quality: Final = QualityJob(executor, lease_manager=leases)

    _add_interval(scheduler, AGGREGATION_JOB_ID, aggregator.run_cycle, minutes=aggregation_minutes)
    _add_interval(scheduler, ANOMALY_JOB_ID, _anomaly_sweep(detector, leases), minutes=anomaly_minutes)
    _add_nightly(scheduler, QUOTA_MIRROR_JOB_ID, mirror.run, hour=mirror_hour)
    _add_nightly(scheduler, FORECAST_JOB_ID, forecast.run, hour=forecast_hour)
    _add_nightly(scheduler, QUALITY_JOB_ID, quality.run, hour=quality_hour)

    verbose_proxy_logger.info(
        "WIT OS FinOps jobs ENABLED by %s=true: aggregation every %dmin (lateness window %dh), anomaly every %dmin, "
        "quota mirror %02d:00 UTC, forecast %02d:00 UTC, quality %02d:00 UTC. Every job takes a database lease, "
        "so exactly one pod runs each cycle.",
        FINOPS_ENABLED_ENV,
        aggregation_minutes,
        lateness_hours,
        anomaly_minutes,
        mirror_hour,
        forecast_hour,
        quality_hour,
    )


def _add_interval(scheduler: JobScheduler, job_id: str, run: Callable[[], Awaitable[object]], *, minutes: int) -> None:
    scheduler.add_job(
        _guarded(job_id, run),
        "interval",
        minutes=minutes,
        id=job_id,
        replace_existing=True,
        misfire_grace_time=APSCHEDULER_MISFIRE_GRACE_TIME,
    )


def _add_nightly(scheduler: JobScheduler, job_id: str, run: Callable[[], Awaitable[object]], *, hour: int) -> None:
    scheduler.add_job(
        _guarded(job_id, run),
        "cron",
        hour=hour,
        minute=0,
        timezone=_CRON_TIMEZONE,
        id=job_id,
        replace_existing=True,
        misfire_grace_time=APSCHEDULER_MISFIRE_GRACE_TIME,
    )


def _guarded(job_id: str, run: Callable[[], Awaitable[object]]) -> Callable[[], Awaitable[None]]:
    """One tick that can fail without taking the scheduler, or the gateway, with it."""

    async def tick() -> None:
        try:
            await run()
        except Exception:  # noqa: BLE001  # a reporting bug must never reach the scheduler; the next tick retries
            verbose_proxy_logger.exception(
                "WIT OS job %s failed this cycle and was contained; the next tick will retry", job_id
            )

    return tick


def _anomaly_sweep(detector: "AnomalyDetector", leases: "JobLeaseManager") -> Callable[[], Awaitable[None]]:
    """The anomaly detector under the same lease every other WIT OS job takes.

    ``AnomalyDetector`` is a reader plus a writer and holds no lease of its own,
    so without this it would be the one job that runs on every replica at once.
    """

    async def sweep() -> None:
        from litellm.proxy.witos.finops.jobs import ANOMALY_JOB_NAME
        from litellm.proxy.witos.shared.scheduler import LeaseHeld

        async with leases.hold(ANOMALY_JOB_NAME) as lease:
            if isinstance(lease, LeaseHeld):
                return
            raised: Final = await detector.run()
            verbose_proxy_logger.info("WIT OS FinOps anomaly sweep raised %d events", len(raised))

    return sweep
