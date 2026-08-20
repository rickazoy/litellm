"""Server-sent events for FinOps (blueprint §1.11).

Two things are worth pushing: an alert event that has just been raised, and a
forecast run that has just replaced what the client is looking at. Both are
polled from the database rather than published from the job, because the job runs
on one pod and the SSE connection is held by another, and a fan-out that needed
Redis would make live updates a deployment prerequisite instead of a feature.

The poll is cheap: two indexed lookups by timestamp, at a five second cadence, on
a connection a human is watching. A keep-alive comment goes out on every tick so
that an idle stream still survives proxies that time out silent connections.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Final, TypedDict

from pydantic import TypeAdapter
from typing_extensions import ReadOnly

from litellm.proxy.witos.finops.history import ScopeRef
from litellm.proxy.witos.shared.sql import SqlExecutor

DEFAULT_POLL_SECONDS: Final = 5.0
DEFAULT_MAX_DURATION: Final = timedelta(hours=1)

_EVENTS_SINCE_SQL: Final = """
SELECT id, created_at, alert_type, severity, scope_type, scope_id, title
FROM "WITOS_FinOpsAlertEvent"
WHERE created_at > ($1::timestamptz AT TIME ZONE 'UTC')
  AND ($2::text IS NULL OR scope_type = $2)
  AND ($3::text IS NULL OR scope_id = $3)
ORDER BY created_at
LIMIT 50
"""

_FORECASTS_SINCE_SQL: Final = """
SELECT MAX(generated_at) AS generated_at, COUNT(*)::int AS rows_written
FROM "WITOS_FinOpsForecast"
WHERE is_latest = true AND generated_at > ($1::timestamptz AT TIME ZONE 'UTC')
  AND ($2::text IS NULL OR scope_type = $2)
  AND ($3::text IS NULL OR scope_id = $3)
"""


class _EventRow(TypedDict):
    id: ReadOnly[str]
    created_at: ReadOnly[datetime]
    alert_type: ReadOnly[str]
    severity: ReadOnly[str]
    scope_type: ReadOnly[str]
    scope_id: ReadOnly[str]
    title: ReadOnly[str]


class _RefreshRow(TypedDict):
    generated_at: ReadOnly[datetime | None]
    rows_written: ReadOnly[int]


_EVENT_ROWS: Final = TypeAdapter(tuple[_EventRow, ...])
_REFRESH_ROWS: Final = TypeAdapter(tuple[_RefreshRow, ...])


class OpenFrame(TypedDict):
    since: ReadOnly[str]


class AlertFrame(TypedDict):
    id: ReadOnly[str]
    created_at: ReadOnly[str]
    alert_type: ReadOnly[str]
    severity: ReadOnly[str]
    scope_type: ReadOnly[str]
    scope_id: ReadOnly[str]
    title: ReadOnly[str]


class RefreshFrame(TypedDict):
    generated_at: ReadOnly[str]


def frame(event: str, data: Mapping[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(dict(data), default=str)}\n\n"  # mutable-ok: json.dumps takes a dict, and it is serialised before it can escape


def _alert_frame(event: "_EventRow") -> AlertFrame:
    rendered: Final[AlertFrame] = {
        "id": event["id"],
        "created_at": event["created_at"].isoformat(),
        "alert_type": event["alert_type"],
        "severity": event["severity"],
        "scope_type": event["scope_type"],
        "scope_id": event["scope_id"],
        "title": event["title"],
    }
    return rendered


def _refresh_frame(generated_at: datetime) -> RefreshFrame:
    rendered: Final[RefreshFrame] = {"generated_at": generated_at.isoformat()}
    return rendered


def _maybe(moment: datetime | None) -> tuple[datetime, ...]:
    return () if moment is None else (moment,)


class FinOpsStream:
    """Polls for new alert events and forecast refreshes and renders them as SSE frames."""

    def __init__(
        self,
        executor: SqlExecutor,
        *,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        max_duration: timedelta = DEFAULT_MAX_DURATION,
    ) -> None:
        self._executor: Final = executor
        self._poll_seconds: Final = poll_seconds
        self._max_duration: Final = max_duration

    async def _events_since(self, since: datetime, scope: ScopeRef | None) -> Sequence[_EventRow]:
        return _EVENT_ROWS.validate_python(
            await self._executor.query(
                _EVENTS_SINCE_SQL,
                since,
                scope.scope_type if scope else None,
                scope.scope_id if scope else None,
            )
        )

    async def _refresh_since(self, since: datetime, scope: ScopeRef | None) -> datetime | None:
        rows: Final = _REFRESH_ROWS.validate_python(
            await self._executor.query(
                _FORECASTS_SINCE_SQL,
                since,
                scope.scope_type if scope else None,
                scope.scope_id if scope else None,
            )
        )
        return rows[0]["generated_at"] if rows and rows[0]["rows_written"] else None

    async def frames(self, *, scope: ScopeRef | None = None, now: datetime | None = None) -> AsyncIterator[str]:
        """Yield SSE frames until the client disconnects or the connection ages out."""
        started: Final = now or datetime.now(timezone.utc)
        cursor = started  # rebind-ok: the stream cursor advances as frames are emitted
        opened: Final[OpenFrame] = {"since": started.isoformat()}
        yield frame("open", opened)
        while datetime.now(timezone.utc) - started < self._max_duration:
            await asyncio.sleep(self._poll_seconds)
            events = await self._events_since(cursor, scope)
            for event in events:
                yield frame("alert", _alert_frame(event))
            refreshed = await self._refresh_since(cursor, scope)
            if refreshed is not None:
                yield frame("forecast_refreshed", _refresh_frame(refreshed))
            cursor = max((cursor, *(event["created_at"] for event in events), *_maybe(refreshed)))
            yield ": keep-alive\n\n"
