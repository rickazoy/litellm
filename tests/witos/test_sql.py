"""``PrismaSqlExecutor`` binds have to survive Prisma's own encoder.

Every FinOps read takes a calendar day and every statement casts it to
``timestamptz``, so ``date`` objects reach ``query_raw`` from a dozen call
sites. Prisma JSON-encodes each argument through a ``singledispatch`` registry
that covers ``datetime``, ``Decimal``, ``Json`` and ``Base64`` and nothing else,
so an unpromoted ``date`` raises at request time and never at import time.

That is exactly how this shipped: the unit suite injected a fake executor and
the integration suite ran raw psycopg, so neither ever put a bind through the
encoder that production uses. ``/witos/finops/overview``, ``/drivers`` and
``/report`` all returned 500 the moment the flag was flipped on the live
gateway. The test below therefore calls the real ``prisma.builder.serializer``
rather than asserting against a fake that would accept anything.
"""

from datetime import date, datetime, timezone
from decimal import Decimal

import pytest

from litellm.proxy.witos.shared.sql import PrismaSqlExecutor, bind

_builder = pytest.importorskip("prisma.builder", reason="prisma-client-py is not installed in this environment")
# ``dumps`` and not ``serializer``: the registry is only json's ``default=``
# fallback, so calling it directly would reject the ``str`` and ``int`` binds
# that json handles natively and prove nothing about the real encode path.
prisma_dumps = _builder.dumps


class RecordingClient:
    """Captures what the executor actually handed to Prisma."""

    def __init__(self) -> None:
        self.args: tuple[object, ...] = ()

    async def query_raw(self, query: str, *args: object) -> list[dict[str, object]]:
        self.args = args
        return []

    async def execute_raw(self, query: str, *args: object) -> int:
        self.args = args
        return 0


def test_bind_promotes_a_date_to_utc_midnight() -> None:
    promoted = bind(date(2026, 8, 20))
    assert promoted == datetime(2026, 8, 20, tzinfo=timezone.utc)


def test_bind_leaves_a_datetime_alone() -> None:
    """``datetime`` subclasses ``date``; an inclusive check would flatten every timestamp to midnight."""
    exact = datetime(2026, 8, 20, 14, 31, 9, tzinfo=timezone.utc)
    assert bind(exact) is exact


@pytest.mark.parametrize("value", [Decimal("12.34"), "global", 7, None, True])
def test_bind_passes_through_types_prisma_already_handles(value: object) -> None:
    assert bind(value) is value


@pytest.mark.parametrize(
    "value",
    [date(2026, 8, 20), datetime(2026, 8, 20, 14, 31, tzinfo=timezone.utc), Decimal("1.5"), "global", 3],
)
def test_every_bound_argument_is_serializable_by_prisma(value: object) -> None:
    """The assertion that would have caught this before it reached the gateway."""
    prisma_dumps(bind(value))


def test_a_raw_date_is_still_rejected_by_prisma() -> None:
    """Pins the reason ``bind`` exists. If prisma-client-py ever registers ``date``, this fails and the promotion can go."""
    with pytest.raises(TypeError, match="not serializable"):
        prisma_dumps(date(2026, 8, 20))


@pytest.mark.asyncio
async def test_query_promotes_dates_before_they_reach_the_client() -> None:
    client = RecordingClient()
    await PrismaSqlExecutor(client).query("SELECT 1", date(2026, 8, 1), date(2026, 8, 20), ["global"])
    assert client.args == (
        datetime(2026, 8, 1, tzinfo=timezone.utc),
        datetime(2026, 8, 20, tzinfo=timezone.utc),
        ["global"],
    )
    prisma_dumps(list(client.args))


@pytest.mark.asyncio
async def test_execute_promotes_dates_too() -> None:
    client = RecordingClient()
    await PrismaSqlExecutor(client).execute("DELETE FROM t WHERE d < $1", date(2026, 1, 1))
    assert client.args == (datetime(2026, 1, 1, tzinfo=timezone.utc),)
