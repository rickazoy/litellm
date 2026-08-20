"""Operator-owned FinOps settings and aggregation watermarks.

Backed by ``WITOS_FinOpsConfig``, which follows the fork's existing
``LiteLLM_Config`` convention (a keyed row with a JSON payload). Every read is
validated into a concrete type here, so nothing downstream handles loose JSON.

The watermarks live here rather than in their own table because they are exactly
what this table is: a small piece of mutable operator/job state that must
survive a pod restart.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from decimal import Decimal
from types import MappingProxyType
from typing import Final, TypedDict

from pydantic import TypeAdapter
from typing_extensions import ReadOnly

from litellm.proxy.witos.shared.sql import SqlExecutor

APPROVED_TAG_DIMENSIONS_KEY: Final = "approved_tag_dimensions"
FX_RATE_KEY: Final = "fx_rate"
HOURLY_WATERMARK_KEY: Final = "watermark.usage_hourly"

# Blueprint §1.4: tag-scope aggregates exist only for these dimensions. Aggregating
# every tag combination is what turns a fact table into a cardinality bomb.
DEFAULT_APPROVED_TAG_DIMENSIONS: Final = (
    "environment",
    "application",
    "cost_center",
    "business_unit",
    "project",
    "customer",
)

# A tag is a FinOps dimension only when it is written `dimension:value`.
TAG_DIMENSION_SEPARATOR: Final = ":"

_STRINGS: Final = TypeAdapter(tuple[str, ...])
_FX_RATES: Final = TypeAdapter(dict[str, Decimal])
_TIMESTAMP: Final = TypeAdapter(datetime)
_USD_ONLY: Final[Mapping[str, Decimal]] = MappingProxyType({"USD": Decimal(1)})


class _ConfigRow(TypedDict):
    value_json: ReadOnly[object]


_CONFIG_ROWS: Final = TypeAdapter(tuple[_ConfigRow, ...])

_READ_SQL: Final = 'SELECT value_json FROM "WITOS_FinOpsConfig" WHERE config_key = $1'

_WRITE_SQL: Final = """
INSERT INTO "WITOS_FinOpsConfig" (config_key, value_json, created_at, updated_at)
VALUES ($1, $2::jsonb, (now() AT TIME ZONE 'UTC'), (now() AT TIME ZONE 'UTC'))
ON CONFLICT (config_key) DO UPDATE SET value_json = EXCLUDED.value_json, updated_at = (now() AT TIME ZONE 'UTC')
"""


class FinOpsConfig:
    """Typed accessor over ``WITOS_FinOpsConfig``."""

    def __init__(self, executor: SqlExecutor) -> None:
        self._executor: Final = executor

    async def _read(self, key: str) -> object | None:
        rows: Final = _CONFIG_ROWS.validate_python(await self._executor.query(_READ_SQL, key))
        return rows[0]["value_json"] if rows else None

    async def _write(self, key: str, value: str) -> None:
        await self._executor.execute(_WRITE_SQL, key, value)

    async def approved_tag_dimensions(self) -> tuple[str, ...]:
        raw: Final = await self._read(APPROVED_TAG_DIMENSIONS_KEY)
        return DEFAULT_APPROVED_TAG_DIMENSIONS if raw is None else _STRINGS.validate_python(raw)

    async def set_approved_tag_dimensions(self, dimensions: Sequence[str]) -> None:
        await self._write(APPROVED_TAG_DIMENSIONS_KEY, _STRINGS.dump_json(tuple(dimensions)).decode())

    async def fx_rates(self) -> Mapping[str, Decimal]:
        """Display-only conversion rates from USD. Internal money stays USD (blueprint §1.14)."""
        raw: Final = await self._read(FX_RATE_KEY)
        return _USD_ONLY if raw is None else MappingProxyType(_FX_RATES.validate_python(raw))

    async def watermark(self, key: str) -> datetime | None:
        raw: Final = await self._read(key)
        if raw is None:
            return None
        parsed: Final = _TIMESTAMP.validate_python(raw)
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)

    async def set_watermark(self, key: str, value: datetime) -> None:
        await self._write(key, _TIMESTAMP.dump_json(value).decode())
