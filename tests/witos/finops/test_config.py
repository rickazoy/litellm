"""Typed access to operator settings and the aggregation watermark."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from litellm.proxy.witos.finops.config import (
    DEFAULT_APPROVED_TAG_DIMENSIONS,
    FX_RATE_KEY,
    HOURLY_WATERMARK_KEY,
    FinOpsConfig,
)

MOMENT = datetime(2026, 8, 12, 9, 0, tzinfo=timezone.utc)


async def test_defaults_apply_when_nothing_is_configured(executor):
    config = FinOpsConfig(executor)
    assert await config.approved_tag_dimensions() == DEFAULT_APPROVED_TAG_DIMENSIONS
    assert await config.fx_rates() == {"USD": Decimal(1)}
    assert await config.watermark(HOURLY_WATERMARK_KEY) is None


async def test_settings_round_trip(executor):
    config = FinOpsConfig(executor)
    await config.set_approved_tag_dimensions(("environment", "squad"))
    assert await config.approved_tag_dimensions() == ("environment", "squad")


async def test_writing_a_setting_twice_updates_in_place(executor, connection):
    config = FinOpsConfig(executor)
    await config.set_approved_tag_dimensions(("a",))
    await config.set_approved_tag_dimensions(("b",))

    rows = await connection.execute('SELECT count(*) FROM "WITOS_FinOpsConfig"')
    assert (await rows.fetchone())[0] == 1
    assert await config.approved_tag_dimensions() == ("b",)


async def test_the_watermark_round_trips_as_utc(executor):
    config = FinOpsConfig(executor)
    await config.set_watermark(HOURLY_WATERMARK_KEY, MOMENT)
    assert await config.watermark(HOURLY_WATERMARK_KEY) == MOMENT


async def test_a_corrupt_setting_is_rejected_rather_than_silently_coerced(executor, connection):
    """Config is operator-editable, so a bad value must fail loudly, not become a default."""
    await connection.execute(
        """
        INSERT INTO "WITOS_FinOpsConfig" (config_key, value_json, updated_at)
        VALUES (%s, '{"USD": "not-a-number"}'::jsonb, now())
        """,
        (FX_RATE_KEY,),
    )

    with pytest.raises(ValidationError):
        await FinOpsConfig(executor).fx_rates()
