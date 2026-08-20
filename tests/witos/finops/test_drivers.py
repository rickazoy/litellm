"""Driver decomposition: each factor moved on its own, and the residual kept honest."""

from datetime import date, timedelta
from decimal import Decimal

import pytest

from litellm.proxy.witos.finops.drivers import DRIVER_ORDER, decompose
from litellm.proxy.witos.finops.pricing import build_price_book

from tests.witos.finops.fakes import priced_day

START = date(2026, 7, 1)
CHEAP = "gpt-4o-mini"
DEAR = "gpt-5"


def window(days: int, *, offset: int = 0, prompt: float, completion: float, cache: float = 0.0, model: str = CHEAP):
    return tuple(
        priced_day(
            START + timedelta(days=offset + step), prompt=prompt, completion=completion, cache=cache, model=model
        )
        for step in range(days)
    )


def book(*models: str):
    return build_price_book(models or (CHEAP,))


def contribution(result, factor: str) -> Decimal:
    return next(item.delta_usd for item in result.contributions if item.factor == factor)


def test_a_pure_volume_increase_is_attributed_to_volume() -> None:
    baseline = window(7, prompt=100_000.0, completion=20_000.0)
    current = window(7, offset=7, prompt=200_000.0, completion=40_000.0)
    result = decompose(baseline, current, book())

    assert result.delta_pct == pytest.approx(100.0, rel=0.01)
    assert contribution(result, "request_volume") > 0
    assert contribution(result, "prompt_size") == 0
    assert contribution(result, "model_mix") == 0


def test_a_bigger_prompt_at_constant_volume_is_attributed_to_prompt_size() -> None:
    baseline = tuple(
        priced_day(START + timedelta(days=step), prompt=100_000.0, completion=20_000.0, cache=0.0, requests=50.0)
        for step in range(7)
    )
    current = tuple(
        priced_day(START + timedelta(days=7 + step), prompt=300_000.0, completion=20_000.0, cache=0.0, requests=50.0)
        for step in range(7)
    )
    result = decompose(baseline, current, book())

    assert contribution(result, "prompt_size") > 0
    assert contribution(result, "prompt_size") > contribution(result, "completion_size")


def test_moving_traffic_onto_an_expensive_model_shows_up_as_mix() -> None:
    baseline = window(7, prompt=100_000.0, completion=20_000.0, model=CHEAP)
    current = window(7, offset=7, prompt=100_000.0, completion=20_000.0, model=DEAR)
    result = decompose(baseline, current, book(CHEAP, DEAR))

    assert contribution(result, "model_mix") > 0
    assert contribution(result, "request_volume") == 0


def test_a_provider_price_change_mid_period_lands_on_the_price_line() -> None:
    """§1.14: the same usage costing more is a price move, not a usage move."""
    baseline = window(7, prompt=100_000.0, completion=20_000.0)
    current = tuple(
        priced_day(START + timedelta(days=7 + step), prompt=100_000.0, completion=20_000.0, cache=0.0)
        for step in range(7)
    )
    doubled = tuple(
        type(day)(
            day=day.day,
            request_count=day.request_count,
            prompt_tokens=day.prompt_tokens,
            completion_tokens=day.completion_tokens,
            cache_read_tokens=day.cache_read_tokens,
            spend_usd=day.spend_usd * 2,
            mix_tokens=day.mix_tokens,
        )
        for day in current
    )
    result = decompose(baseline, doubled, book())

    assert contribution(result, "price_map") > 0
    assert contribution(result, "request_volume") == 0
    assert result.delta_pct == pytest.approx(100.0, rel=0.01)


def test_the_contributions_and_the_residual_add_up_to_the_actual_change() -> None:
    """Attributing interaction terms to a named factor would read better and be wrong."""
    baseline = window(7, prompt=100_000.0, completion=20_000.0)
    current = window(7, offset=7, prompt=260_000.0, completion=90_000.0, cache=30_000.0)
    result = decompose(baseline, current, book())

    total = sum((item.delta_usd for item in result.contributions), Decimal(0))
    assert total == pytest.approx(result.delta_usd, rel=1e-9)


def test_every_factor_is_reported_even_when_it_did_not_move() -> None:
    baseline = window(7, prompt=100_000.0, completion=20_000.0)
    result = decompose(baseline, baseline, book())

    assert tuple(item.factor for item in result.contributions) == (*DRIVER_ORDER, "residual")
    assert result.delta_usd == 0
    assert all(item.delta_usd == 0 for item in result.contributions)


def test_windows_of_different_lengths_are_compared_as_daily_rates() -> None:
    baseline = window(7, prompt=100_000.0, completion=20_000.0)
    current = window(3, offset=7, prompt=100_000.0, completion=20_000.0)
    result = decompose(baseline, current, book())

    assert result.delta_usd == pytest.approx(Decimal(0), abs=Decimal("0.000001"))
