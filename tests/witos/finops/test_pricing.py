"""Pricing through LiteLLM's calculator: unknown models, zero-cost models, and the snapshot hash."""

from decimal import Decimal

from litellm.proxy.witos.finops.pricing import (
    ModelPrice,
    PricingFidelity,
    build_price_book,
    build_snapshot,
)

KNOWN = "gpt-4o-mini"
UNKNOWN = "totally-made-up-model-xyz"


def price(model: str, *, input_: str, output: str, cache: str = "0", known: bool = True) -> ModelPrice:
    return ModelPrice(
        model=model,
        input_per_token=Decimal(input_),
        output_per_token=Decimal(output),
        cache_read_per_token=Decimal(cache),
        known=known,
    )


def test_prices_come_from_the_cost_calculator_not_a_local_catalog() -> None:
    book = build_price_book((KNOWN,))

    assert book.prices[KNOWN].known is True
    assert book.prices[KNOWN].input_per_token > 0
    assert book.prices[KNOWN].output_per_token > book.prices[KNOWN].input_per_token


def test_an_unknown_model_is_recorded_rather_than_silently_priced_at_zero() -> None:
    book = build_price_book((KNOWN, UNKNOWN))

    assert book.unknown_models == frozenset({UNKNOWN})
    assert book.prices[UNKNOWN].input_per_token == 0
    assert book.unknown_share({KNOWN: 0.75, UNKNOWN: 0.25}) == 0.25


def test_a_zero_cost_model_is_known_and_prices_at_zero() -> None:
    book = build_snapshot((price("free-local-model", input_="0", output="0"),))
    blended = book.blend({"free-local-model": 1.0})

    assert book.unknown_models == frozenset()
    assert (
        blended.cost_of(
            prompt_tokens=Decimal(1_000_000), completion_tokens=Decimal(1_000_000), cache_read_tokens=Decimal(0)
        )
        == 0
    )


def test_the_snapshot_hash_changes_when_a_provider_price_changes() -> None:
    before = build_snapshot((price("m", input_="0.000001", output="0.000002"),))
    after = build_snapshot((price("m", input_="0.0000012", output="0.000002"),))
    same = build_snapshot((price("m", input_="0.000001", output="0.000002"),))

    assert before.snapshot_hash != after.snapshot_hash
    assert before.snapshot_hash == same.snapshot_hash


def test_the_hash_does_not_depend_on_the_order_models_were_resolved_in() -> None:
    one = build_snapshot((price("a", input_="1", output="2"), price("b", input_="3", output="4")))
    other = build_snapshot((price("b", input_="3", output="4"), price("a", input_="1", output="2")))

    assert one.snapshot_hash == other.snapshot_hash


def test_blending_weights_each_model_by_its_share_of_the_mix() -> None:
    book = build_snapshot(
        (price("cheap", input_="0.000001", output="0.000002"), price("dear", input_="0.00001", output="0.00002"))
    )
    blended = book.blend({"cheap": 0.5, "dear": 0.5})

    assert blended.input_per_token == Decimal("0.0000055")


def test_shifting_the_mix_towards_the_cheaper_model_lowers_the_blended_price() -> None:
    book = build_snapshot(
        (price("cheap", input_="0.000001", output="0.000002"), price("dear", input_="0.00001", output="0.00002"))
    )

    assert (
        book.blend({"cheap": 0.9, "dear": 0.1}).input_per_token
        < book.blend({"cheap": 0.1, "dear": 0.9}).input_per_token
    )


def test_scenario_price_adjustments_are_deltas_on_the_real_catalog() -> None:
    book = build_snapshot((price("m", input_="0.000010", output="0.000020"),))
    discounted = book.with_adjustments({"m": (Decimal("0.8"), Decimal("0.8"))})

    assert discounted.prices["m"].input_per_token == Decimal("0.000008")
    assert discounted.snapshot_hash != book.snapshot_hash


def test_fidelity_calibrates_inside_the_tolerance_and_refuses_outside_it() -> None:
    """The calibration exists so provider cache-token conventions do not silently skew a forecast."""
    close = PricingFidelity(reconstructed=Decimal("105"), actual=Decimal("100"))
    far = PricingFidelity(reconstructed=Decimal("300"), actual=Decimal("100"))

    assert close.is_trustworthy is True
    assert close.calibration() < 1
    assert far.is_trustworthy is False
    assert far.calibration() == 1


def test_a_scope_with_no_spend_and_no_reconstruction_is_trustworthy() -> None:
    empty = PricingFidelity(reconstructed=Decimal(0), actual=Decimal(0))

    assert empty.is_trustworthy is True
    assert empty.calibration() == 1
