"""Pricing forecast usage through LiteLLM's own cost calculator (blueprint §0.1, §1.3).

There is exactly one price catalog in this system and WIT OS does not own it.
Every unit price here comes back from ``litellm.cost_calculator.cost_per_token``,
asked for a single token, which is the same function that priced the request when
it happened. A second catalog would drift from the first within a quarter and the
forecast would start disagreeing with the invoice for reasons nobody could trace.

Two things follow from forecasting usage and pricing it afterwards:

A forecast is reproducible only if the prices it used are recoverable, so every
forecast stores a ``pricing_snapshot_hash`` over the exact prices that produced
it. When a provider changes a price the hash changes, which is what makes "the
forecast moved because the price map moved" a detectable event rather than an
unexplained jump.

A model the cost map has never heard of is priced at zero, and that is a lie the
caller has to be told about rather than protected from. ``unknown_models`` and
``fidelity`` exist so the engine can refuse to publish a token-based forecast
whose prices do not reproduce the spend that actually happened, and fall back to
forecasting cost directly (§1.3's ``ACTUAL_COST_BASED``).
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType
from typing import Final

from litellm._logging import verbose_proxy_logger

# Reconstructed spend within this band of actual spend means the price book
# explains the invoice, so usage may be forecast and priced. Outside it something
# the price map cannot see is driving the bill (a provider service charge, an
# unmapped alias, a negotiated rate) and the honest move is to forecast the cost
# series itself.
FIDELITY_TOLERANCE: Final = Decimal("0.10")

_UNIT_TOKENS: Final = 1


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """Per-token USD prices for one model group, as LiteLLM prices it today."""

    model: str
    input_per_token: Decimal
    output_per_token: Decimal
    cache_read_per_token: Decimal
    known: bool

    def as_snapshot(self) -> tuple[str, str, str, str, bool]:
        return (
            self.model,
            str(self.input_per_token),
            str(self.output_per_token),
            str(self.cache_read_per_token),
            self.known,
        )


@dataclass(frozen=True, slots=True)
class BlendedPrice:
    """Mix-weighted unit prices: what one more token of each kind costs this scope."""

    input_per_token: Decimal
    output_per_token: Decimal
    cache_read_per_token: Decimal

    def cost_of(self, *, prompt_tokens: Decimal, completion_tokens: Decimal, cache_read_tokens: Decimal) -> Decimal:
        return (
            prompt_tokens * self.input_per_token
            + completion_tokens * self.output_per_token
            + cache_read_tokens * self.cache_read_per_token
        )

    def scaled(self, factor: Decimal) -> "BlendedPrice":
        return BlendedPrice(
            input_per_token=self.input_per_token * factor,
            output_per_token=self.output_per_token * factor,
            cache_read_per_token=self.cache_read_per_token * factor,
        )


ZERO_PRICE: Final = BlendedPrice(
    input_per_token=Decimal(0), output_per_token=Decimal(0), cache_read_per_token=Decimal(0)
)


@dataclass(frozen=True, slots=True)
class PriceBook:
    """The prices a forecast was computed with, and the hash that pins them."""

    prices: Mapping[str, ModelPrice]
    snapshot_hash: str

    @property
    def unknown_models(self) -> frozenset[str]:
        return frozenset(name for name, price in self.prices.items() if not price.known)

    def blend(self, mix_shares: Mapping[str, float]) -> BlendedPrice:
        """Weight each model's unit prices by its share of the scope's tokens."""
        total: Final = sum(mix_shares.values())
        if total <= 0:
            return ZERO_PRICE
        weights: Final = MappingProxyType({model: Decimal(str(share / total)) for model, share in mix_shares.items()})
        return BlendedPrice(
            input_per_token=self._weighted(weights, "input"),
            output_per_token=self._weighted(weights, "output"),
            cache_read_per_token=self._weighted(weights, "cache_read"),
        )

    def _weighted(self, weights: Mapping[str, Decimal], component: str) -> Decimal:
        return sum(
            (weight * _component(self.prices.get(model), component) for model, weight in weights.items()),
            Decimal(0),
        )

    def unknown_share(self, mix_shares: Mapping[str, float]) -> float:
        total: Final = sum(mix_shares.values())
        if total <= 0:
            return 0.0
        unknown: Final = self.unknown_models
        return sum(share for model, share in mix_shares.items() if model in unknown) / total

    def with_adjustments(self, adjustments: Mapping[str, tuple[Decimal, Decimal]]) -> "PriceBook":
        """A scenario's price overrides, expressed as multipliers on LiteLLM's prices.

        Deltas rather than absolute prices, so a what-if about a 20% discount
        stays anchored to the real catalog instead of forking it.
        """
        return build_snapshot(
            tuple(
                _adjusted(price, adjustments[model]) if model in adjustments else price
                for model, price in self.prices.items()
            )
        )


def _component(price: ModelPrice | None, component: str) -> Decimal:
    if price is None:
        return Decimal(0)
    match component:
        case "input":
            return price.input_per_token
        case "output":
            return price.output_per_token
        case _:
            return price.cache_read_per_token


def _adjusted(price: ModelPrice, factors: tuple[Decimal, Decimal]) -> ModelPrice:
    input_factor, output_factor = factors
    return ModelPrice(
        model=price.model,
        input_per_token=price.input_per_token * input_factor,
        output_per_token=price.output_per_token * output_factor,
        cache_read_per_token=price.cache_read_per_token * input_factor,
        known=price.known,
    )


def build_snapshot(prices: Sequence[ModelPrice]) -> PriceBook:
    ordered: Final = [  # mutable-ok: json.dumps takes a list, and it is serialised before it can escape
        price.as_snapshot() for price in sorted(prices, key=lambda price: price.model)
    ]
    snapshot: Final = json.dumps(ordered, separators=(",", ":"))
    return PriceBook(
        prices=MappingProxyType({price.model: price for price in prices}),
        snapshot_hash=hashlib.sha256(snapshot.encode()).hexdigest(),
    )


def _unit_prices(model: str) -> tuple[Decimal, Decimal, Decimal] | None:
    """Ask LiteLLM what one input, output and cached-read token costs for ``model``."""
    from litellm.cost_calculator import (
        cost_per_token,  # pyright: ignore[reportUnknownVariableType]  # the cost calculator carries an untyped parameter
    )

    try:
        prompt_cost, completion_cost = cost_per_token(  # pyright: ignore[reportUnknownMemberType]  # the cost calculator's kwargs are untyped
            model=model, prompt_tokens=_UNIT_TOKENS, completion_tokens=_UNIT_TOKENS
        )
        cached_cost, _ = cost_per_token(  # pyright: ignore[reportUnknownMemberType]  # the cost calculator's kwargs are untyped
            model=model, prompt_tokens=0, completion_tokens=0, cache_read_input_tokens=_UNIT_TOKENS
        )
    except Exception as error:  # noqa: BLE001  # an unpriceable model is data about the mix, not a failure
        verbose_proxy_logger.debug("WIT OS FinOps cannot price %s: %s", model, error)
        return None
    return Decimal(str(prompt_cost)), Decimal(str(completion_cost)), Decimal(str(cached_cost))


def build_price_book(models: Sequence[str]) -> PriceBook:
    """Resolve every model in the mix, marking the ones LiteLLM cannot price."""
    return build_snapshot(tuple(_price_of(model) for model in sorted(frozenset(models))))


def _price_of(model: str) -> ModelPrice:
    resolved: Final = _unit_prices(model)
    if resolved is None:
        return ModelPrice(
            model=model,
            input_per_token=Decimal(0),
            output_per_token=Decimal(0),
            cache_read_per_token=Decimal(0),
            known=False,
        )
    input_price, output_price, cache_price = resolved
    return ModelPrice(
        model=model,
        input_per_token=input_price,
        output_per_token=output_price,
        cache_read_per_token=cache_price,
        known=True,
    )


@dataclass(frozen=True, slots=True)
class PricingFidelity:
    """How closely the price book reproduces spend that already happened."""

    reconstructed: Decimal
    actual: Decimal

    @property
    def ratio(self) -> Decimal | None:
        return None if self.actual <= 0 else self.reconstructed / self.actual

    @property
    def is_trustworthy(self) -> bool:
        """True when reconstruction is close enough to the invoice to price a forecast with."""
        ratio: Final = self.ratio
        if ratio is None:
            return self.reconstructed <= 0
        return abs(ratio - Decimal(1)) <= FIDELITY_TOLERANCE

    def calibration(self) -> Decimal:
        """The multiplier that makes repriced usage reproduce the spend already booked.

        Providers disagree about whether cached reads are inside ``prompt_tokens``
        or beside them, and a scope can carry a negotiated rate the public price
        map has never seen. Rather than encode a per-provider convention that
        would be wrong for the next provider, the price book is calibrated
        against this scope's own invoice. The calibration is only ever applied
        inside the fidelity tolerance: further out than that, something the price
        map cannot see is driving the bill, and the strategy falls back to
        forecasting cost directly instead of scaling a wrong model until it fits.
        """
        if self.reconstructed <= 0 or not self.is_trustworthy:
            return Decimal(1)
        return self.actual / self.reconstructed
