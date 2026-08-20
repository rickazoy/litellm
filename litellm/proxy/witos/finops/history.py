"""The daily history a forecast is built from, and the ladder that decides what may be built.

Blueprint §1.3 forecasts *usage* and prices it afterwards, so the history this
module hands the engine is a set of usage series (requests, prompt tokens,
completion tokens, cache reads) plus the model mix, with spend carried alongside
only as the actual to reconcile against.

Two properties matter more than the shapes:

A missing day is a zero, not a gap. An entity that made no calls on Sunday spent
nothing on Sunday, and a forecaster that silently drops the row learns a seven
day week has six days in it. ``build_history`` therefore materialises every
calendar day between the first observation and ``as_of``.

The ladder (§1.6) is the honesty control. Under three days there is no forecast
at all, only a state, because three points cannot separate level from trend from
noise and a number produced from them is decoration. Each rung above that widens
the candidate roster rather than the confidence: a fourteen day history may not
compete a weekly ETS model it cannot possibly validate.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

UsageMetric: TypeAlias = Literal["request_count", "prompt_tokens", "completion_tokens", "cache_read_tokens"]
ForecastMetric: TypeAlias = Literal["spend_usd", "prompt_tokens", "completion_tokens", "total_tokens", "request_count"]
HistoryTier: TypeAlias = Literal[
    "insufficient_history", "burn_rate", "trend_weekday", "seasonal_trend", "full_ensemble"
]
Confidence: TypeAlias = Literal["none", "low", "medium", "high"]

USAGE_METRICS: Final[tuple[UsageMetric, ...]] = (
    "request_count",
    "prompt_tokens",
    "completion_tokens",
    "cache_read_tokens",
)

# Blueprint §1.6. The boundaries are the contract, not a heuristic: a scope that
# gains its third day of history must start producing a burn rate, and one that
# gains its twenty-eighth must start being allowed a seasonal model.
MIN_FORECASTABLE_DAYS: Final = 3
BURN_RATE_MAX_DAYS: Final = 7
TREND_WEEKDAY_MAX_DAYS: Final = 28
SEASONAL_MAX_DAYS: Final = 90

# §1.6: an entity idle this long is not a forecast with a very low mean, it is an
# entity nobody is asking about. Skipped rather than published as ~0.
DORMANT_AFTER_DAYS: Final = 30

_TIER_CONFIDENCE: Final[Mapping[HistoryTier, Confidence]] = MappingProxyType(
    {
        "insufficient_history": "none",
        "burn_rate": "low",
        "trend_weekday": "medium",
        "seasonal_trend": "medium",
        "full_ensemble": "high",
    }
)


@dataclass(frozen=True, slots=True)
class ScopeRef:
    """One forecastable entity: a scope type from the fact tables plus its id."""

    scope_type: str
    scope_id: str

    def __str__(self) -> str:
        return f"{self.scope_type}:{self.scope_id}"


@dataclass(frozen=True, slots=True)
class DailyUsage:
    """One calendar day of one scope, as the fact tables recorded it."""

    day: date
    request_count: float
    prompt_tokens: float
    completion_tokens: float
    cache_read_tokens: float
    spend_usd: Decimal
    mix_tokens: Mapping[str, float]

    @property
    def total_tokens(self) -> float:
        return self.prompt_tokens + self.completion_tokens


def _empty_day(day: date) -> DailyUsage:
    return DailyUsage(
        day=day,
        request_count=0.0,
        prompt_tokens=0.0,
        completion_tokens=0.0,
        cache_read_tokens=0.0,
        spend_usd=Decimal(0),
        mix_tokens=MappingProxyType({}),
    )


@dataclass(frozen=True, slots=True)
class UsageHistory:
    """A contiguous, zero-filled daily history for one scope, oldest day first."""

    scope: ScopeRef
    days: tuple[DailyUsage, ...]

    @property
    def start_day(self) -> date:
        return self.days[0].day

    @property
    def end_day(self) -> date:
        return self.days[-1].day

    @property
    def history_days(self) -> int:
        """Calendar days covered, which is what the ladder in §1.6 is expressed in."""
        return len(self.days)

    @property
    def active_days(self) -> int:
        return sum(1 for day in self.days if day.request_count > 0 or day.spend_usd > 0)

    @property
    def last_active_day(self) -> date | None:
        active: Final = tuple(day.day for day in self.days if day.request_count > 0 or day.spend_usd > 0)
        return active[-1] if active else None

    def is_dormant(self, *, dormant_after_days: int = DORMANT_AFTER_DAYS) -> bool:
        last: Final = self.last_active_day
        return last is None or (self.end_day - last).days >= dormant_after_days

    def usage_series(self, metric: UsageMetric) -> tuple[float, ...]:
        match metric:
            case "request_count":
                return tuple(day.request_count for day in self.days)
            case "prompt_tokens":
                return tuple(day.prompt_tokens for day in self.days)
            case "completion_tokens":
                return tuple(day.completion_tokens for day in self.days)
            case "cache_read_tokens":
                return tuple(day.cache_read_tokens for day in self.days)

    def spend_series(self) -> tuple[float, ...]:
        return tuple(float(day.spend_usd) for day in self.days)

    def total_spend(self) -> Decimal:
        return sum((day.spend_usd for day in self.days), Decimal(0))

    def tail(self, days: int) -> "UsageHistory":
        return UsageHistory(scope=self.scope, days=self.days[-days:] if days < len(self.days) else self.days)

    def mix_tokens_total(self, *, window_days: int) -> Mapping[str, float]:
        """Tokens per model group over the trailing window, the input to the mix forecast."""
        recent: Final = self.days[-window_days:] if window_days < len(self.days) else self.days
        groups: Final = frozenset(group for day in recent for group in day.mix_tokens)
        return MappingProxyType(
            {group: sum(day.mix_tokens.get(group, 0.0) for day in recent) for group in sorted(groups)}
        )


def build_history(scope: ScopeRef, observed: Sequence[DailyUsage], *, as_of: date) -> UsageHistory | None:
    """Zero-fill ``observed`` into a contiguous history ending at ``as_of``.

    ``as_of`` is the last day the fact tables are complete for, normally
    yesterday: including a partially aggregated today would train every model on
    a day that is guaranteed to be an undercount.
    """
    if not observed:
        return None
    by_day: Final = MappingProxyType({day.day: day for day in observed})
    first: Final = min(by_day)
    if as_of < first:
        return None
    span: Final = (as_of - first).days + 1
    return UsageHistory(
        scope=scope,
        days=tuple(_day_at(by_day, first + timedelta(days=offset)) for offset in range(span)),
    )


def _day_at(by_day: Mapping[date, DailyUsage], day: date) -> DailyUsage:
    return by_day.get(day) or _empty_day(day)


def classify_history(history_days: int) -> HistoryTier:
    """Map a history length onto its rung of the §1.6 ladder."""
    if history_days < MIN_FORECASTABLE_DAYS:
        return "insufficient_history"
    if history_days < BURN_RATE_MAX_DAYS:
        return "burn_rate"
    if history_days < TREND_WEEKDAY_MAX_DAYS:
        return "trend_weekday"
    if history_days < SEASONAL_MAX_DAYS:
        return "seasonal_trend"
    return "full_ensemble"


def tier_confidence(tier: HistoryTier) -> Confidence:
    return _TIER_CONFIDENCE[tier]
