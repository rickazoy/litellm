"""Policy-set cache with cross-pod invalidation (§2.8).

A policy change has to take effect without a proxy restart, and it has to take
effect on every pod, not just the one that served the API call. That is a
generation counter in memory plus a Redis pub/sub broadcast: publishing bumps
every subscriber's generation, every cached entry keyed to the old generation
becomes unreachable, and the next request reloads from the database.

The compiled artefacts (re2 programs, parsed ASTs) are per-pod by necessity, so
Redis carries the invalidation signal rather than the objects. TTL is the floor,
not the mechanism: with pub/sub unavailable, entries still expire, so the worst
case is a stale policy for `ttl_seconds`, not forever.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol, TypeAlias

from litellm._logging import verbose_proxy_logger
from litellm.proxy.witos.policy_fabric.scope import ScopedPolicy

if TYPE_CHECKING:
    from litellm.caching.redis_cache import RedisCache

POLICY_INVALIDATION_CHANNEL: Final = "witos_dlp.policy_change"
DEFAULT_CACHE_TTL_SECONDS: Final = 30.0

PolicySetLoader: TypeAlias = Callable[
    [str | None], Awaitable[tuple[ScopedPolicy, ...]]  # mutable-ok: generation-keyed cache map, never handed out
]  # mutable-ok: generation-keyed cache map, never handed out


@dataclass(frozen=True, slots=True)
class PolicySetKey:
    """Blueprint key shape: `dlp:policyset:{org}:{team}:{key}:{version}`."""

    organization_id: str | None
    team_id: str | None
    key_alias: str | None
    generation: int

    def __str__(self) -> str:
        return (
            f"dlp:policyset:{self.organization_id or '*'}:{self.team_id or '*'}:"
            f"{self.key_alias or '*'}:{self.generation}"
        )


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    policies: tuple[ScopedPolicy, ...]
    expires_at: float


class PolicySetCache:
    def __init__(
        self,
        loader: PolicySetLoader,
        ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._loader: Final = loader
        self._ttl: Final = ttl_seconds
        self._clock: Final = clock
        # mutable-ok: generation-keyed cache, drained by invalidation and never handed out
        self._entries: Final[dict[str, _CacheEntry]] = {}  # mutable-ok: generation-keyed cache map, never handed out
        self._generation = 0  # rebind-ok: monotonic invalidation counter

    @property
    def generation(self) -> int:
        return self._generation

    def key_for(self, organization_id: str | None, team_id: str | None, key_alias: str | None) -> PolicySetKey:
        return PolicySetKey(
            organization_id=organization_id,
            team_id=team_id,
            key_alias=key_alias,
            generation=self._generation,
        )

    async def get(self, key: PolicySetKey) -> tuple[ScopedPolicy, ...]:
        cache_key: Final = str(key)
        entry: Final = self._entries.get(cache_key)
        now: Final = self._clock()
        if entry is not None and entry.expires_at > now:
            return entry.policies
        loaded: Final = await self._loader(key.organization_id)
        self._entries[cache_key] = _CacheEntry(policies=loaded, expires_at=now + self._ttl)
        return loaded

    def invalidate(self) -> None:
        """Bump the generation so every cached entry becomes unreachable."""
        self._generation += 1
        self._entries.clear()


class _PubSub(Protocol):
    def subscribe(self, *channels: str) -> Awaitable[object]: ...

    def get_message(self, ignore_subscribe_messages: bool, timeout: float) -> Awaitable[object]: ...

    def aclose(self) -> Awaitable[object]: ...


class _PubSubClient(Protocol):
    def publish(self, channel: str, message: str) -> Awaitable[int]: ...

    def pubsub(self) -> _PubSub: ...


def policy_invalidation_channel(redis_cache: RedisCache) -> str:
    namespace: Final = getattr(redis_cache, "namespace", None)
    if namespace is None:
        return POLICY_INVALIDATION_CHANNEL
    return f"{namespace}:{POLICY_INVALIDATION_CHANNEL}"


def _pubsub_capable_client(redis_cache: RedisCache) -> _PubSubClient | None:
    """Cluster clients have no usable pub/sub here, so they degrade to TTL."""
    from redis.asyncio import Redis

    client: Final[object] = redis_cache.init_async_client()  # pyright: ignore[reportUnknownMemberType]  # redis generics
    if isinstance(client, Redis):
        return client
    return None


def coordination_redis_cache() -> RedisCache | None:
    from litellm.proxy.proxy_server import redis_usage_cache

    return redis_usage_cache


async def publish_policy_invalidation(redis_cache: RedisCache | None) -> bool:
    if redis_cache is None:
        return False
    try:
        client: Final = _pubsub_capable_client(redis_cache)
        if client is None:
            verbose_proxy_logger.debug(
                "WIT OS DLP: policy invalidation not published; cluster redis has no pub/sub support"
            )
            return False
        await client.publish(policy_invalidation_channel(redis_cache), POLICY_INVALIDATION_CHANNEL)
    except Exception as err:  # noqa: BLE001  # best effort; a policy write must not fail on a redis hiccup
        verbose_proxy_logger.warning("WIT OS DLP: policy invalidation publish failed: %s", err)
        return False
    return True


async def apply_invalidation_message(caches: Sequence[PolicySetCache]) -> None:
    for cache in caches:
        cache.invalidate()
