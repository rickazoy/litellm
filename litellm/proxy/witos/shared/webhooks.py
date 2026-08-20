"""HMAC-signed outbound webhooks (blueprint §1.11).

A WIT OS alert arrives at a customer's event bus as an HTTP POST, and the
receiver has to be able to tell it apart from anything else that can reach that
URL. Every request carries ``X-WITOS-Signature: sha256=<hex>`` over
``<timestamp>.<body>``, so a receiver that verifies it gets both authenticity and
a replay window: a captured request cannot be replayed usefully once its
timestamp is outside the receiver's tolerance, which signing the body alone would
not prevent.

Secrets are never stored in the database. A channel names a secret and the value
is read from ``WITOS_WEBHOOK_SECRET_<NAME>`` in the environment, resolved at
delivery time so rotating it takes effect without a redeploy or a migration.
"""

import hashlib
import hmac
import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Protocol, TypeAlias, TypedDict

from typing_extensions import ReadOnly

from litellm._logging import verbose_proxy_logger

SECRET_ENV_PREFIX: Final = "WITOS_WEBHOOK_SECRET_"
SIGNATURE_HEADER: Final = "X-WITOS-Signature"
TIMESTAMP_HEADER: Final = "X-WITOS-Timestamp"
EVENT_HEADER: Final = "X-WITOS-Event-Id"
DEFAULT_TIMEOUT_SECONDS: Final = 5.0


class WebhookPayload(TypedDict):
    """What a receiver gets. Declared so the wire format is a contract, not an accident."""

    alert_type: ReadOnly[str]
    severity: ReadOnly[str]
    scope_type: ReadOnly[str]
    scope_id: ReadOnly[str]
    title: ReadOnly[str]
    body: ReadOnly[Mapping[str, object]]


@dataclass(frozen=True, slots=True)
class Delivered:
    status_code: int


@dataclass(frozen=True, slots=True)
class DeliveryFailed:
    reason: str


DeliveryResult: TypeAlias = Delivered | DeliveryFailed


class HttpPoster(Protocol):
    """The one HTTP operation this module needs, injected so tests never open a socket."""

    async def post(self, url: str, *, content: bytes, headers: Mapping[str, str], timeout: float) -> int: ...


class HttpxPoster:
    """``HttpPoster`` over httpx, which the proxy already depends on."""

    async def post(self, url: str, *, content: bytes, headers: Mapping[str, str], timeout: float) -> int:
        import httpx

        sendable: Final = dict(headers)  # mutable-ok: httpx types its headers argument as a mutable mapping
        async with httpx.AsyncClient(timeout=timeout) as client:
            response: Final = await client.post(url, content=content, headers=sendable)
            return response.status_code


def resolve_secret(name: str) -> str | None:
    return os.environ.get(f"{SECRET_ENV_PREFIX}{name.upper()}")


def sign(body: bytes, secret: str, *, timestamp: int) -> str:
    """``sha256=<hex>`` over ``<timestamp>.<body>``."""
    signed: Final = f"{timestamp}.".encode() + body
    return "sha256=" + hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()


def verify(body: bytes, secret: str, *, timestamp: int, signature: str) -> bool:
    """Constant-time check, exported so a receiver implementation can share it."""
    return hmac.compare_digest(sign(body, secret, timestamp=timestamp), signature)


async def deliver(
    url: str,
    payload: Mapping[str, object],
    *,
    secret_name: str | None,
    event_id: str,
    poster: HttpPoster,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> DeliveryResult:
    """POST a signed payload. A missing secret is refused, never sent unsigned."""
    document: Final = dict(payload)  # mutable-ok: json.dumps takes a dict, and it is serialised before it can escape
    body: Final = json.dumps(document, separators=(",", ":"), default=str).encode()
    timestamp: Final = int(time.time())
    secret: Final = resolve_secret(secret_name) if secret_name else None
    if secret_name and secret is None:
        return DeliveryFailed(reason=f"{SECRET_ENV_PREFIX}{secret_name.upper()} is not set")
    unsigned: Final = (
        ("Content-Type", "application/json"),
        (TIMESTAMP_HEADER, str(timestamp)),
        (EVENT_HEADER, event_id),
    )
    signature: Final = () if secret is None else ((SIGNATURE_HEADER, sign(body, secret, timestamp=timestamp)),)
    headers: Final = MappingProxyType(dict(unsigned + signature))
    try:
        status: Final = await poster.post(url, content=body, headers=headers, timeout=timeout)
    except Exception as error:  # noqa: BLE001  # a webhook the customer's endpoint refused must not fail the job
        verbose_proxy_logger.warning("WIT OS webhook to %s failed: %s", url, error)
        return DeliveryFailed(reason=str(error))
    if status >= 400:
        return DeliveryFailed(reason=f"HTTP {status}")
    return Delivered(status_code=status)
