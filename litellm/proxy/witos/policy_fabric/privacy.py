"""Per-connection data-sharing controls (§2.11).

Customers ask, correctly, whether the DLP integration itself exfiltrates the
data it is meant to protect. These toggles are the answer, and they are
enforced before a payload is handed to any vendor rather than documented and
hoped for.

Defaults are the conservative ones. `store_local_content` defaults to NO, so
receipts stay metadata-only unless somebody deliberately turns content storage
on.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

MAX_EXCERPT_CHARS: Final = 200


@dataclass(frozen=True, slots=True)
class PrivacySettings:
    send_full_content_to_provider: bool = False
    send_identity: bool = False
    send_application_id: bool = False
    send_conversation_context: bool = False
    store_vendor_findings: bool = False
    store_local_content: bool = False

    @staticmethod
    def from_json(raw: object) -> PrivacySettings:
        if not isinstance(raw, Mapping):
            return PrivacySettings()
        return PrivacySettings(
            send_full_content_to_provider=_flag(raw, "send_full_content_to_provider"),
            send_identity=_flag(raw, "send_identity"),
            send_application_id=_flag(raw, "send_application_id"),
            send_conversation_context=_flag(raw, "send_conversation_context"),
            store_vendor_findings=_flag(raw, "store_vendor_findings"),
            store_local_content=_flag(raw, "store_local_content"),
        )

    def to_json(self) -> Mapping[str, bool]:
        return {  # mutable-ok: dict-shaped JSON body
            "send_full_content_to_provider": self.send_full_content_to_provider,
            "send_identity": self.send_identity,
            "send_application_id": self.send_application_id,
            "send_conversation_context": self.send_conversation_context,
            "store_vendor_findings": self.store_vendor_findings,
            "store_local_content": self.store_local_content,
        }


@dataclass(frozen=True, slots=True)
class VendorPayload:
    """What is actually allowed to leave the process for a delegated call."""

    content: str
    correlation_id: str
    identity: str | None
    application: str | None


def minimise_for_vendor(
    content: str,
    correlation_id: str,
    identity: str | None,
    application: str | None,
    settings: PrivacySettings,
) -> VendorPayload:
    return VendorPayload(
        content=content if settings.send_full_content_to_provider else content[:MAX_EXCERPT_CHARS],
        correlation_id=correlation_id,
        identity=identity if settings.send_identity else None,
        application=application if settings.send_application_id else None,
    )


def _flag(raw: Mapping[str, object], key: str) -> bool:
    return raw.get(key) is True
