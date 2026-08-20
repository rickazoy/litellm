"""Canonical classifier taxonomy and vendor mapping registry (§2.4).

Two rules govern this module.

Vendor identity is never discarded. A mapping row carries the vendor's own id
and display name alongside the canonical class, so a decision receipt can always
answer "which vendor detector fired" and not merely "something PII-ish fired".

Mapping is a parsing security boundary. Vendor classifier ids and names are
untrusted strings: they are length-checked, never used to build a class name by
concatenation without normalisation, and two rows that disagree about the same
vendor id are a `MappingConflict`, not a last-write-wins overwrite.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import reduce
from types import MappingProxyType
from typing import Final

CANONICAL_CLASSES: Final[frozenset[str]] = frozenset(
    {
        "PII.EMAIL",
        "PII.SSN",
        "PII.PASSPORT",
        "PII.DOB",
        "PII.PHONE",
        "PII.NAME",
        "PII.ADDRESS",
        "PII.DRIVERS_LICENSE",
        "PII.NATIONAL_ID",
        "PII.IP_ADDRESS",
        "PHI.MEDICAL_RECORD",
        "PHI.DIAGNOSIS",
        "PHI.TREATMENT",
        "PHI.INSURANCE_ID",
        "PCI.CREDIT_CARD",
        "PCI.CVV",
        "FINANCIAL.BANK_ACCOUNT",
        "FINANCIAL.ROUTING_NUMBER",
        "FINANCIAL.IBAN",
        "FINANCIAL.SWIFT",
        "SECRET.API_KEY",
        "SECRET.PASSWORD",
        "SECRET.PRIVATE_KEY",
        "SECRET.ACCESS_TOKEN",
        "IP.SOURCE_CODE",
        "IP.TRADE_SECRET",
    }
)

CUSTOM_CLASS_PREFIX: Final = "CUSTOM."
_MAX_VENDOR_ID_LEN: Final = 256
_MAX_VENDOR_NAME_LEN: Final = 256

# Presidio entity type -> canonical class. This is the whole of "data_class
# leaves resolvable locally route through Presidio": if a canonical class is a
# value here, the compiler plans it as a local Presidio lookup instead of a
# delegated vendor call.
PRESIDIO_ENTITY_TO_CANONICAL: Final[Mapping[str, str]] = MappingProxyType(
    {
        "EMAIL_ADDRESS": "PII.EMAIL",
        "US_SSN": "PII.SSN",
        "US_PASSPORT": "PII.PASSPORT",
        "DATE_TIME": "PII.DOB",
        "PHONE_NUMBER": "PII.PHONE",
        "PERSON": "PII.NAME",
        "LOCATION": "PII.ADDRESS",
        "US_DRIVER_LICENSE": "PII.DRIVERS_LICENSE",
        "IP_ADDRESS": "PII.IP_ADDRESS",
        "MEDICAL_LICENSE": "PHI.MEDICAL_RECORD",
        "CREDIT_CARD": "PCI.CREDIT_CARD",
        "US_BANK_NUMBER": "FINANCIAL.BANK_ACCOUNT",
        "IBAN_CODE": "FINANCIAL.IBAN",
        "CRYPTO": "SECRET.PRIVATE_KEY",
    }
)

CANONICAL_TO_PRESIDIO_ENTITY: Final[Mapping[str, str]] = MappingProxyType(
    {canonical: entity for entity, canonical in PRESIDIO_ENTITY_TO_CANONICAL.items()}
)


def is_valid_canonical_class(value: str) -> bool:
    return value in CANONICAL_CLASSES or _is_wellformed_custom_class(value)


def _is_wellformed_custom_class(value: str) -> bool:
    if not value.startswith(CUSTOM_CLASS_PREFIX):
        return False
    parts: Final = value.split(".")
    return len(parts) == 3 and all(part for part in parts)


def custom_class(vendor: str, name: str) -> str:
    """Build a `CUSTOM.<vendor>.<name>` class from untrusted vendor strings."""
    return f"{CUSTOM_CLASS_PREFIX}{_slug(vendor)}.{_slug(name)}"


def _slug(raw: str) -> str:
    cleaned: Final = "".join(char if char.isalnum() else "_" for char in raw.strip())
    collapsed: Final = "_".join(part for part in cleaned.split("_") if part)
    return collapsed.upper() or "UNKNOWN"


def resolves_locally(canonical_class: str) -> bool:
    """True when Presidio can answer this class without calling a vendor."""
    return canonical_class in CANONICAL_TO_PRESIDIO_ENTITY


@dataclass(frozen=True, slots=True)
class ConfidenceTranslation:
    """How a vendor's confidence becomes a 0..1 canonical confidence.

    `divisor` covers the common 0-100 and 0-1000 scales. Kept explicit so a
    receipt's `confidence` is never a number nobody can explain.
    """

    divisor: float = 1.0

    def translate(self, vendor_confidence: float) -> float:
        if self.divisor <= 0:
            return 0.0
        return min(1.0, max(0.0, vendor_confidence / self.divisor))


@dataclass(frozen=True, slots=True)
class ClassifierMapping:
    vendor: str
    vendor_classifier_id: str
    vendor_name: str
    canonical_class: str
    confidence_translation: ConfidenceTranslation
    metadata: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class MappingConflict:
    vendor: str
    vendor_classifier_id: str
    existing_canonical_class: str
    incoming_canonical_class: str

    @property
    def reason(self) -> str:
        return (
            f"{self.vendor}:{self.vendor_classifier_id} already maps to "
            f"{self.existing_canonical_class}; refusing to remap to {self.incoming_canonical_class}"
        )


@dataclass(frozen=True, slots=True)
class MappingRejection:
    vendor: str
    vendor_classifier_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class ClassifierRegistry:
    """Immutable registry. `merge` returns a new registry plus what it refused."""

    rows: Mapping[str, ClassifierMapping]

    @staticmethod
    def empty() -> ClassifierRegistry:
        return ClassifierRegistry(rows=MappingProxyType({}))

    def get(self, vendor: str, vendor_classifier_id: str) -> ClassifierMapping | None:
        return self.rows.get(_row_key(vendor, vendor_classifier_id))

    def canonical_for(self, vendor: str, vendor_classifier_id: str) -> str | None:
        row: Final = self.get(vendor, vendor_classifier_id)
        return None if row is None else row.canonical_class

    def for_vendor(self, vendor: str) -> tuple[ClassifierMapping, ...]:
        return tuple(row for row in self.rows.values() if row.vendor == vendor)

    def merge(self, incoming: Iterable[ClassifierMapping]) -> MergeResult:
        """Fold rows in, one at a time.

        Conflicts are detected against rows already accepted in this same batch,
        not only against the pre-existing registry: two vendor exports that
        disagree inside one sync are exactly the case that must not resolve to
        whichever row happened to be parsed last.
        """
        return reduce(_merge_one, incoming, MergeResult(registry=self, conflicts=(), rejections=()))


@dataclass(frozen=True, slots=True)
class MergeResult:
    registry: ClassifierRegistry
    conflicts: tuple[MappingConflict, ...]
    rejections: tuple[MappingRejection, ...]


def _merge_one(acc: MergeResult, row: ClassifierMapping) -> MergeResult:
    rejection_reason: Final = _validate(row)
    if rejection_reason is not None:
        rejected: Final = MappingRejection(
            vendor=row.vendor,
            vendor_classifier_id=row.vendor_classifier_id,
            reason=rejection_reason,
        )
        return MergeResult(
            registry=acc.registry,
            conflicts=acc.conflicts,
            rejections=(*acc.rejections, rejected),
        )
    conflict: Final = _conflict_for(acc.registry.rows, row)
    if conflict is not None:
        return MergeResult(
            registry=acc.registry,
            conflicts=(*acc.conflicts, conflict),
            rejections=acc.rejections,
        )
    merged_rows: Final = MappingProxyType({**acc.registry.rows, _row_key(row.vendor, row.vendor_classifier_id): row})
    return MergeResult(
        registry=ClassifierRegistry(rows=merged_rows),
        conflicts=acc.conflicts,
        rejections=acc.rejections,
    )


def _row_key(vendor: str, vendor_classifier_id: str) -> str:
    return f"{vendor}\x00{vendor_classifier_id}"


def _validate(row: ClassifierMapping) -> str | None:
    if not row.vendor or not row.vendor_classifier_id:
        return "vendor and vendor_classifier_id are required"
    if len(row.vendor_classifier_id) > _MAX_VENDOR_ID_LEN:
        return f"vendor_classifier_id exceeds {_MAX_VENDOR_ID_LEN} characters"
    if len(row.vendor_name) > _MAX_VENDOR_NAME_LEN:
        return f"vendor_name exceeds {_MAX_VENDOR_NAME_LEN} characters"
    if not is_valid_canonical_class(row.canonical_class):
        return f"{row.canonical_class!r} is not a canonical class or a well-formed CUSTOM.<vendor>.<name>"
    return None


def _conflict_for(
    existing: Mapping[str, ClassifierMapping],
    row: ClassifierMapping,
) -> MappingConflict | None:
    current: Final = existing.get(_row_key(row.vendor, row.vendor_classifier_id))
    if current is None or current.canonical_class == row.canonical_class:
        return None
    return MappingConflict(
        vendor=row.vendor,
        vendor_classifier_id=row.vendor_classifier_id,
        existing_canonical_class=current.canonical_class,
        incoming_canonical_class=row.canonical_class,
    )
