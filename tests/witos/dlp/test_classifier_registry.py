"""Classifier mapping: conflicts, vendor identity, untrusted input (§2.4, §2.14)."""

from __future__ import annotations

from litellm.proxy.witos.policy_fabric.classifier_registry import (
    ClassifierMapping,
    ClassifierRegistry,
    ConfidenceTranslation,
    custom_class,
    is_valid_canonical_class,
    resolves_locally,
)


def mapping(
    vendor: str = "cyera",
    vendor_classifier_id: str = "d3",
    canonical_class: str = "PII.SSN",
    vendor_name: str = "US Social Security Number",
) -> ClassifierMapping:
    return ClassifierMapping(
        vendor=vendor,
        vendor_classifier_id=vendor_classifier_id,
        vendor_name=vendor_name,
        canonical_class=canonical_class,
        confidence_translation=ConfidenceTranslation(divisor=100.0),
        metadata={"source": "test"},
    )


def test_a_row_keeps_vendor_identity_alongside_the_canonical_class() -> None:
    result = ClassifierRegistry.empty().merge((mapping(),))
    row = result.registry.get("cyera", "d3")
    assert row is not None
    assert row.canonical_class == "PII.SSN"
    assert row.vendor == "cyera"
    assert row.vendor_classifier_id == "d3"
    assert row.vendor_name == "US Social Security Number"


def test_conflicting_remap_is_refused_and_the_original_survives() -> None:
    first = ClassifierRegistry.empty().merge((mapping(),))
    second = first.registry.merge((mapping(canonical_class="PCI.CREDIT_CARD"),))
    assert len(second.conflicts) == 1
    assert second.conflicts[0].existing_canonical_class == "PII.SSN"
    assert second.conflicts[0].incoming_canonical_class == "PCI.CREDIT_CARD"
    assert second.registry.canonical_for("cyera", "d3") == "PII.SSN"


def test_conflict_inside_a_single_batch_is_detected_not_last_write_wins() -> None:
    result = ClassifierRegistry.empty().merge(
        (mapping(), mapping(canonical_class="PHI.DIAGNOSIS"))
    )
    assert len(result.conflicts) == 1
    assert result.registry.canonical_for("cyera", "d3") == "PII.SSN"


def test_an_identical_remap_is_not_a_conflict() -> None:
    first = ClassifierRegistry.empty().merge((mapping(),))
    second = first.registry.merge((mapping(vendor_name="renamed by vendor"),))
    assert second.conflicts == ()
    assert second.registry.canonical_for("cyera", "d3") == "PII.SSN"


def test_untrusted_rows_are_rejected_with_a_reason() -> None:
    rejected = ClassifierRegistry.empty().merge(
        (
            mapping(canonical_class="NOT.A.REAL.CLASS.AT.ALL"),
            mapping(vendor_classifier_id="x" * 300, canonical_class="PII.EMAIL"),
            mapping(vendor="", canonical_class="PII.EMAIL"),
        )
    )
    assert len(rejected.rejections) == 3
    assert rejected.registry.rows == {}


def test_custom_classes_are_accepted_and_slugged_from_untrusted_names() -> None:
    assert is_valid_canonical_class("CUSTOM.cyera.LEARNED_CONTRACT")
    assert not is_valid_canonical_class("CUSTOM.only_two_parts")
    assert custom_class("cyera ai", "learned/contract terms") == "CUSTOM.CYERA_AI.LEARNED_CONTRACT_TERMS"
    assert custom_class("!!!", "???") == "CUSTOM.UNKNOWN.UNKNOWN"


def test_local_resolvability_decides_presidio_versus_delegation() -> None:
    assert resolves_locally("PII.SSN")
    assert not resolves_locally("CUSTOM.cyera.LEARNED_CONTRACT")
    assert not resolves_locally("IP.TRADE_SECRET")


def test_confidence_translation_clamps_into_zero_to_one() -> None:
    translation = ConfidenceTranslation(divisor=100.0)
    assert translation.translate(90) == 0.9
    assert translation.translate(400) == 1.0
    assert translation.translate(-5) == 0.0
    assert ConfidenceTranslation(divisor=0.0).translate(50) == 0.0
