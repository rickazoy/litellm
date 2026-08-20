"""Redaction span arithmetic, including Unicode and multibyte text (§2.14).

Offsets are code points. A redactor that quietly treated them as bytes would
mask the wrong slice of a string containing accents or emoji, and would leak the
tail of the value it was asked to hide. Each case here fails loudly if that
regresses.
"""

from __future__ import annotations

from litellm.proxy.witos.policy_fabric.evaluator import Finding, FindingSource
from litellm.proxy.witos.policy_fabric.redaction import apply_redaction, merge_spans
from litellm.proxy.witos.policy_fabric.types import RedactStrategy


def finding(start: int, end: int, classifier: str = "PII.SSN", confidence: float = 1.0) -> Finding:
    return Finding(
        classifier=classifier,
        start=start,
        end=end,
        confidence=confidence,
        source=FindingSource.LOCAL_PATTERN,
    )


def test_mask_replaces_exactly_the_span_and_nothing_else() -> None:
    text = "ssn 123-45-6789 end"
    result = apply_redaction(text, (finding(4, 15),), RedactStrategy.MASK)
    assert result.text == "ssn *********** end"
    assert len(result.text) == len(text)
    assert result.performed


def test_offsets_are_code_points_not_bytes() -> None:
    text = "héllo café 123-45-6789 naïve"
    start = text.index("123-45-6789")
    result = apply_redaction(text, (finding(start, start + 11),), RedactStrategy.MASK)
    assert result.text == "héllo café *********** naïve"
    assert "123" not in result.text
    assert "6789" not in result.text
    assert len(text.encode("utf-8")) != len(text)


def test_offsets_survive_astral_plane_characters() -> None:
    text = "🙂🙂 secret 987-65-4321 tail 🙂"
    start = text.index("987-65-4321")
    result = apply_redaction(text, (finding(start, start + 11),), RedactStrategy.MASK)
    assert result.text.startswith("🙂🙂 secret ")
    assert result.text.endswith(" tail 🙂")
    assert "987" not in result.text
    assert "4321" not in result.text


def test_multiple_spans_are_rewritten_left_to_right_without_drift() -> None:
    text = "a 111-11-1111 b 222-22-2222 c"
    first = text.index("111-11-1111")
    second = text.index("222-22-2222")
    result = apply_redaction(
        text, (finding(second, second + 11), finding(first, first + 11)), RedactStrategy.MASK
    )
    assert result.text == "a *********** b *********** c"


def test_overlapping_findings_merge_into_one_span() -> None:
    spans = merge_spans((finding(4, 15), finding(10, 20, classifier="PCI.CREDIT_CARD")))
    assert len(spans) == 1
    assert (spans[0].start, spans[0].end) == (4, 20)
    assert spans[0].classifier == "MULTIPLE"


def test_adjacent_but_disjoint_findings_stay_separate() -> None:
    spans = merge_spans((finding(0, 5), finding(6, 10)))
    assert [(span.start, span.end) for span in spans] == [(0, 5), (6, 10)]


def test_replace_and_remove_and_hash_strategies() -> None:
    text = "ssn 123-45-6789 end"
    assert apply_redaction(text, (finding(4, 15),), RedactStrategy.REPLACE).text == "ssn <PII.SSN> end"
    assert apply_redaction(text, (finding(4, 15),), RedactStrategy.REMOVE).text == "ssn  end"
    hashed = apply_redaction(text, (finding(4, 15),), RedactStrategy.HASH).text
    assert hashed.startswith("ssn <PII.SSN:") and hashed.endswith("> end")
    assert "123-45-6789" not in hashed


def test_out_of_range_spans_are_ignored_rather_than_truncating_the_text() -> None:
    text = "short"
    result = apply_redaction(text, (finding(2, 400),), RedactStrategy.MASK)
    assert result.text == text
    assert not result.performed


def test_no_findings_leaves_the_text_untouched() -> None:
    assert apply_redaction("clean text", (), RedactStrategy.MASK).text == "clean text"
