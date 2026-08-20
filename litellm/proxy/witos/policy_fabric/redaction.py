"""Span rewriting for REDACT and MASK (§2.8).

Offsets are Python string indices, so they are code points, not bytes and not
UTF-16 units. An emoji is one index wide here and four bytes on the wire; a
redactor that mixes the two truncates a span and leaks the tail of a card
number. Every function in this module works in code points end to end and the
tests pin that with multibyte fixtures.

Overlapping findings are merged before rewriting. Two detectors firing on the
same SSN must produce one masked span, not a mask applied twice with the second
one's offsets pointing into already-rewritten text.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from functools import reduce
from typing import Final

from litellm.proxy.witos.policy_fabric.evaluator import Finding
from litellm.proxy.witos.policy_fabric.types import RedactStrategy

MASK_CHARACTER: Final = "*"


@dataclass(frozen=True, slots=True)
class Span:
    start: int
    end: int
    classifier: str

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class RedactionResult:
    text: str
    spans: tuple[Span, ...]

    @property
    def performed(self) -> bool:
        return bool(self.spans)


def merge_spans(findings: Iterable[Finding]) -> tuple[Span, ...]:
    ordered: Final = tuple(
        sorted(
            (
                Span(start=finding.start, end=finding.end, classifier=finding.classifier)
                for finding in findings
                if finding.end > finding.start
            ),
            key=lambda span: (span.start, span.end),
        )
    )

    def fold(acc: tuple[Span, ...], span: Span) -> tuple[Span, ...]:
        if not acc:
            return (span,)
        previous: Final = acc[-1]
        if span.start > previous.end:
            return (*acc, span)
        merged: Final = Span(
            start=previous.start,
            end=max(previous.end, span.end),
            classifier=previous.classifier if previous.classifier == span.classifier else "MULTIPLE",
        )
        return (*acc[:-1], merged)

    empty: Final[tuple[Span, ...]] = ()
    return reduce(fold, ordered, empty)


def apply_redaction(text: str, findings: Iterable[Finding], strategy: RedactStrategy) -> RedactionResult:
    spans: Final = tuple(span for span in merge_spans(findings) if 0 <= span.start <= span.end <= len(text))
    if not spans:
        return RedactionResult(text=text, spans=())

    def fold(acc: tuple[str, int], span: Span) -> tuple[str, int]:
        rendered, cursor = acc
        return (rendered + text[cursor : span.start] + _replacement(text, span, strategy), span.end)

    seed: Final[tuple[str, int]] = ("", 0)
    rewritten, last_cursor = reduce(fold, spans, seed)
    return RedactionResult(text=rewritten + text[last_cursor:], spans=spans)


def _replacement(text: str, span: Span, strategy: RedactStrategy) -> str:
    match strategy:
        case RedactStrategy.MASK:
            return MASK_CHARACTER * span.length
        case RedactStrategy.REPLACE:
            return f"<{span.classifier}>"
        case RedactStrategy.HASH:
            return f"<{span.classifier}:{_span_digest(text[span.start : span.end])}>"
        case RedactStrategy.REMOVE:
            return ""


def _span_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
