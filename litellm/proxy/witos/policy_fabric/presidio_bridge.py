"""Bridge to LiteLLM's existing Presidio integration (§2.8).

PII detection is not reimplemented here. `data_class` leaves that Presidio can
answer are routed to the analyzer LiteLLM already ships, and the response is
translated into canonical classes and `Finding` offsets. Writing a second
recogniser set would mean two detectors to keep in sync and two sets of false
negatives to explain.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from litellm.proxy.witos.policy_fabric.classifier_registry import PRESIDIO_ENTITY_TO_CANONICAL
from litellm.proxy.witos.policy_fabric.evaluator import Finding, FindingSource

if TYPE_CHECKING:
    from litellm.proxy.guardrails.guardrail_hooks.presidio import _OPTIONAL_PresidioPIIMasking


@dataclass(frozen=True, slots=True)
class LiteLLMPresidioAnalyzer:
    masking: _OPTIONAL_PresidioPIIMasking

    async def analyze(self, text: str, entities: tuple[str, ...]) -> tuple[Finding, ...]:
        raw: Final = await self.masking.analyze_text(
            text=text,
            presidio_config=None,
            request_data={},  # mutable-ok: presidio response shape
        )  # mutable-ok: presidio response shape
        if not isinstance(raw, list):
            return ()
        return _to_findings(raw, entities)


def _to_findings(raw: Sequence[object], entities: tuple[str, ...]) -> tuple[Finding, ...]:
    wanted: Final = frozenset(entities)
    return tuple(finding for finding in (_to_finding(item, wanted) for item in raw) if finding is not None)


def _to_finding(item: object, wanted: frozenset[str]) -> Finding | None:
    if not isinstance(item, dict):
        return None
    entity_type: Final = item.get("entity_type")
    start: Final = item.get("start")
    end: Final = item.get("end")
    score: Final = item.get("score")
    if not isinstance(entity_type, str) or entity_type not in wanted:
        return None
    if not isinstance(start, int) or not isinstance(end, int) or end <= start:
        return None
    canonical: Final = PRESIDIO_ENTITY_TO_CANONICAL.get(entity_type)
    if canonical is None:
        return None
    return Finding(
        classifier=canonical,
        start=start,
        end=end,
        confidence=float(score) if isinstance(score, (int, float)) else 1.0,
        source=FindingSource.PRESIDIO,
    )


def build_presidio_analyzer() -> LiteLLMPresidioAnalyzer | None:
    """Reuse a configured Presidio guardrail if the proxy already has one."""
    import litellm
    from litellm.proxy.guardrails.guardrail_hooks.presidio import _OPTIONAL_PresidioPIIMasking

    for callback in litellm.callbacks:
        if isinstance(callback, _OPTIONAL_PresidioPIIMasking):
            return LiteLLMPresidioAnalyzer(masking=callback)
    return None
