"""Decision receipts (§2.5, §2.8).

The single rule this module exists to enforce: a receipt records that something
was found, never what was found. A receipt containing an actual SSN would make
the product create the breach it is there to prevent, and receipts outlive
requests, get exported, and get read by people who are not cleared for the
content.

Masking is structural and happens at write time. `MatchedClassifier` has no
field capable of holding matched text, `build_receipt` is the only constructor,
and it takes `Finding` objects that carry offsets rather than substrings. The
optional correlation hash is emitted only when a salt is configured, because an
unsalted SHA-256 of a nine-digit number is a lookup table, not a one-way
function.

Writes never block the request. The guardrail calls `record`, which is
synchronous and cannot await; a scheduled flush drains the buffer.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Protocol

from typing_extensions import ReadOnly, TypedDict

from litellm._logging import verbose_proxy_logger
from litellm.proxy.witos.policy_fabric.evaluator import Finding
from litellm.proxy.witos.policy_fabric.types import (
    EnforcementKind,
    EvaluationDirection,
    PolicyAction,
    StreamingMode,
)

MAX_BUFFERED_RECEIPTS: Final = 10_000
MAX_OFFSETS_PER_CLASSIFIER: Final = 64
MATCH_HASH_SALT_ENV: Final = "WITOS_DLP_MATCH_HASH_SALT"


@dataclass(frozen=True, slots=True)
class MatchedClassifier:
    classifier: str
    count: int
    offsets: tuple[tuple[int, int], ...]
    confidence: float


@dataclass(frozen=True, slots=True)
class ReceiptScope:
    organization_id: str | None
    team_id: str | None
    user_id: str | None
    key_alias: str | None
    model: str | None
    application: str | None


@dataclass(frozen=True, slots=True)
class DecisionReceipt:
    request_id: str
    scope: ReceiptScope
    policy_id: str
    policy_version: int
    direction: EvaluationDirection
    decision: PolicyAction
    enforcement: EnforcementKind
    matched_classifiers: tuple[MatchedClassifier, ...]
    provider: str | None = None
    external_policy_id: str | None = None
    matched_rule_ids: tuple[str, ...] = ()
    risk_score: float | None = None
    vendor_request_id: str | None = None
    evaluation_latency_ms: int | None = None
    fail_mode_triggered: bool = False
    redaction_performed: bool = False
    shadow: bool = False
    streaming_mode: StreamingMode | None = None
    match_hash: str | None = None

    @property
    def prevented(self) -> bool:
        """Whether the decision actually stopped disclosure.

        A shadow decision never prevents anything, and neither does a block
        recorded after the client already received the bytes.
        """
        return not self.shadow and self.enforcement is EnforcementKind.PREVENTION


class DecisionRow(TypedDict):
    request_id: ReadOnly[str]
    scope_json: ReadOnly[str]
    policy_id: ReadOnly[str]
    policy_version: ReadOnly[int]
    provider: ReadOnly[str | None]
    external_policy_id: ReadOnly[str | None]
    direction: ReadOnly[str]
    decision: ReadOnly[str]
    risk_score: ReadOnly[float | None]
    matched_classifiers: ReadOnly[str]
    matched_rule_ids: ReadOnly[str]
    vendor_request_id: ReadOnly[str | None]
    evaluation_latency_ms: ReadOnly[int | None]
    fail_mode_triggered: ReadOnly[bool]
    redaction_performed: ReadOnly[bool]
    shadow: ReadOnly[bool]
    streaming_mode: ReadOnly[str | None]
    prevented: ReadOnly[bool]
    organization_id: ReadOnly[str | None]
    match_hash: ReadOnly[str | None]


def summarise_findings(findings: Sequence[Finding]) -> tuple[MatchedClassifier, ...]:
    classifiers: Final = tuple(dict.fromkeys(finding.classifier for finding in findings))
    return tuple(
        MatchedClassifier(
            classifier=classifier,
            count=sum(1 for finding in findings if finding.classifier == classifier),
            offsets=tuple((finding.start, finding.end) for finding in findings if finding.classifier == classifier)[
                :MAX_OFFSETS_PER_CLASSIFIER
            ],
            confidence=max(
                (finding.confidence for finding in findings if finding.classifier == classifier),
                default=0.0,
            ),
        )
        for classifier in classifiers
    )


def correlation_hash(content: str, findings: Sequence[Finding], salt: str | None) -> str | None:
    """Salted digest of the matched substrings, for correlating repeat offenders.

    Returns None without a salt. An unsalted digest of a short, structured value
    such as an SSN or a card number is reversible by brute force in seconds, so
    emitting one would be storing the finding with extra steps.
    """
    if not salt or not findings:
        return None
    material: Final = "\x00".join(sorted(content[finding.start : finding.end] for finding in findings))
    return hashlib.sha256(f"{salt}\x00{material}".encode()).hexdigest()


def to_row(receipt: DecisionReceipt) -> DecisionRow:
    row: Final[DecisionRow] = {
        "request_id": receipt.request_id,
        "scope_json": json.dumps(
            {  # mutable-ok: json.dumps input, serialised immediately
                "organization_id": receipt.scope.organization_id,
                "team_id": receipt.scope.team_id,
                "user_id": receipt.scope.user_id,
                "key_alias": receipt.scope.key_alias,
                "model": receipt.scope.model,
                "application": receipt.scope.application,
            }
        ),
        "policy_id": receipt.policy_id,
        "policy_version": receipt.policy_version,
        "provider": receipt.provider,
        "external_policy_id": receipt.external_policy_id,
        "direction": receipt.direction.value,
        "decision": receipt.decision.value,
        "risk_score": receipt.risk_score,
        "matched_classifiers": json.dumps(
            tuple(
                {
                    "class": matched.classifier,  # mutable-ok: json.dumps input, serialised immediately
                    "count": matched.count,
                    "offsets": matched.offsets,
                    "confidence": matched.confidence,
                }
                for matched in receipt.matched_classifiers
            )
        ),
        "matched_rule_ids": json.dumps(receipt.matched_rule_ids),
        "vendor_request_id": receipt.vendor_request_id,
        "evaluation_latency_ms": receipt.evaluation_latency_ms,
        "fail_mode_triggered": receipt.fail_mode_triggered,
        "redaction_performed": receipt.redaction_performed,
        "shadow": receipt.shadow,
        "streaming_mode": None if receipt.streaming_mode is None else receipt.streaming_mode.value,
        "prevented": receipt.prevented,
        "organization_id": receipt.scope.organization_id,
        "match_hash": receipt.match_hash,
    }
    return row


class DecisionTableActions(Protocol):
    async def create_many(self, data: Sequence[Mapping[str, object]]) -> int: ...


@dataclass(frozen=True, slots=True)
class QueueStats:
    buffered: int
    dropped: int


class DecisionReceiptQueue:
    """Bounded in-memory buffer drained by a scheduled flush.

    `record` is synchronous and never awaits, so no request ever waits on a
    receipt insert. When the buffer is full the oldest receipt is dropped and
    counted: losing an audit row is bad, stalling inference to write one is
    worse, and a silent loss would be worst of all.
    """

    def __init__(self, max_size: int = MAX_BUFFERED_RECEIPTS) -> None:
        self._buffer: Final[deque[DecisionReceipt]] = deque(  # mutable-ok: a ring buffer is the point here
            maxlen=max_size
        )
        self._dropped = 0  # rebind-ok: overflow counter, only ever incremented

    def record(self, receipt: DecisionReceipt) -> None:
        if len(self._buffer) == self._buffer.maxlen:
            self._dropped += 1
        self._buffer.append(receipt)

    def drain(self) -> tuple[DecisionReceipt, ...]:
        drained: Final = tuple(self._buffer)
        self._buffer.clear()
        return drained

    def stats(self) -> QueueStats:
        return QueueStats(buffered=len(self._buffer), dropped=self._dropped)


async def flush_decisions(queue: DecisionReceiptQueue, table: DecisionTableActions) -> int:
    receipts: Final = queue.drain()
    if not receipts:
        return 0
    rows: Final = tuple(to_row(receipt) for receipt in receipts)
    try:
        await table.create_many(data=rows)
    except Exception as err:  # noqa: BLE001  # a receipt write must never take the proxy down
        verbose_proxy_logger.warning("WIT OS DLP: failed to flush %d decision receipts: %s", len(rows), err)
        return 0
    return len(rows)
