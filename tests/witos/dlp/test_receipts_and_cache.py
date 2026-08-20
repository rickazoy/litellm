"""Receipt masking and policy-set cache invalidation (§2.5, §2.8, §2.14).

The masking tests are the ones that matter most in this file. A decision receipt
outlives the request, gets exported, and gets read by people who are not cleared
for the content. If a raw SSN can reach the database through this path, the
product has created the breach it exists to prevent.
"""

from __future__ import annotations

import json
from typing import Mapping, Sequence

from conftest import SSN_VALUE, engine_over, policy_document, scoped

from litellm.proxy.witos.policy_fabric.evaluator import Finding, FindingSource, RequestScope
from litellm.proxy.witos.policy_fabric.policy_cache import PolicySetCache, PolicySetKey
from litellm.proxy.witos.policy_fabric.receipts import (
    DecisionReceipt,
    DecisionReceiptQueue,
    ReceiptScope,
    correlation_hash,
    flush_decisions,
    summarise_findings,
    to_row,
)
from litellm.proxy.witos.policy_fabric.runtime import EngineConfig, EvaluationRequest
from litellm.proxy.witos.policy_fabric.scope import ScopedPolicy
from litellm.proxy.witos.policy_fabric.types import (
    EnforcementKind,
    EvaluationDirection,
    PolicyAction,
)

SECRET_TEXT = f"my social is {SSN_VALUE} please keep it safe"


class RecordingTable:
    def __init__(self) -> None:
        self.rows: list[Mapping[str, object]] = []

    async def create_many(self, data: Sequence[Mapping[str, object]]) -> int:
        self.rows.extend(data)
        return len(data)


class ExplodingTable:
    async def create_many(self, data: Sequence[Mapping[str, object]]) -> int:
        del data
        raise ConnectionError("database is down")


def _receipt(findings: tuple[Finding, ...], salt: str | None = None) -> DecisionReceipt:
    return DecisionReceipt(
        request_id="req-1",
        scope=ReceiptScope(
            organization_id="org-1",
            team_id="payments",
            user_id="u1",
            key_alias="k1",
            model="gpt-5",
            application="crm",
        ),
        policy_id="p1",
        policy_version=1,
        direction=EvaluationDirection.INPUT,
        decision=PolicyAction.BLOCK,
        enforcement=EnforcementKind.PREVENTION,
        matched_classifiers=summarise_findings(findings),
        match_hash=correlation_hash(SECRET_TEXT, findings, salt),
    )


def _ssn_finding() -> Finding:
    start = SECRET_TEXT.index(SSN_VALUE)
    return Finding(
        classifier="PII.SSN",
        start=start,
        end=start + len(SSN_VALUE),
        confidence=0.99,
        source=FindingSource.LOCAL_PATTERN,
    )


def test_a_summarised_finding_carries_offsets_and_counts_never_values() -> None:
    summarised = summarise_findings((_ssn_finding(), _ssn_finding()))
    assert len(summarised) == 1
    assert summarised[0].classifier == "PII.SSN"
    assert summarised[0].count == 2
    assert summarised[0].confidence == 0.99
    assert all(isinstance(offset, tuple) for offset in summarised[0].offsets)
    assert SSN_VALUE not in json.dumps([summarised[0].classifier, list(summarised[0].offsets)])


def test_no_raw_finding_can_reach_the_database_row() -> None:
    row = to_row(_receipt((_ssn_finding(),)))
    serialised = json.dumps(dict(row), default=str)
    assert SSN_VALUE not in serialised
    assert "123" not in json.loads(row["matched_classifiers"])[0]["class"]
    start = SECRET_TEXT.index(SSN_VALUE)
    assert json.loads(row["matched_classifiers"])[0] == {
        "class": "PII.SSN",
        "count": 1,
        "offsets": [[start, start + len(SSN_VALUE)]],
        "confidence": 0.99,
    }


async def test_a_full_engine_run_never_writes_the_matched_text() -> None:
    engine = engine_over((scoped(policy_document("PCI")),))
    result = await engine.evaluate(
        EvaluationRequest(
            request_id="req-1",
            content=SECRET_TEXT,
            direction=EvaluationDirection.INPUT,
            scope=RequestScope(organization_id="org-1"),
        )
    )
    queue = DecisionReceiptQueue()
    for receipt in result.receipts:
        queue.record(receipt)
    table = RecordingTable()
    written = await flush_decisions(queue, table)
    assert written == 1
    serialised = json.dumps([dict(row) for row in table.rows], default=str)
    assert SSN_VALUE not in serialised
    assert "PII.SSN" in serialised


def test_the_correlation_hash_is_omitted_without_a_salt() -> None:
    assert correlation_hash(SECRET_TEXT, (_ssn_finding(),), None) is None
    assert correlation_hash(SECRET_TEXT, (_ssn_finding(),), "") is None
    assert correlation_hash(SECRET_TEXT, (), "salt") is None


def test_the_correlation_hash_is_stable_salted_and_irreversible_looking() -> None:
    first = correlation_hash(SECRET_TEXT, (_ssn_finding(),), "salt-a")
    same = correlation_hash(SECRET_TEXT, (_ssn_finding(),), "salt-a")
    different_salt = correlation_hash(SECRET_TEXT, (_ssn_finding(),), "salt-b")
    assert first == same
    assert first != different_salt
    assert first is not None and SSN_VALUE not in first


async def test_an_engine_configured_with_a_salt_still_writes_no_raw_value() -> None:
    engine = engine_over(
        (scoped(policy_document("PCI")),), config=EngineConfig(match_hash_salt="pepper")
    )
    result = await engine.evaluate(
        EvaluationRequest(
            request_id="req-1",
            content=SECRET_TEXT,
            direction=EvaluationDirection.INPUT,
            scope=RequestScope(),
        )
    )
    row = to_row(result.receipts[0])
    assert row["match_hash"] is not None
    assert SSN_VALUE not in json.dumps(dict(row), default=str)


def test_a_shadow_receipt_is_never_marked_prevented() -> None:
    receipt = DecisionReceipt(
        request_id="r",
        scope=ReceiptScope(None, None, None, None, None, None),
        policy_id="p",
        policy_version=1,
        direction=EvaluationDirection.INPUT,
        decision=PolicyAction.BLOCK,
        enforcement=EnforcementKind.PREVENTION,
        matched_classifiers=(),
        shadow=True,
    )
    assert receipt.prevented is False
    assert to_row(receipt)["prevented"] is False


def test_the_receipt_queue_never_blocks_and_counts_what_it_drops() -> None:
    queue = DecisionReceiptQueue(max_size=2)
    for _ in range(5):
        queue.record(_receipt((_ssn_finding(),)))
    stats = queue.stats()
    assert stats.buffered == 2
    assert stats.dropped == 3
    assert len(queue.drain()) == 2
    assert queue.stats().buffered == 0


async def test_a_database_failure_does_not_propagate_out_of_the_flush() -> None:
    queue = DecisionReceiptQueue()
    queue.record(_receipt((_ssn_finding(),)))
    assert await flush_decisions(queue, ExplodingTable()) == 0


async def test_the_policy_cache_serves_from_memory_until_invalidated() -> None:
    calls = {"count": 0}
    policies: tuple[ScopedPolicy, ...] = (scoped(policy_document("PCI")),)

    async def loader(organization_id: str | None) -> tuple[ScopedPolicy, ...]:
        del organization_id
        calls["count"] += 1
        return policies

    now = [0.0]
    cache = PolicySetCache(loader=loader, ttl_seconds=30.0, clock=lambda: now[0])

    key = cache.key_for("org-1", "team-1", "key-1")
    assert await cache.get(key) == policies
    assert await cache.get(key) == policies
    assert calls["count"] == 1, "a warm cache must not hit the database"

    cache.invalidate()
    assert await cache.get(cache.key_for("org-1", "team-1", "key-1")) == policies
    assert calls["count"] == 2, "invalidation must force a reload without a restart"


async def test_the_cache_key_carries_scope_and_generation() -> None:
    async def loader(organization_id: str | None) -> tuple[ScopedPolicy, ...]:
        del organization_id
        return ()

    cache = PolicySetCache(loader=loader)
    before = cache.key_for("org-1", "team-1", "key-1")
    assert str(before) == "dlp:policyset:org-1:team-1:key-1:0"
    cache.invalidate()
    assert str(cache.key_for("org-1", "team-1", "key-1")).endswith(":1")
    assert str(PolicySetKey(None, None, None, 7)) == "dlp:policyset:*:*:*:7"


async def test_an_expired_entry_reloads_even_without_an_invalidation() -> None:
    calls = {"count": 0}

    async def loader(organization_id: str | None) -> tuple[ScopedPolicy, ...]:
        del organization_id
        calls["count"] += 1
        return ()

    now = [0.0]
    cache = PolicySetCache(loader=loader, ttl_seconds=10.0, clock=lambda: now[0])
    key = cache.key_for(None, None, None)
    await cache.get(key)
    now[0] = 100.0
    await cache.get(key)
    assert calls["count"] == 2
