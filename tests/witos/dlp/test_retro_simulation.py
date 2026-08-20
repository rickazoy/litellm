"""Retroactive policy simulation.

Fixture receipt rows only: no database, no network, no proxy. The properties
that matter are all decidable from a handful of rows, and the one that matters
most is negative. A leaf that cannot be replayed must never come back as "did
not match", because a policy that reads as harmless only because its regex
clause was skipped is the worst thing this feature could produce.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from datetime import datetime, timedelta, timezone
from typing import Any, Final, Mapping, Sequence

import pytest
from fastapi import HTTPException

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.witos.policy_fabric.aggregator import PolicyVerdict, aggregate
from litellm.proxy.witos.policy_fabric.canonical import PolicyValidationFailure, parse_policy
from litellm.proxy.witos.policy_fabric.evaluator import Truth
from litellm.proxy.witos.policy_fabric.retro import (
    DEFAULT_ROW_CAP,
    Delta,
    IndeterminateReason,
    RecordedDecision,
    RecordedScope,
    RetroResult,
    ScopeFilter,
    compare,
    group_units,
    load_window,
    parse_decision_row,
    simulate,
)
from litellm.proxy.witos.policy_fabric.retro_endpoints import (
    MAX_WINDOW_DAYS,
    RetroWindow,
    RunRequest,
    _candidate,
    _require_simulation_scopes,
    _run_row,
    _validated_window,
    result_view,
)
from litellm.proxy.witos.policy_fabric.types import (
    FederationMode,
    PolicyAction,
    RedactStrategy,
    Severity,
)

WINDOW_START: Final = datetime(2026, 5, 1, tzinfo=timezone.utc)
WINDOW_END: Final = datetime(2026, 8, 1, tzinfo=timezone.utc)

FULL_SCOPE: Final[Mapping[str, Any]] = {
    "organization_id": "org-1",
    "team_id": "payments",
    "user_id": "u-7",
    "key_alias": "svc-checkout",
    "model": "gpt-5.1",
    "application": "checkout-bot",
}


@dataclass(frozen=True)
class Row:
    """A `WITOS_DLPDecision` row as prisma hands it back."""

    decision_id: str
    request_id: str
    created_at: datetime
    direction: str
    decision: str
    policy_id: str
    shadow: bool
    scope_json: str
    matched_classifiers: str


class FakeDecisions:
    def __init__(self, rows: Sequence[object]) -> None:
        self._rows = tuple(rows)

    async def find_many(
        self,
        where: Mapping[str, object] | None = None,
        order: Mapping[str, str] | None = None,
        skip: int = 0,
        take: int = 100,
    ) -> Sequence[object]:
        return self._rows[skip : skip + take]


def row(
    decision_id: str = "d1",
    request_id: str = "r1",
    classes: Sequence[Mapping[str, Any]] = ({"class": "PII.SSN", "count": 1, "confidence": 0.95},),
    decision: str = "AUDIT",
    direction: str = "input",
    shadow: bool = False,
    scope: Mapping[str, Any] | None = None,
    created_at: datetime | None = None,
    policy_id: str = "p-existing",
) -> Row:
    return Row(
        decision_id=decision_id,
        request_id=request_id,
        created_at=created_at or datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc),
        direction=direction,
        decision=decision,
        policy_id=policy_id,
        shadow=shadow,
        scope_json=json.dumps(dict(FULL_SCOPE) if scope is None else dict(scope)),
        matched_classifiers=json.dumps(list(classes)),
    )


def candidate(
    condition: Mapping[str, Any],
    on_match: str = "BLOCK",
    direction: str = "both",
    scope: Mapping[str, Any] | None = None,
    exceptions: Sequence[Mapping[str, Any]] | None = None,
) -> Any:
    document: dict[str, Any] = {
        "wit_dps_version": "2.0",
        "name": "candidate",
        "mode": "mirror",
        "severity": "high",
        "direction": direction,
        "condition": dict(condition),
        "actions": {"on_match": on_match},
    }
    if scope is not None:
        document["scope"] = dict(scope)
    if exceptions is not None:
        document["exceptions"] = [dict(entry) for entry in exceptions]
    parsed = parse_policy(document)
    assert not isinstance(parsed, PolicyValidationFailure), parsed
    return parsed


async def run(
    condition: Mapping[str, Any],
    rows: Sequence[Row],
    on_match: str = "BLOCK",
    scope_filter: ScopeFilter = ScopeFilter(),
    row_cap: int = DEFAULT_ROW_CAP,
    direction: str = "both",
    scope: Mapping[str, Any] | None = None,
    exceptions: Sequence[Mapping[str, Any]] | None = None,
) -> RetroResult:
    window = await load_window(FakeDecisions(rows), where={}, row_cap=row_cap, page_size=2)
    return await simulate(
        candidate=candidate(condition, on_match, direction, scope, exceptions),
        window=window,
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        scope_filter=scope_filter,
        row_cap=row_cap,
    )


DATA_CLASS_SSN: Final[Mapping[str, Any]] = {"type": "data_class", "class": "PII.SSN"}
REGEX_LEAF: Final[Mapping[str, Any]] = {"type": "regex", "pattern": r"\d{3}-\d{2}-\d{4}"}


async def test_data_class_policy_matches_recorded_findings() -> None:
    result = await run(DATA_CLASS_SSN, (row(),))

    assert result.matched_requests == 1
    assert result.indeterminate_requests == 0
    assert result.action_distribution == {"BLOCK": 1}
    assert result.sample_decision_ids == ("d1",)


async def test_data_class_absent_from_findings_does_not_match() -> None:
    result = await run(DATA_CLASS_SSN, (row(classes=({"class": "PII.EMAIL", "count": 2, "confidence": 0.9},)),))

    assert result.matched_requests == 0
    assert result.unmatched_requests == 1
    assert result.indeterminate_requests == 0


async def test_min_count_and_min_confidence_are_replayed() -> None:
    findings = ({"class": "PII.SSN", "count": 1, "confidence": 0.4},)
    below_count = await run({**DATA_CLASS_SSN, "min_count": 2}, (row(classes=findings),))
    below_confidence = await run({**DATA_CLASS_SSN, "min_confidence": 0.9}, (row(classes=findings),))

    assert below_count.matched_requests == 0
    assert below_confidence.matched_requests == 0


async def test_regex_leaf_is_indeterminate_and_never_a_non_match() -> None:
    result = await run(REGEX_LEAF, (row(),))

    assert result.indeterminate_requests == 1
    assert result.unmatched_requests == 0
    assert result.matched_requests == 0
    assert result.indeterminate_reasons == {IndeterminateReason.CONTENT_NOT_RETAINED.value: 1}
    assert result.indeterminate_leaf_types == {"regex": 1}


@pytest.mark.parametrize("leaf_type", ["dictionary", "keyword"])
async def test_term_leaves_are_indeterminate(leaf_type: str) -> None:
    result = await run({"type": leaf_type, "terms": ["acquisition"]}, (row(),))

    assert result.indeterminate_requests == 1
    assert result.unmatched_requests == 0
    assert result.indeterminate_leaf_types == {leaf_type: 1}


async def test_all_with_indeterminate_and_no_false_is_indeterminate() -> None:
    result = await run({"operator": "ALL", "conditions": [DATA_CLASS_SSN, REGEX_LEAF]}, (row(),))

    assert result.indeterminate_requests == 1
    assert result.matched_requests == 0
    assert result.unmatched_requests == 0


async def test_all_with_a_determinate_false_stays_false() -> None:
    absent = {"type": "data_class", "class": "PII.PASSPORT"}
    result = await run({"operator": "ALL", "conditions": [absent, REGEX_LEAF]}, (row(),))

    assert result.unmatched_requests == 1
    assert result.indeterminate_requests == 0


async def test_any_with_a_determinate_true_stays_true() -> None:
    result = await run({"operator": "ANY", "conditions": [DATA_CLASS_SSN, REGEX_LEAF]}, (row(),))

    assert result.matched_requests == 1
    assert result.indeterminate_requests == 0


async def test_any_with_only_false_and_indeterminate_is_indeterminate() -> None:
    absent = {"type": "data_class", "class": "PII.PASSPORT"}
    result = await run({"operator": "ANY", "conditions": [absent, REGEX_LEAF]}, (row(),))

    assert result.indeterminate_requests == 1
    assert result.unmatched_requests == 0


async def test_not_of_indeterminate_stays_indeterminate() -> None:
    result = await run({"operator": "NOT", "condition": REGEX_LEAF}, (row(),))

    assert result.indeterminate_requests == 1
    assert result.unmatched_requests == 0
    assert result.matched_requests == 0


async def test_not_of_a_determinate_match_is_false() -> None:
    result = await run({"operator": "NOT", "condition": DATA_CLASS_SSN}, (row(),))

    assert result.unmatched_requests == 1
    assert result.indeterminate_requests == 0


async def test_sensitivity_and_provider_leaves_are_indeterminate_not_false() -> None:
    sensitivity = await run({"type": "sensitivity", "value": "restricted"}, (row(),))
    provider = await run({"type": "provider", "value": "openai"}, (row(),))

    assert sensitivity.indeterminate_requests == 1
    assert sensitivity.indeterminate_reasons == {
        IndeterminateReason.CLASSIFICATION_DETAIL_NOT_RECORDED.value: 1
    }
    assert provider.indeterminate_requests == 1
    assert provider.indeterminate_reasons == {IndeterminateReason.SCOPE_NOT_RECORDED.value: 1}


async def test_recorded_scope_leaves_replay_as_determinate() -> None:
    hit = await run({"type": "team", "value": "payments"}, (row(),))
    miss = await run({"type": "team", "value": "research"}, (row(),))

    assert hit.matched_requests == 1
    assert miss.unmatched_requests == 1
    assert miss.indeterminate_requests == 0


async def test_absent_scope_key_is_indeterminate_rather_than_no_match() -> None:
    without_team = {key: value for key, value in FULL_SCOPE.items() if key != "team_id"}
    result = await run({"type": "team", "value": "payments"}, (row(scope=without_team),))

    assert result.indeterminate_requests == 1
    assert result.unmatched_requests == 0


async def test_recorded_null_scope_value_is_a_determinate_no_match() -> None:
    """A recorded null means the request had no team. An absent key means nobody wrote one down."""
    null_team = {**FULL_SCOPE, "team_id": None}
    result = await run({"type": "team", "value": "payments"}, (row(scope=null_team),))

    assert result.unmatched_requests == 1
    assert result.indeterminate_requests == 0


async def test_direction_gate_excludes_the_other_direction() -> None:
    result = await run(DATA_CLASS_SSN, (row(direction="output"),), direction="input")

    assert result.unmatched_requests == 1
    assert result.matched_requests == 0


async def test_policy_scope_narrows_by_recorded_team() -> None:
    scoped_out = await run(
        DATA_CLASS_SSN,
        (row(),),
        scope={"entities": [{"type": "team", "id": "research"}]},
    )
    scoped_in = await run(
        DATA_CLASS_SSN,
        (row(),),
        scope={"entities": [{"type": "team", "id": "payments"}]},
    )

    assert scoped_out.unmatched_requests == 1
    assert scoped_out.indeterminate_requests == 0
    assert scoped_in.matched_requests == 1


async def test_exception_on_an_unrecorded_field_is_indeterminate() -> None:
    """A group exception could have excluded this request, and nothing recorded the groups."""
    result = await run(DATA_CLASS_SSN, (row(),), exceptions=({"type": "group", "value": "secops"},))

    assert result.indeterminate_requests == 1
    assert result.matched_requests == 0
    assert result.indeterminate_reasons == {IndeterminateReason.SCOPE_NOT_RECORDED.value: 1}


async def test_exception_on_a_recorded_field_excludes_determinately() -> None:
    result = await run(DATA_CLASS_SSN, (row(),), exceptions=({"type": "team", "value": "payments"},))

    assert result.unmatched_requests == 1
    assert result.indeterminate_requests == 0


async def test_findings_from_every_policy_on_a_request_are_replayed_together() -> None:
    """Two policies each recorded half of what a candidate needs. The candidate sees both."""
    rows = (
        row(decision_id="d1", classes=({"class": "PII.SSN", "count": 1, "confidence": 0.9},)),
        row(decision_id="d2", classes=({"class": "PCI.CREDIT_CARD", "count": 1, "confidence": 0.9},)),
    )
    both = {
        "operator": "ALL",
        "conditions": [DATA_CLASS_SSN, {"type": "data_class", "class": "PCI.CREDIT_CARD"}],
    }
    result = await run(both, rows)

    assert result.evaluated_requests == 1
    assert result.matched_requests == 1
    assert sorted(result.sample_decision_ids) == ["d1", "d2"]


async def test_recorded_action_follows_the_runtime_aggregator() -> None:
    actions = ("AUDIT", "REDACT", "BLOCK", "WARN")
    rows = tuple(
        row(decision_id=f"d{index}", decision=action, policy_id=f"p{index}")
        for index, action in enumerate(actions)
    )
    unit = group_units(tuple(_parsed(rows)))[0]
    runtime = aggregate(
        tuple(
            PolicyVerdict(
                policy_id=f"p{index}",
                policy_name=f"p{index}",
                policy_version=1,
                action=PolicyAction(action),
                severity=Severity.HIGH,
                priority=100,
                mode=FederationMode.MIRROR,
                shadow=False,
                redact_strategy=RedactStrategy.MASK,
            )
            for index, action in enumerate(actions)
        )
    )

    assert unit.recorded_action is runtime.action is PolicyAction.BLOCK


async def test_shadow_rows_contribute_findings_but_not_the_recorded_action() -> None:
    rows = (
        row(decision_id="d1", decision="AUDIT", shadow=False, classes=()),
        row(
            decision_id="d2",
            decision="BLOCK",
            shadow=True,
            classes=({"class": "PII.SSN", "count": 1, "confidence": 0.9},),
        ),
    )
    unit = group_units(tuple(_parsed(rows)))[0]
    result = await run(DATA_CLASS_SSN, rows)

    assert unit.recorded_action is PolicyAction.AUDIT
    assert result.matched_requests == 1
    assert result.newly_blocked == 1


async def test_newly_blocked_is_the_headline_number() -> None:
    result = await run(DATA_CLASS_SSN, (row(decision="AUDIT"),), on_match="BLOCK")

    assert result.newly_blocked == 1
    assert result.newly_allowed == 0
    assert result.unchanged == 0
    assert result.outcomes[0].delta is Delta.NEWLY_BLOCKED


async def test_already_blocked_traffic_is_unchanged() -> None:
    result = await run(DATA_CLASS_SSN, (row(decision="BLOCK"),), on_match="BLOCK")

    assert result.newly_blocked == 0
    assert result.unchanged == 1


async def test_a_candidate_that_does_not_match_previously_blocked_traffic_is_newly_allowed() -> None:
    result = await run(
        {"type": "data_class", "class": "PII.PASSPORT"},
        (row(decision="BLOCK"),),
        on_match="BLOCK",
    )

    assert result.newly_allowed == 1
    assert result.newly_blocked == 0


async def test_a_weaker_action_is_newly_allowed_and_a_stronger_one_newly_restricted() -> None:
    weaker = await run(DATA_CLASS_SSN, (row(decision="BLOCK"),), on_match="WARN")
    stronger = await run(DATA_CLASS_SSN, (row(decision="AUDIT"),), on_match="REDACT")

    assert weaker.newly_allowed == 1
    assert stronger.newly_restricted == 1
    assert stronger.newly_blocked == 0


async def test_indeterminate_requests_are_never_counted_as_a_delta() -> None:
    result = await run(REGEX_LEAF, (row(decision="BLOCK"),), on_match="BLOCK")

    assert result.newly_blocked == 0
    assert result.newly_allowed == 0
    assert result.unchanged == 0
    assert result.indeterminate_requests == 1


async def test_scope_filter_selects_the_requests_it_names() -> None:
    rows = (
        row(decision_id="d1", request_id="r1"),
        row(decision_id="d2", request_id="r2", scope={**FULL_SCOPE, "team_id": "research"}),
    )
    filtered = await run(DATA_CLASS_SSN, rows, scope_filter=ScopeFilter(team_id="research"))
    globbed = await run(DATA_CLASS_SSN, rows, scope_filter=ScopeFilter(model="gpt-*"))

    assert filtered.evaluated_requests == 1
    assert filtered.sample_decision_ids == ("d2",)
    assert globbed.evaluated_requests == 2


async def test_breakdowns_carry_the_indeterminate_population_beside_the_matches() -> None:
    rows = (
        row(decision_id="d1", request_id="r1"),
        row(decision_id="d2", request_id="r2", scope={**FULL_SCOPE, "team_id": "research"}),
    )
    condition = {"operator": "ANY", "conditions": [{"type": "team", "value": "payments"}, REGEX_LEAF]}
    result = await run(condition, rows)

    assert result.by_team["payments"].matched == 1
    assert result.by_team["payments"].indeterminate == 0
    assert result.by_team["research"].matched == 0
    assert result.by_team["research"].indeterminate == 1


async def test_daily_series_shows_the_spike() -> None:
    day_one = datetime(2026, 6, 1, 9, 0, tzinfo=timezone.utc)
    rows = (
        row(decision_id="d1", request_id="r1", created_at=day_one),
        row(decision_id="d2", request_id="r2", created_at=day_one + timedelta(days=1)),
        row(decision_id="d3", request_id="r3", created_at=day_one + timedelta(days=1, hours=2)),
    )
    result = await run(DATA_CLASS_SSN, rows)

    assert [(point.day, point.newly_blocked) for point in result.daily] == [
        ("2026-06-01", 1),
        ("2026-06-02", 2),
    ]


async def test_the_cap_produces_a_labelled_partial_rather_than_a_silent_truncation() -> None:
    rows = tuple(
        row(
            decision_id=f"d{index}",
            request_id=f"r{index}",
            created_at=datetime(2026, 6, 1, tzinfo=timezone.utc) + timedelta(hours=index),
        )
        for index in range(10)
    )
    result = await run(DATA_CLASS_SSN, rows, row_cap=4)

    assert result.truncated is True
    assert result.scanned_decisions == 4
    assert result.evaluated_requests == 4
    assert result.scanned_through == datetime(2026, 6, 1, 3, tzinfo=timezone.utc)
    assert result_view(result)["window"]["partial"] is True  # type: ignore[index]


async def test_a_window_inside_the_cap_is_not_partial() -> None:
    rows = tuple(row(decision_id=f"d{index}", request_id=f"r{index}") for index in range(4))
    result = await run(DATA_CLASS_SSN, rows, row_cap=4)

    assert result.truncated is False
    assert result.scanned_through is None


async def test_paging_walks_the_whole_window() -> None:
    rows = tuple(row(decision_id=f"d{index}", request_id=f"r{index}") for index in range(7))
    window = await load_window(FakeDecisions(rows), where={}, row_cap=100, page_size=2)

    assert len(window.decisions) == 7
    assert window.truncated is False


async def test_compare_isolates_what_an_edit_changes() -> None:
    rows = (row(decision_id="d1", decision="AUDIT"),)
    window = await load_window(FakeDecisions(rows), where={}, row_cap=100)
    comparison = await compare(
        left=candidate(DATA_CLASS_SSN, on_match="BLOCK"),
        right=candidate({**DATA_CLASS_SSN, "min_count": 5}, on_match="BLOCK"),
        window=window,
        window_start=WINDOW_START,
        window_end=WINDOW_END,
    )

    assert comparison.only_left_blocks == 1
    assert comparison.only_right_blocks == 0
    assert comparison.both_block == 0
    assert comparison.divergent_requests == 1
    assert comparison.left.newly_blocked == 1
    assert comparison.right.newly_blocked == 0


async def test_simulation_never_serialises_a_content_field() -> None:
    """A rogue receipt column and a rogue finding key both stop at the parser."""
    leaked = "123-45-6789"
    rows = (
        Row(
            decision_id="d1",
            request_id="r1",
            created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
            direction="input",
            decision="AUDIT",
            policy_id="p1",
            shadow=False,
            scope_json=json.dumps({**FULL_SCOPE, "content": leaked, "prompt": leaked}),
            matched_classifiers=json.dumps(
                [{"class": "PII.SSN", "count": 1, "confidence": 0.9, "value": leaked, "text": leaked}]
            ),
        ),
    )
    result = await run(DATA_CLASS_SSN, rows)
    serialised = json.dumps(result_view(result), default=str)

    assert result.matched_requests == 1
    assert leaked not in serialised
    assert not _forbidden_keys(json.loads(serialised))
    assert not _forbidden_keys(dict(_scope_of(rows[0]).fields))


def test_the_replay_types_have_nowhere_to_put_content() -> None:
    assert {field.name for field in fields(RecordedDecision)}.isdisjoint(
        {"content", "text", "prompt", "response", "value", "raw"}
    )


def _forbidden_keys(payload: object) -> set[str]:
    forbidden = {"content", "text", "prompt", "response", "value", "raw", "match", "snippet"}
    match payload:
        case dict():
            nested = {key for value in payload.values() for key in _forbidden_keys(value)}
            return (set(payload) & forbidden) | nested
        case list():
            return {key for entry in payload for key in _forbidden_keys(entry)}
        case _:
            return set()


def _scope_of(entry: Row) -> RecordedScope:
    parsed = parse_decision_row(entry)
    assert parsed is not None
    return parsed.scope


def _parsed(rows: Sequence[Row]) -> Sequence[RecordedDecision]:
    parsed = tuple(parse_decision_row(entry) for entry in rows)
    assert all(entry is not None for entry in parsed)
    return tuple(entry for entry in parsed if entry is not None)


def test_simulation_needs_test_and_view_but_never_enforce() -> None:
    """A lead has to be able to evaluate a change they are not permitted to make."""
    lead = UserAPIKeyAuth(api_key="sk-lead", metadata={"scopes": ["dlp:test", "dlp:view_decisions"]})
    enforcer_only = UserAPIKeyAuth(api_key="sk-enf", metadata={"scopes": ["dlp:enforce", "dlp:approve"]})

    _require_simulation_scopes(lead)

    with pytest.raises(HTTPException) as refused:
        _require_simulation_scopes(enforcer_only)
    assert refused.value.status_code == 403


def test_a_window_wider_than_the_maximum_is_refused() -> None:
    with pytest.raises(HTTPException) as refused:
        _validated_window(
            RetroWindow(start=WINDOW_START, end=WINDOW_START + timedelta(days=MAX_WINDOW_DAYS + 1))
        )
    assert refused.value.status_code == 400


def test_an_invalid_candidate_is_a_400_and_not_an_empty_simulation() -> None:
    with pytest.raises(HTTPException) as refused:
        _candidate({"wit_dps_version": "2.0", "name": "broken"})
    assert refused.value.status_code == 400


async def test_a_persisted_run_stores_counts_and_ids_but_no_findings() -> None:
    leaked = "123-45-6789"
    rows = (
        Row(
            decision_id="d1",
            request_id="r1",
            created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
            direction="input",
            decision="AUDIT",
            policy_id="p1",
            shadow=False,
            scope_json=json.dumps({**FULL_SCOPE, "content": leaked}),
            matched_classifiers=json.dumps([{"class": "PII.SSN", "count": 1, "confidence": 0.9, "value": leaked}]),
        ),
    )
    result = await run(DATA_CLASS_SSN, rows)
    payload = RunRequest(
        candidate=dict(DATA_CLASS_SSN),
        window=RetroWindow(start=WINDOW_START, end=WINDOW_END),
        name="pci tightening",
    )
    stored = _run_row("run-1", result, payload, UserAPIKeyAuth(api_key="sk-lead", user_id="u-7"))
    serialised = json.dumps(stored, default=str)

    assert stored["newly_blocked"] == 1
    assert stored["status"] == "complete"
    assert json.loads(str(stored["sample_decision_ids"])) == ["d1"]
    assert leaked not in serialised
    assert not _forbidden_keys(json.loads(serialised))


def test_truth_enum_has_exactly_three_states() -> None:
    """The tri-state is the design. A fourth value, or a missing one, breaks it."""
    assert {member.value for member in Truth} == {"true", "false", "indeterminate"}
