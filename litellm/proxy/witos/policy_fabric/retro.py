"""Retroactive policy simulation (§2.4, §2.5, §2.7).

Shadow mode answers "what will this policy do from here on", which means waiting
weeks before anyone dares switch it on. This module answers the question a
security lead actually has to put in a change ticket: what would it have done to
the traffic we already had. That is answerable only because a decision receipt
records the matched canonical classifiers and the scope of every evaluation
while deliberately storing none of the content.

The same rule that makes the receipt safe is what bounds this feature. A leaf
that needs the original text cannot be replayed, ever, and the only honest
result for it is INDETERMINATE. Collapsing it to "no match" would produce a
policy that reads as harmless because its regex clause was silently skipped,
which is the most dangerous output this feature could have. Every count here
therefore carries its indeterminate population beside it instead of folding it
into the negatives.

The AST walk is the runtime's. `evaluator.evaluate` decides how ALL, ANY and NOT
combine and `ACTION_PRECEDENCE` decides which action wins, so a simulation and a
live evaluation cannot disagree about what a policy means. Only leaf resolution
differs, and only by replaying what was recorded instead of scanning text that
no longer exists.

Two limits are inherent rather than incidental, and `COVERAGE_CAVEATS` carries
them into every response. Requests that matched no policy at the time left no
receipt, so every match count is a lower bound. And a recorded finding set is
what the detectors running at the time found, so a class nothing was looking for
reads as absent rather than as proven absent.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from itertools import groupby
from types import MappingProxyType
from typing import Final, Protocol, TypeVar

from litellm.proxy.witos.policy_fabric.canonical import WitDpsPolicy
from litellm.proxy.witos.policy_fabric.compiler import CompiledPolicy
from litellm.proxy.witos.policy_fabric.condition_ast import (
    ClassifierLeaf,
    ContextLeaf,
    LeafNode,
    PatternLeaf,
    TermsLeaf,
    ToolArgumentLeaf,
    iter_leaves,
)
from litellm.proxy.witos.policy_fabric.evaluator import (
    EvaluationContext,
    LeafOutcome,
    PolicyLeafResolver,
    RequestScope,
    Truth,
    evaluate,
    glob_match,
)
from litellm.proxy.witos.policy_fabric.types import (
    ACTION_PRECEDENCE,
    CompileStatus,
    EvaluationDirection,
    EvaluationTier,
    LeafType,
    PolicyAction,
    policy_covers_direction,
)

DEFAULT_ROW_CAP: Final = 50_000
MAX_ROW_CAP: Final = 250_000
MAX_SAMPLE_DECISION_IDS: Final = 50
UNATTRIBUTED: Final = "(unattributed)"

_PAGE_SIZE: Final = 1_000
_EnumT: Final = TypeVar("_EnumT", bound=Enum)

COVERAGE_CAVEATS: Final[tuple[str, ...]] = (
    "Only requests that produced a decision receipt are in scope. A request that matched no policy at the "
    "time left no receipt, so every match count here is a lower bound.",
    "Recorded findings are what the detectors running at the time found. A canonical class nothing was "
    "looking for reads as absent, never as proven absent.",
    "regex, dictionary and keyword leaves replay as indeterminate because no content is stored. They are "
    "never counted as not matching.",
)


class IndeterminateReason(str, Enum):
    """Why a leaf could not be replayed. Never a reason to report `no match`."""

    CONTENT_NOT_RETAINED = "content_not_retained"
    CLASSIFICATION_DETAIL_NOT_RECORDED = "classification_detail_not_recorded"
    SCOPE_NOT_RECORDED = "scope_not_recorded"


class Delta(str, Enum):
    NEWLY_BLOCKED = "newly_blocked"
    NEWLY_RESTRICTED = "newly_restricted"
    NEWLY_ALLOWED = "newly_allowed"
    UNCHANGED = "unchanged"
    UNDETERMINED = "undetermined"


# Replayable from the recorded finding set.
FINDING_REPLAYABLE_LEAVES: Final[frozenset[LeafType]] = frozenset({LeafType.DATA_CLASS})

# Structurally impossible: replaying these needs the original text, which is the
# one thing a receipt is guaranteed not to hold.
CONTENT_LEAVES: Final[frozenset[LeafType]] = frozenset(
    {LeafType.REGEX, LeafType.DICTIONARY, LeafType.KEYWORD, LeafType.TOOL_ARGUMENT}
)

# The receipt records which canonical class fired, not its sensitivity band, its
# label or which engine produced it.
CLASSIFICATION_DETAIL_LEAVES: Final[frozenset[LeafType]] = frozenset(
    {LeafType.SENSITIVITY, LeafType.SENSITIVITY_LABEL, LeafType.CLASSIFICATION_SOURCE}
)

# Replayable exactly when the receipt's scope carries the field. Today's writer
# records six of them, so the rest resolve indeterminate until it records more.
_LEAF_SCOPE_FIELD: Final[Mapping[LeafType, str]] = MappingProxyType(
    {
        LeafType.IDENTITY: "user_id",
        LeafType.TEAM: "team_id",
        LeafType.ORGANIZATION: "organization_id",
        LeafType.APPLICATION: "application",
        LeafType.MODEL: "model",
        LeafType.MODEL_GROUP: "model_group",
        LeafType.PROVIDER: "provider",
        LeafType.DESTINATION: "destination",
        LeafType.GROUP: "groups",
        LeafType.FILE_TYPE: "file_types",
        LeafType.TOOL: "tool_name",
    }
)

_ENTITY_SCOPE_FIELD: Final[Mapping[str, str]] = MappingProxyType(
    {
        "organization": "organization_id",
        "team": "team_id",
        "user": "user_id",
        "key_alias": "key_alias",
        "end_user": "end_user_id",
    }
)

_EXCEPTION_SCOPE_FIELD: Final[Mapping[str, str]] = MappingProxyType(
    {
        "identity": "user_id",
        "group": "groups",
        "team": "team_id",
        "organization": "organization_id",
        "application": "application",
        "key_alias": "key_alias",
        "model": "model",
    }
)

# A whitelist rather than a filter: a scope key outside this set cannot reach a
# `RecordedScope`, so no future column and no malformed row can smuggle content
# through the receipt's JSON.
_SCOPE_FIELDS: Final[frozenset[str]] = frozenset(
    {*_LEAF_SCOPE_FIELD.values(), *_ENTITY_SCOPE_FIELD.values(), *_EXCEPTION_SCOPE_FIELD.values()}
)


@dataclass(frozen=True, slots=True)
class RecordedFinding:
    """A receipt's classifier entry. Class, how many, how confident. Never what."""

    classifier: str
    count: int
    confidence: float


@dataclass(frozen=True, slots=True)
class RecordedScope:
    fields: Mapping[str, str | None]

    def records(self, field: str) -> bool:
        """Whether the receipt carried the field at all.

        A recorded `None` means the request genuinely had no team; an absent key
        means nobody wrote one down. The first supports a negative answer, the
        second cannot.
        """
        return field in self.fields

    def value(self, field: str) -> str | None:
        return self.fields.get(field)

    def as_request_scope(self) -> RequestScope:
        return RequestScope(
            organization_id=self.value("organization_id"),
            team_id=self.value("team_id"),
            user_id=self.value("user_id"),
            key_alias=self.value("key_alias"),
            end_user_id=self.value("end_user_id"),
            application=self.value("application"),
            model=self.value("model"),
            model_group=self.value("model_group"),
            provider=self.value("provider"),
            destination=self.value("destination"),
        )


@dataclass(frozen=True, slots=True)
class RecordedDecision:
    decision_id: str
    request_id: str
    created_at: datetime
    direction: EvaluationDirection
    action: PolicyAction
    policy_id: str
    shadow: bool
    scope: RecordedScope
    findings: tuple[RecordedFinding, ...]


@dataclass(frozen=True, slots=True)
class ReplayUnit:
    """One recorded evaluation: one request, one direction, every policy that fired.

    The unit rather than the row is what a candidate is replayed against.
    Findings recorded by different policies describe the same payload, so
    evaluating each row separately would hide a candidate that matches only on
    the union, and hiding a match is the failure mode that matters here.
    """

    request_id: str
    direction: EvaluationDirection
    created_at: datetime
    scope: RecordedScope
    findings: tuple[RecordedFinding, ...]
    recorded_action: PolicyAction
    decision_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScopeFilter:
    team_id: str | None = None
    model: str | None = None
    application: str | None = None
    user_id: str | None = None

    def keeps(self, scope: RecordedScope) -> bool:
        return all(
            pattern is None or glob_match(pattern, scope.value(field))
            for field, pattern in (
                ("team_id", self.team_id),
                ("model", self.model),
                ("application", self.application),
                ("user_id", self.user_id),
            )
        )


NO_SCOPE_FILTER: Final = ScopeFilter()


@dataclass(frozen=True, slots=True)
class UnitOutcome:
    unit: ReplayUnit
    truth: Truth
    candidate_action: PolicyAction | None
    delta: Delta
    reasons: frozenset[IndeterminateReason]
    leaf_types: frozenset[LeafType]


@dataclass(frozen=True, slots=True)
class Breakdown:
    matched: int
    newly_blocked: int
    indeterminate: int


@dataclass(frozen=True, slots=True)
class DailyPoint:
    day: str
    evaluated: int
    matched: int
    newly_blocked: int
    indeterminate: int


@dataclass(frozen=True, slots=True)
class RetroResult:
    candidate_name: str
    candidate_hash: str
    candidate_action: PolicyAction
    window_start: datetime
    window_end: datetime
    scanned_decisions: int
    evaluated_requests: int
    matched_requests: int
    unmatched_requests: int
    indeterminate_requests: int
    newly_blocked: int
    newly_restricted: int
    newly_allowed: int
    unchanged: int
    action_distribution: Mapping[str, int]
    by_team: Mapping[str, Breakdown]
    by_model: Mapping[str, Breakdown]
    by_application: Mapping[str, Breakdown]
    by_direction: Mapping[str, Breakdown]
    by_class: Mapping[str, Breakdown]
    daily: tuple[DailyPoint, ...]
    indeterminate_reasons: Mapping[str, int]
    indeterminate_leaf_types: Mapping[str, int]
    sample_decision_ids: tuple[str, ...]
    truncated: bool
    row_cap: int
    scanned_through: datetime | None
    outcomes: tuple[UnitOutcome, ...]


@dataclass(frozen=True, slots=True)
class RetroComparison:
    left: RetroResult
    right: RetroResult
    only_left_blocks: int
    only_right_blocks: int
    both_block: int
    divergent_requests: int
    indeterminate_either: int


@dataclass(frozen=True, slots=True)
class LoadedWindow:
    decisions: tuple[RecordedDecision, ...]
    scanned: int
    truncated: bool

    @property
    def scanned_through(self) -> datetime | None:
        """The last instant actually covered. Only meaningful when truncated."""
        return max((decision.created_at for decision in self.decisions), default=None)


class DecisionSource(Protocol):
    """The slice of the decision table the loader needs. Injectable, so the
    simulator is testable without a database."""

    async def find_many(
        self,
        where: Mapping[str, object] | None = ...,
        order: Mapping[str, str] | None = ...,
        skip: int = ...,
        take: int = ...,
    ) -> Sequence[object]: ...


def indeterminate_reason(leaf: LeafNode, scope: RecordedScope) -> IndeterminateReason | None:
    """Why this leaf cannot be replayed against this receipt, or None if it can."""
    match leaf:
        case PatternLeaf() | TermsLeaf() | ToolArgumentLeaf():
            return IndeterminateReason.CONTENT_NOT_RETAINED
        case ClassifierLeaf():
            if leaf.leaf_type in FINDING_REPLAYABLE_LEAVES:
                return None
            return IndeterminateReason.CLASSIFICATION_DETAIL_NOT_RECORDED
        case ContextLeaf():
            field: Final = _LEAF_SCOPE_FIELD.get(leaf.leaf_type)
            if field is None or not scope.records(field):
                return IndeterminateReason.SCOPE_NOT_RECORDED
            return None


def parse_decision_row(row: object) -> RecordedDecision | None:
    """Project a receipt row onto the fields a replay may see.

    Every field is read by name, so a column this function does not know about
    cannot reach a `RecordedDecision`, whatever a future migration adds.
    """
    decision_id: Final = getattr(row, "decision_id", None)
    request_id: Final = getattr(row, "request_id", None)
    created_at: Final = getattr(row, "created_at", None)
    if not isinstance(decision_id, str) or not isinstance(request_id, str) or not isinstance(created_at, datetime):
        return None
    direction: Final = _enum_or_none(EvaluationDirection, getattr(row, "direction", None))
    action: Final = _enum_or_none(PolicyAction, getattr(row, "decision", None))
    if direction is None or action is None:
        return None
    return RecordedDecision(
        decision_id=decision_id,
        request_id=request_id,
        created_at=created_at,
        direction=direction,
        action=action,
        policy_id=str(getattr(row, "policy_id", "")),
        shadow=getattr(row, "shadow", False) is True,
        scope=_recorded_scope(_decode(getattr(row, "scope_json", None))),
        findings=_recorded_findings(_decode(getattr(row, "matched_classifiers", None))),
    )


def group_units(decisions: Sequence[RecordedDecision]) -> tuple[ReplayUnit, ...]:
    ordered: Final = sorted(decisions, key=_unit_key)
    units: Final = tuple(_unit(tuple(group)) for _, group in groupby(ordered, key=_unit_key))
    return tuple(sorted(units, key=lambda unit: (unit.created_at, unit.request_id, unit.direction.value)))


async def load_window(
    source: DecisionSource,
    where: Mapping[str, object],
    row_cap: int = DEFAULT_ROW_CAP,
    page_size: int = _PAGE_SIZE,
) -> LoadedWindow:
    """Page ascending through the window, one row past the cap.

    Ascending order means a truncated run is a prefix of the window rather than
    an arbitrary sample, so `scanned_through` states exactly how much of the
    window the numbers cover. The sentinel row is what separates "exactly at the
    cap" from "there was more", because a partial result that calls itself
    complete is worse than no result.
    """
    rows: Final = await _pages(source, where, limit=row_cap + 1, page_size=page_size, acc=())
    kept: Final = rows[:row_cap]
    parsed: Final = tuple(decision for decision in (parse_decision_row(row) for row in kept) if decision is not None)
    return LoadedWindow(decisions=parsed, scanned=len(kept), truncated=len(rows) > row_cap)


async def simulate(
    candidate: WitDpsPolicy,
    window: LoadedWindow,
    window_start: datetime,
    window_end: datetime,
    scope_filter: ScopeFilter = NO_SCOPE_FILTER,
    row_cap: int = DEFAULT_ROW_CAP,
) -> RetroResult:
    kept: Final = tuple(decision for decision in window.decisions if scope_filter.keeps(decision.scope))
    units: Final = group_units(kept)
    plan: Final = replay_plan(candidate)
    leaves: Final = MappingProxyType({leaf.node_id: leaf for leaf in iter_leaves(candidate.condition)})
    outcomes: Final = tuple([await _simulate_unit(candidate, plan, leaves, unit) for unit in units])
    return _aggregate(
        candidate=candidate,
        outcomes=outcomes,
        window=window,
        window_start=window_start,
        window_end=window_end,
        row_cap=row_cap,
    )


async def compare(
    left: WitDpsPolicy,
    right: WitDpsPolicy,
    window: LoadedWindow,
    window_start: datetime,
    window_end: datetime,
    scope_filter: ScopeFilter = NO_SCOPE_FILTER,
    row_cap: int = DEFAULT_ROW_CAP,
) -> RetroComparison:
    """Replay two candidates over one window so an edit is measured against its
    predecessor rather than against a fresh set of rows."""
    left_result: Final = await simulate(left, window, window_start, window_end, scope_filter, row_cap)
    right_result: Final = await simulate(right, window, window_start, window_end, scope_filter, row_cap)
    paired: Final = tuple(zip(left_result.outcomes, right_result.outcomes, strict=True))
    return RetroComparison(
        left=left_result,
        right=right_result,
        only_left_blocks=sum(1 for one, other in paired if _blocks(one) and not _blocks(other)),
        only_right_blocks=sum(1 for one, other in paired if _blocks(other) and not _blocks(one)),
        both_block=sum(1 for one, other in paired if _blocks(one) and _blocks(other)),
        divergent_requests=sum(1 for one, other in paired if one.candidate_action is not other.candidate_action),
        indeterminate_either=sum(1 for one, other in paired if Truth.INDETERMINATE in (one.truth, other.truth)),
    )


def replay_plan(policy: WitDpsPolicy) -> CompiledPolicy:
    """A compile plan with no detectors.

    Replay never scans text, so no pattern is compiled and every leaf costs the
    same. Leaving `patterns` and `presidio_leaves` empty is also what makes the
    runtime resolver answer INDETERMINATE for the leaves this module must never
    answer False for.
    """
    return CompiledPolicy(
        policy=policy,
        patterns=MappingProxyType({}),
        presidio_leaves=MappingProxyType({}),
        delegated_leaves=frozenset(),
        tiers=MappingProxyType({leaf.node_id: EvaluationTier.CONTEXT for leaf in iter_leaves(policy.condition)}),
        compile_status=CompileStatus.COMPILED_LOCAL,
    )


@dataclass(frozen=True, slots=True)
class ReplayLeafResolver:
    """Recorded findings and scope in, tri-state out.

    The guard is the whole point. Without it the runtime resolver would answer
    False for a scope field nobody recorded, because `glob_match` against None
    is False, and "we never wrote that down" would silently become "it did not
    match".
    """

    unit: ReplayUnit
    runtime: PolicyLeafResolver

    async def resolve(self, leaf: LeafNode, context: EvaluationContext) -> LeafOutcome:
        if indeterminate_reason(leaf, self.unit.scope) is not None:
            return LeafOutcome(truth=Truth.INDETERMINATE)
        match leaf:
            case ClassifierLeaf():
                return _replay_data_class(leaf, self.unit.findings)
            case _:
                return await self.runtime.resolve(leaf, context)


async def _simulate_unit(
    candidate: WitDpsPolicy,
    plan: CompiledPolicy,
    leaves: Mapping[str, LeafNode],
    unit: ReplayUnit,
) -> UnitOutcome:
    gate: Final = _applicability(candidate, unit)
    if gate is Truth.FALSE:
        return _outcome(unit, Truth.FALSE, candidate, reasons=(), leaf_types=())
    context: Final = EvaluationContext(content="", direction=unit.direction, scope=unit.scope.as_request_scope())
    evaluated: Final = await evaluate(
        plan, context, ReplayLeafResolver(unit=unit, runtime=PolicyLeafResolver(compiled=plan))
    )
    truth: Final = _all_of((gate, evaluated.truth))
    unreplayable: Final = tuple(
        (leaf, reason)
        for leaf, reason in (
            (leaves[node_id], indeterminate_reason(leaves[node_id], unit.scope))
            for node_id in evaluated.visited_leaf_ids
            if node_id in leaves
        )
        if reason is not None
    )
    scope_gap: Final = (IndeterminateReason.SCOPE_NOT_RECORDED,) if gate is Truth.INDETERMINATE else ()
    return _outcome(
        unit,
        truth,
        candidate,
        reasons=(*(reason for _, reason in unreplayable), *scope_gap),
        leaf_types=tuple(_leaf_type_of(leaf) for leaf, _ in unreplayable),
    )


def _outcome(
    unit: ReplayUnit,
    truth: Truth,
    candidate: WitDpsPolicy,
    reasons: Sequence[IndeterminateReason],
    leaf_types: Sequence[LeafType],
) -> UnitOutcome:
    action: Final = _candidate_action(truth, candidate.actions.on_match)
    return UnitOutcome(
        unit=unit,
        truth=truth,
        candidate_action=action,
        delta=_delta(action, unit.recorded_action),
        reasons=frozenset(reasons) if truth is Truth.INDETERMINATE else frozenset(),
        leaf_types=frozenset(leaf_types) if truth is Truth.INDETERMINATE else frozenset(),
    )


def _candidate_action(truth: Truth, on_match: PolicyAction) -> PolicyAction | None:
    match truth:
        case Truth.TRUE:
            return on_match
        case Truth.FALSE:
            return PolicyAction.ALLOW
        case Truth.INDETERMINATE:
            return None


def _delta(candidate: PolicyAction | None, recorded: PolicyAction) -> Delta:
    """Candidate against what the recorded policy set actually did.

    `newly_allowed` compares the candidate alone with the effective recorded
    action, so it is exposure only when the candidate replaces what ran. For an
    additional policy it is noise, and `compare` is the honest tool for an edit.
    """
    if candidate is None:
        return Delta.UNDETERMINED
    if candidate is PolicyAction.BLOCK and recorded is not PolicyAction.BLOCK:
        return Delta.NEWLY_BLOCKED
    candidate_rank: Final = ACTION_PRECEDENCE[candidate]
    recorded_rank: Final = ACTION_PRECEDENCE[recorded]
    if candidate_rank > recorded_rank:
        return Delta.NEWLY_RESTRICTED
    if candidate_rank < recorded_rank:
        return Delta.NEWLY_ALLOWED
    return Delta.UNCHANGED


def _replay_data_class(leaf: ClassifierLeaf, findings: Sequence[RecordedFinding]) -> LeafOutcome:
    """`min_confidence` is replayed against the entry's recorded confidence, which
    the receipt writer stores as the maximum over the findings it summarised. A
    mixed-confidence entry can therefore contribute its full count, which
    over-reports matches rather than under-reporting them."""
    matched: Final = sum(
        finding.count
        for finding in findings
        if finding.classifier == leaf.value and finding.confidence >= leaf.min_confidence
    )
    return LeafOutcome(truth=Truth.TRUE if matched >= leaf.min_count else Truth.FALSE)


def _applicability(policy: WitDpsPolicy, unit: ReplayUnit) -> Truth:
    """Tenant, direction, declared scope and exceptions, as a tri-state.

    `scope.applicable_policies` answers the same question for live traffic, but
    it answers in two values. A scope clause resting on a field the receipt does
    not carry has to come back indeterminate here rather than excluding the
    request.
    """
    if not policy_covers_direction(policy.direction, unit.direction):
        return Truth.FALSE
    return _all_of(
        (
            _entities(policy, unit.scope),
            _patterns(policy.scope.models, ("model", "model_group"), unit.scope),
            _patterns(policy.scope.applications, ("application",), unit.scope),
            _negate(_exceptions(policy, unit.scope)),
        )
    )


def _entities(policy: WitDpsPolicy, scope: RecordedScope) -> Truth:
    if not policy.scope.entities:
        return Truth.TRUE
    return _any_of(
        tuple(
            _glob_truth(entity.entity_id, _ENTITY_SCOPE_FIELD.get(entity.entity_type), scope)
            for entity in policy.scope.entities
        )
    )


def _exceptions(policy: WitDpsPolicy, scope: RecordedScope) -> Truth:
    if not policy.exceptions:
        return Truth.FALSE
    return _any_of(
        tuple(
            _glob_truth(exception.value, _EXCEPTION_SCOPE_FIELD.get(exception.exception_type), scope)
            for exception in policy.exceptions
        )
    )


def _patterns(patterns: Sequence[str], fields: Sequence[str], scope: RecordedScope) -> Truth:
    if not patterns:
        return Truth.TRUE
    return _any_of(tuple(_glob_truth(pattern, field, scope) for pattern in patterns for field in fields))


def _glob_truth(pattern: str, field: str | None, scope: RecordedScope) -> Truth:
    if field is None:
        return Truth.FALSE
    if not scope.records(field):
        return Truth.INDETERMINATE
    return Truth.TRUE if glob_match(pattern, scope.value(field)) else Truth.FALSE


def _all_of(truths: Sequence[Truth]) -> Truth:
    if Truth.FALSE in truths:
        return Truth.FALSE
    if Truth.INDETERMINATE in truths:
        return Truth.INDETERMINATE
    return Truth.TRUE


def _any_of(truths: Sequence[Truth]) -> Truth:
    if Truth.TRUE in truths:
        return Truth.TRUE
    if Truth.INDETERMINATE in truths:
        return Truth.INDETERMINATE
    return Truth.FALSE


def _negate(truth: Truth) -> Truth:
    match truth:
        case Truth.TRUE:
            return Truth.FALSE
        case Truth.FALSE:
            return Truth.TRUE
        case Truth.INDETERMINATE:
            return Truth.INDETERMINATE


def _aggregate(
    candidate: WitDpsPolicy,
    outcomes: tuple[UnitOutcome, ...],
    window: LoadedWindow,
    window_start: datetime,
    window_end: datetime,
    row_cap: int,
) -> RetroResult:
    deltas: Final = _tally(outcome.delta.value for outcome in outcomes)
    matched: Final = tuple(outcome for outcome in outcomes if outcome.truth is Truth.TRUE)
    indeterminate: Final = tuple(outcome for outcome in outcomes if outcome.truth is Truth.INDETERMINATE)
    return RetroResult(
        candidate_name=candidate.name,
        candidate_hash=candidate.canonical_hash,
        candidate_action=candidate.actions.on_match,
        window_start=window_start,
        window_end=window_end,
        scanned_decisions=window.scanned,
        evaluated_requests=len(outcomes),
        matched_requests=len(matched),
        unmatched_requests=sum(1 for outcome in outcomes if outcome.truth is Truth.FALSE),
        indeterminate_requests=len(indeterminate),
        newly_blocked=deltas.get(Delta.NEWLY_BLOCKED.value, 0),
        newly_restricted=deltas.get(Delta.NEWLY_RESTRICTED.value, 0),
        newly_allowed=deltas.get(Delta.NEWLY_ALLOWED.value, 0),
        unchanged=deltas.get(Delta.UNCHANGED.value, 0),
        action_distribution=_tally(
            outcome.candidate_action.value for outcome in matched if outcome.candidate_action
        ),
        by_team=_breakdown(outcomes, lambda unit: (_scope_key(unit, "team_id"),)),
        by_model=_breakdown(outcomes, lambda unit: (_scope_key(unit, "model"),)),
        by_application=_breakdown(outcomes, lambda unit: (_scope_key(unit, "application"),)),
        by_direction=_breakdown(outcomes, lambda unit: (unit.direction.value,)),
        by_class=_breakdown(outcomes, _classes_of),
        daily=_daily(outcomes),
        indeterminate_reasons=_tally(reason.value for outcome in indeterminate for reason in outcome.reasons),
        indeterminate_leaf_types=_tally(
            leaf_type.value for outcome in indeterminate for leaf_type in outcome.leaf_types
        ),
        sample_decision_ids=tuple(decision_id for outcome in matched for decision_id in outcome.unit.decision_ids)[
            :MAX_SAMPLE_DECISION_IDS
        ],
        truncated=window.truncated,
        row_cap=row_cap,
        scanned_through=window.scanned_through if window.truncated else None,
        outcomes=outcomes,
    )


def _breakdown(
    outcomes: Sequence[UnitOutcome],
    keys: Callable[[ReplayUnit], tuple[str, ...]],
) -> Mapping[str, Breakdown]:
    matched: Final = _tally(key for outcome in outcomes if outcome.truth is Truth.TRUE for key in keys(outcome.unit))
    blocked: Final = _tally(
        key for outcome in outcomes if outcome.delta is Delta.NEWLY_BLOCKED for key in keys(outcome.unit)
    )
    unknown: Final = _tally(
        key for outcome in outcomes if outcome.truth is Truth.INDETERMINATE for key in keys(outcome.unit)
    )
    return MappingProxyType(
        {
            key: Breakdown(
                matched=matched.get(key, 0),
                newly_blocked=blocked.get(key, 0),
                indeterminate=unknown.get(key, 0),
            )
            for key in sorted(matched.keys() | blocked.keys() | unknown.keys())
        }
    )


def _tally(values: Iterable[str]) -> Mapping[str, int]:
    """One O(n) pass, frozen before it leaves.

    `Counter` is the only mutable structure in this module, it exists for the
    length of this call, and a 90 day window is too many rows to count any other
    way.
    """
    return MappingProxyType(dict(Counter(values)))  # mutable-ok: local tally, frozen on the way out


def _daily(outcomes: Sequence[UnitOutcome]) -> tuple[DailyPoint, ...]:
    evaluated: Final = _tally(_day(outcome) for outcome in outcomes)
    matched: Final = _tally(_day(outcome) for outcome in outcomes if outcome.truth is Truth.TRUE)
    blocked: Final = _tally(_day(outcome) for outcome in outcomes if outcome.delta is Delta.NEWLY_BLOCKED)
    unknown: Final = _tally(_day(outcome) for outcome in outcomes if outcome.truth is Truth.INDETERMINATE)
    return tuple(
        DailyPoint(
            day=day,
            evaluated=evaluated.get(day, 0),
            matched=matched.get(day, 0),
            newly_blocked=blocked.get(day, 0),
            indeterminate=unknown.get(day, 0),
        )
        for day in sorted(evaluated)
    )


def _day(outcome: UnitOutcome) -> str:
    return outcome.unit.created_at.date().isoformat()


def _scope_key(unit: ReplayUnit, field: str) -> str:
    value: Final = unit.scope.value(field)
    return value if value else UNATTRIBUTED


def _classes_of(unit: ReplayUnit) -> tuple[str, ...]:
    return tuple(sorted(frozenset(finding.classifier for finding in unit.findings))) or (UNATTRIBUTED,)


def _blocks(outcome: UnitOutcome) -> bool:
    return outcome.candidate_action is PolicyAction.BLOCK


def _leaf_type_of(leaf: LeafNode) -> LeafType:
    match leaf:
        case PatternLeaf():
            return LeafType.REGEX
        case ToolArgumentLeaf():
            return LeafType.TOOL_ARGUMENT
        case ClassifierLeaf() | TermsLeaf() | ContextLeaf():
            return leaf.leaf_type


def _unit_key(decision: RecordedDecision) -> tuple[str, str]:
    return (decision.request_id, decision.direction.value)


def _unit(rows: tuple[RecordedDecision, ...]) -> ReplayUnit:
    enforced: Final = tuple(row.action for row in rows if not row.shadow)
    return ReplayUnit(
        request_id=rows[0].request_id,
        direction=rows[0].direction,
        created_at=min(row.created_at for row in rows),
        scope=max(rows, key=lambda row: len(row.scope.fields)).scope,
        findings=tuple(finding for row in rows for finding in row.findings),
        recorded_action=max(enforced, key=lambda action: ACTION_PRECEDENCE[action], default=PolicyAction.ALLOW),
        decision_ids=tuple(row.decision_id for row in rows),
    )


async def _pages(
    source: DecisionSource,
    where: Mapping[str, object],
    limit: int,
    page_size: int,
    acc: tuple[object, ...],
) -> tuple[object, ...]:
    remaining: Final = limit - len(acc)
    if remaining <= 0:
        return acc
    page: Final = await source.find_many(
        where=where,
        order={"created_at": "asc"},  # mutable-ok: prisma order fragment is dict-shaped
        skip=len(acc),
        take=min(page_size, remaining),
    )
    if not page:
        return acc
    return await _pages(source, where, limit, page_size, (*acc, *tuple(page)))


def _decode(raw: object) -> object:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
    return raw


def _recorded_scope(raw: object) -> RecordedScope:
    if not isinstance(raw, Mapping):
        return RecordedScope(fields=MappingProxyType({}))
    return RecordedScope(
        fields=MappingProxyType(
            {
                str(key): value
                for key, value in raw.items()
                if str(key) in _SCOPE_FIELDS and (value is None or isinstance(value, str))
            }
        )
    )


def _recorded_findings(raw: object) -> tuple[RecordedFinding, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    return tuple(finding for finding in (_recorded_finding(entry) for entry in raw) if finding is not None)


def _recorded_finding(entry: object) -> RecordedFinding | None:
    if not isinstance(entry, Mapping):
        return None
    classifier: Final = entry.get("class")
    if not isinstance(classifier, str) or not classifier:
        return None
    return RecordedFinding(
        classifier=classifier,
        count=_count_of(entry.get("count")),
        confidence=_confidence_of(entry.get("confidence")),
    )


def _count_of(raw: object) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        return 1
    return raw


def _confidence_of(raw: object) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return 0.0
    return min(1.0, max(0.0, float(raw)))


def _enum_or_none(enum_type: type[_EnumT], raw: object) -> _EnumT | None:
    if not isinstance(raw, str):
        return None
    try:
        return enum_type(raw)
    except ValueError:
        return None
