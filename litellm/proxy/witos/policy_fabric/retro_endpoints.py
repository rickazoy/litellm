"""`/witos/dlp/retro/*` API (§2.12).

Three rules shape this surface.

Simulation never requires `dlp:enforce`. Evaluating a change is not making it,
and the person who has to answer "what would this have done" is usually not yet
the person allowed to switch it on. `dlp:test` plus `dlp:view_decisions` is the
whole grant, and nothing here writes a policy, a receipt or a guardrail decision.

Nothing content-shaped is emitted. Responses carry counts, canonical class names,
scope values and decision ids. `RecordedDecision` has no field that could hold
matched text, so there is nothing to redact on the way out.

A capped window says so. Rows are read in ascending time and one row past the
cap, so a truncated run reports the instant it actually reached instead of
presenting a prefix as if it were the whole window.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Final

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.witos.policy_fabric.canonical import (
    PolicyValidationFailure,
    WitDpsPolicy,
    parse_policy,
)
from litellm.proxy.witos.policy_fabric.rbac import DLPPermission, require_permission, tenant_filter
from litellm.proxy.witos.policy_fabric.retro import (
    CLASSIFICATION_DETAIL_LEAVES,
    CONTENT_LEAVES,
    COVERAGE_CAVEATS,
    DEFAULT_ROW_CAP,
    FINDING_REPLAYABLE_LEAVES,
    MAX_ROW_CAP,
    Breakdown,
    LoadedWindow,
    RetroComparison,
    RetroResult,
    ScopeFilter,
    compare,
    load_window,
    simulate,
)
from litellm.proxy.witos.policy_fabric.store import tables_for
from litellm.proxy.witos.policy_fabric.types import EvaluationDirection

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

router: Final = APIRouter(prefix="/retro", tags=["WIT OS DLP"])  # mutable-ok: fastapi tags argument

MAX_WINDOW_DAYS: Final = 400
_MAX_RUN_PAGE: Final = 200


class RetroWindow(BaseModel):
    start: datetime
    end: datetime


class RetroFilters(BaseModel):
    organization_id: str | None = None
    team_id: str | None = None
    model: str | None = None
    application: str | None = None
    user_id: str | None = None
    direction: EvaluationDirection | None = None
    include_shadow: bool = False


class SimulateRequest(BaseModel):
    candidate: dict[str, object] = Field(...)  # mutable-ok: dict-shaped JSON body
    window: RetroWindow
    filters: RetroFilters = RetroFilters()
    row_cap: int = Field(default=DEFAULT_ROW_CAP, ge=1, le=MAX_ROW_CAP)


class RunRequest(SimulateRequest):
    name: str | None = None


class CompareRequest(BaseModel):
    candidate_a: dict[str, object] = Field(...)  # mutable-ok: dict-shaped JSON body
    candidate_b: dict[str, object] = Field(...)  # mutable-ok: dict-shaped JSON body
    window: RetroWindow
    filters: RetroFilters = RetroFilters()
    row_cap: int = Field(default=DEFAULT_ROW_CAP, ge=1, le=MAX_ROW_CAP)


@router.post("/simulate")
async def simulate_candidate(
    payload: SimulateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Mapping[str, object]:
    """Replay an unsaved draft. Nothing is written, so a draft can be tried before it exists."""
    _require_simulation_scopes(user_api_key_dict)
    result: Final = await _simulate(payload, _candidate(payload.candidate), user_api_key_dict)
    return result_view(result)


@router.post("/compare")
async def compare_candidates(
    payload: CompareRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Mapping[str, object]:
    _require_simulation_scopes(user_api_key_dict)
    left: Final = _candidate(payload.candidate_a)
    right: Final = _candidate(payload.candidate_b)
    window: Final = _validated_window(payload.window)
    loaded: Final = await _load(payload.filters, window, payload.row_cap, user_api_key_dict)
    comparison: Final = await compare(
        left=left,
        right=right,
        window=loaded,
        window_start=window.start,
        window_end=window.end,
        scope_filter=_scope_filter(payload.filters),
        row_cap=payload.row_cap,
    )
    return comparison_view(comparison)


@router.post("/runs")
async def create_run(
    payload: RunRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Mapping[str, object]:
    """Persist a simulation so a change ticket can cite it."""
    _require_simulation_scopes(user_api_key_dict)
    result: Final = await _simulate(payload, _candidate(payload.candidate), user_api_key_dict)
    run_id: Final = str(uuid.uuid4())
    tables: Final = tables_for(_require_prisma())
    await tables.retro_runs.create(data=_run_row(run_id, result, payload, user_api_key_dict))
    return {"run_id": run_id, **result_view(result)}  # mutable-ok: dict-shaped JSON body


@router.get("/runs")
async def list_runs(
    candidate_policy_hash: str | None = None,
    limit: int = Query(default=50, le=_MAX_RUN_PAGE),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Mapping[str, object]:
    require_permission(user_api_key_dict, DLPPermission.VIEW_DECISIONS)
    tables: Final = tables_for(_require_prisma())
    hash_filter: Final = {"candidate_policy_hash": candidate_policy_hash}  # mutable-ok: prisma where fragment
    filters: Final = {} if candidate_policy_hash is None else hash_filter  # mutable-ok: prisma where fragment
    rows: Final = await tables.retro_runs.find_many(
        where=_merged_where(filters, user_api_key_dict),
        order={"created_at": "desc"},  # mutable-ok: prisma order fragment is dict-shaped
        take=limit,
    )
    return {"runs": [run_view(row) for row in rows]}  # mutable-ok: dict-shaped JSON body


@router.get("/runs/{run_id}")
async def get_run(
    run_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Mapping[str, object]:
    require_permission(user_api_key_dict, DLPPermission.VIEW_DECISIONS)
    tables: Final = tables_for(_require_prisma())
    row: Final = await tables.retro_runs.find_first(
        where=_merged_where({"run_id": run_id}, user_api_key_dict)  # mutable-ok: prisma where fragment is dict-shaped
    )
    if row is None:
        raise HTTPException(status_code=404, detail={"error": "retro run not found"})  # mutable-ok: dict-shaped body
    return run_view(row)


def result_view(result: RetroResult) -> Mapping[str, object]:
    return {  # mutable-ok: dict-shaped JSON body
        "candidate": {  # mutable-ok: dict-shaped JSON body
            "name": result.candidate_name,
            "canonical_hash": result.candidate_hash,
            "on_match": result.candidate_action.value,
        },
        "headline": {  # mutable-ok: dict-shaped JSON body
            "newly_blocked": result.newly_blocked,
            "newly_blocked_is_lower_bound": True,
            "indeterminate_requests": result.indeterminate_requests,
        },
        "window": {  # mutable-ok: dict-shaped JSON body
            "start": result.window_start,
            "end": result.window_end,
            "scanned_through": result.scanned_through,
            "partial": result.truncated,
        },
        "coverage": {  # mutable-ok: dict-shaped JSON body
            "scanned_decisions": result.scanned_decisions,
            "evaluated_requests": result.evaluated_requests,
            "row_cap": result.row_cap,
            "basis": "recorded decision receipts",
            "caveats": list(COVERAGE_CAVEATS),  # mutable-ok: dict-shaped JSON body
        },
        "requests": {  # mutable-ok: dict-shaped JSON body
            "matched": result.matched_requests,
            "not_matched": result.unmatched_requests,
            "indeterminate": result.indeterminate_requests,
        },
        "delta": {  # mutable-ok: dict-shaped JSON body
            "newly_blocked": result.newly_blocked,
            "newly_restricted": result.newly_restricted,
            "newly_allowed": result.newly_allowed,
            "unchanged": result.unchanged,
            "undetermined": result.indeterminate_requests,
            "basis": (
                "candidate action against the effective recorded action. `newly_allowed` is exposure only "
                "when the candidate replaces the recorded policy, not when it is added alongside it."
            ),
        },
        "action_distribution": dict(result.action_distribution),  # mutable-ok: dict-shaped JSON body
        "by_team": _breakdown_view(result.by_team),
        "by_model": _breakdown_view(result.by_model),
        "by_application": _breakdown_view(result.by_application),
        "by_direction": _breakdown_view(result.by_direction),
        "by_class": _breakdown_view(result.by_class),
        "daily": [  # mutable-ok: dict-shaped JSON body
            {  # mutable-ok: dict-shaped JSON body
                "day": point.day,
                "evaluated": point.evaluated,
                "matched": point.matched,
                "newly_blocked": point.newly_blocked,
                "indeterminate": point.indeterminate,
            }
            for point in result.daily
        ],
        "indeterminate": indeterminate_view(result),
        "sample_decision_ids": list(result.sample_decision_ids),  # mutable-ok: dict-shaped JSON body
    }


def indeterminate_view(result: RetroResult) -> Mapping[str, object]:
    return {  # mutable-ok: dict-shaped JSON body
        "requests": result.indeterminate_requests,
        "reasons": dict(result.indeterminate_reasons),  # mutable-ok: dict-shaped JSON body
        "leaf_types": dict(result.indeterminate_leaf_types),  # mutable-ok: dict-shaped JSON body
        "replayable_leaf_types": sorted(leaf.value for leaf in FINDING_REPLAYABLE_LEAVES),
        "content_dependent_leaf_types": sorted(leaf.value for leaf in CONTENT_LEAVES),
        "classification_detail_leaf_types": sorted(leaf.value for leaf in CLASSIFICATION_DETAIL_LEAVES),
        "note": "An indeterminate leaf is never counted as a non-match.",
    }


def comparison_view(comparison: RetroComparison) -> Mapping[str, object]:
    return {  # mutable-ok: dict-shaped JSON body
        "a": result_view(comparison.left),
        "b": result_view(comparison.right),
        "diff": {  # mutable-ok: dict-shaped JSON body
            "only_a_blocks": comparison.only_left_blocks,
            "only_b_blocks": comparison.only_right_blocks,
            "both_block": comparison.both_block,
            "divergent_requests": comparison.divergent_requests,
            "indeterminate_in_either": comparison.indeterminate_either,
        },
    }


def run_view(row: object) -> Mapping[str, object]:
    return {  # mutable-ok: dict-shaped JSON body
        "run_id": getattr(row, "run_id", None),
        "created_at": getattr(row, "created_at", None),
        "created_by": getattr(row, "created_by", None),
        "organization_id": getattr(row, "organization_id", None),
        "name": getattr(row, "name", None),
        "candidate_policy_hash": getattr(row, "candidate_policy_hash", None),
        "candidate_policy_name": getattr(row, "candidate_policy_name", None),
        "window_start": getattr(row, "window_start", None),
        "window_end": getattr(row, "window_end", None),
        "filters": _decoded(row, "filters_json"),
        "status": getattr(row, "status", None),
        "scanned_decisions": getattr(row, "scanned_decisions", None),
        "evaluated_requests": getattr(row, "evaluated_requests", None),
        "matched_requests": getattr(row, "matched_requests", None),
        "indeterminate_requests": getattr(row, "indeterminate_requests", None),
        "newly_blocked": getattr(row, "newly_blocked", None),
        "newly_allowed": getattr(row, "newly_allowed", None),
        "unchanged": getattr(row, "unchanged", None),
        "aggregates": _decoded(row, "aggregates_json"),
        "indeterminate": _decoded(row, "indeterminate_json"),
        "sample_decision_ids": _decoded(row, "sample_decision_ids"),
        "row_cap": getattr(row, "row_cap", None),
        "truncated": getattr(row, "truncated", None),
    }


def _breakdown_view(breakdowns: Mapping[str, Breakdown]) -> Mapping[str, Mapping[str, int]]:
    return {  # mutable-ok: dict-shaped JSON body
        key: {  # mutable-ok: dict-shaped JSON body
            "matched": value.matched,
            "newly_blocked": value.newly_blocked,
            "indeterminate": value.indeterminate,
        }
        for key, value in breakdowns.items()
    }


async def _simulate(
    payload: SimulateRequest,
    candidate: WitDpsPolicy,
    caller: UserAPIKeyAuth,
) -> RetroResult:
    window: Final = _validated_window(payload.window)
    loaded: Final = await _load(payload.filters, window, payload.row_cap, caller)
    return await simulate(
        candidate=candidate,
        window=loaded,
        window_start=window.start,
        window_end=window.end,
        scope_filter=_scope_filter(payload.filters),
        row_cap=payload.row_cap,
    )


async def _load(
    filters: RetroFilters,
    window: RetroWindow,
    row_cap: int,
    caller: UserAPIKeyAuth,
) -> LoadedWindow:
    tables: Final = tables_for(_require_prisma())
    return await load_window(source=tables.decisions, where=_decision_where(filters, window, caller), row_cap=row_cap)


def _decision_where(
    filters: RetroFilters,
    window: RetroWindow,
    caller: UserAPIKeyAuth,
) -> Mapping[str, object]:
    """Column-level filters only.

    `team`, `model` and `application` live inside `scope_json`, so they are
    applied by `ScopeFilter` after parsing. That means the row cap counts rows
    scanned rather than rows kept, which is why a capped run reports how far
    through the window it got.
    """
    base: Final[Mapping[str, object]] = {  # mutable-ok: prisma where fragment is dict-shaped
        "created_at": {"gte": window.start, "lte": window.end},  # mutable-ok: prisma where fragment is dict-shaped
        **({} if filters.include_shadow else {"shadow": False}),  # mutable-ok: prisma where fragment is dict-shaped
        **_optional("direction", None if filters.direction is None else filters.direction.value),
        **_optional("organization_id", filters.organization_id),
    }
    return _merged_where(base, caller)


def _optional(field: str, value: str | None) -> Mapping[str, object]:
    return {} if value is None else {field: value}  # mutable-ok: prisma where fragment is dict-shaped


def _scope_filter(filters: RetroFilters) -> ScopeFilter:
    return ScopeFilter(
        team_id=filters.team_id,
        model=filters.model,
        application=filters.application,
        user_id=filters.user_id,
    )


def _run_row(
    run_id: str,
    result: RetroResult,
    payload: RunRequest,
    caller: UserAPIKeyAuth,
) -> Mapping[str, object]:
    """Counts, ids and filters. No finding ever reaches a column here."""
    return {  # mutable-ok: prisma create payload is dict-shaped
        "run_id": run_id,
        "created_by": caller.user_id,
        "organization_id": payload.filters.organization_id or caller.org_id,
        "name": payload.name,
        "candidate_policy_hash": result.candidate_hash,
        "candidate_policy_name": result.candidate_name,
        "window_start": result.window_start,
        "window_end": result.window_end,
        "filters_json": json.dumps(payload.filters.model_dump(mode="json")),
        "status": "partial" if result.truncated else "complete",
        "scanned_decisions": result.scanned_decisions,
        "evaluated_requests": result.evaluated_requests,
        "matched_requests": result.matched_requests,
        "indeterminate_requests": result.indeterminate_requests,
        "newly_blocked": result.newly_blocked,
        "newly_restricted": result.newly_restricted,
        "newly_allowed": result.newly_allowed,
        "unchanged": result.unchanged,
        "aggregates_json": json.dumps(_aggregates(result), default=str),
        "indeterminate_json": json.dumps(indeterminate_view(result)),
        "sample_decision_ids": json.dumps(result.sample_decision_ids),
        "row_cap": result.row_cap,
        "truncated": result.truncated,
    }


def _aggregates(result: RetroResult) -> Mapping[str, object]:
    return {  # mutable-ok: dict-shaped JSON body
        "action_distribution": dict(result.action_distribution),  # mutable-ok: dict-shaped JSON body
        "by_team": _breakdown_view(result.by_team),
        "by_model": _breakdown_view(result.by_model),
        "by_application": _breakdown_view(result.by_application),
        "by_direction": _breakdown_view(result.by_direction),
        "by_class": _breakdown_view(result.by_class),
        "daily": [  # mutable-ok: dict-shaped JSON body
            {  # mutable-ok: dict-shaped JSON body
                "day": point.day,
                "evaluated": point.evaluated,
                "matched": point.matched,
                "newly_blocked": point.newly_blocked,
                "indeterminate": point.indeterminate,
            }
            for point in result.daily
        ],
        "scanned_through": result.scanned_through,
    }


def _candidate(document: Mapping[str, object]) -> WitDpsPolicy:
    parsed: Final = parse_policy(document)
    if isinstance(parsed, PolicyValidationFailure):
        raise HTTPException(
            status_code=400,
            detail={  # mutable-ok: dict-shaped JSON body
                "error": "candidate policy is not a valid WIT-DPS document",
                "errors": list(parsed.errors),  # mutable-ok: dict-shaped JSON body
            },
        )
    return parsed


def _validated_window(window: RetroWindow) -> RetroWindow:
    if window.end <= window.start:
        raise HTTPException(
            status_code=400,
            detail={"error": "window end must be after window start"},  # mutable-ok: dict-shaped JSON body
        )
    if window.end - window.start > timedelta(days=MAX_WINDOW_DAYS):
        raise HTTPException(
            status_code=400,
            detail={  # mutable-ok: dict-shaped JSON body
                "error": f"window exceeds the {MAX_WINDOW_DAYS} day maximum"
            },
        )
    if window.end > datetime.now(timezone.utc) + timedelta(days=1):
        raise HTTPException(
            status_code=400,
            detail={"error": "window end is in the future"},  # mutable-ok: dict-shaped JSON body
        )
    return window


def _require_simulation_scopes(caller: UserAPIKeyAuth) -> None:
    """`dlp:enforce` is deliberately absent. Evaluating a change is not making it."""
    require_permission(caller, DLPPermission.TEST)
    require_permission(caller, DLPPermission.VIEW_DECISIONS)


def _merged_where(base: Mapping[str, object], caller: UserAPIKeyAuth) -> Mapping[str, object]:
    scoped: Final = tenant_filter(caller)
    return base if scoped is None else {**base, **scoped}  # mutable-ok: prisma where fragment is dict-shaped


def _decoded(row: object, field: str) -> object:
    raw: Final = getattr(row, field, None)
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
    return raw


def _require_prisma() -> PrismaClient:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail=CommonProxyErrors.db_not_connected_error.value)
    return prisma_client
