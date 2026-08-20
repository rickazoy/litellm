"""The `wit_dlp` guardrail's non-streaming hooks (§2.8).

These cover the adapter layer between LiteLLM's hook signatures and the engine:
message extraction, per-message redaction offsets, the block response shape, and
the v1 tool interface.
"""

from __future__ import annotations

import pytest
from conftest import SSN_VALUE, engine_over, policy_document, scoped

from fastapi import HTTPException

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.wit_dlp import WitDlpGuardrail
from litellm.proxy.witos.policy_fabric.types import EvaluationDirection, PolicyStatus

CLEAN = "what is the weather today"
DIRTY = f"my ssn is {SSN_VALUE}"


def guardrail(policies, streaming_mode: str | None = None) -> WitDlpGuardrail:
    return WitDlpGuardrail(
        guardrail_name="wit_dlp",
        default_on=True,
        streaming_mode=streaming_mode,
        engine=engine_over(policies),
    )


def auth(org_id: str | None = None) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key="sk-test", org_id=org_id)


def chat(*contents: str) -> dict:
    return {"model": "gpt-5", "messages": [{"role": "user", "content": content} for content in contents]}


async def test_a_clean_request_passes_through_untouched() -> None:
    guard = guardrail((scoped(policy_document("PCI")),))
    assert await guard.async_pre_call_hook(auth(), None, chat(CLEAN), "completion") is None


async def test_a_blocking_policy_raises_the_documented_violation_shape() -> None:
    document = policy_document("PCI - Cardholder Data")
    document["actions"]["block_message"] = "Blocked by policy 'PCI - Cardholder Data'."
    document["source"] = {"vendor": "custom_witdps", "external_policy_id": "ext-1"}
    guard = guardrail((scoped(document),))

    with pytest.raises(HTTPException) as raised:
        await guard.async_pre_call_hook(auth(), None, chat(DIRTY), "completion")

    assert raised.value.status_code == 400
    error = raised.value.detail["error"]
    assert error["type"] == "witos_dlp_policy_violation"
    assert error["policy"] == "PCI - Cardholder Data"
    assert error["source_vendor"] == "custom_witdps"
    assert error["message"] == "Blocked by policy 'PCI - Cardholder Data'."
    assert error["enforcement"] == "prevention"


async def test_the_violation_detail_never_carries_the_matched_value() -> None:
    guard = guardrail((scoped(policy_document("PCI")),))
    with pytest.raises(HTTPException) as raised:
        await guard.async_pre_call_hook(auth(), None, chat(DIRTY), "completion")
    assert SSN_VALUE not in str(raised.value.detail)


async def test_redaction_rewrites_only_the_offending_message() -> None:
    guard = guardrail((scoped(policy_document("PII", on_match="REDACT", redact_strategy="mask")),))
    updated = await guard.async_pre_call_hook(auth(), None, chat(CLEAN, DIRTY, CLEAN), "completion")
    assert updated is not None
    messages = updated["messages"]
    assert messages[0]["content"] == CLEAN
    assert messages[2]["content"] == CLEAN
    assert SSN_VALUE not in messages[1]["content"]
    assert messages[1]["content"].startswith("my ssn is ")
    assert len(messages[1]["content"]) == len(DIRTY)


async def test_redaction_offsets_are_relative_to_the_message_they_came_from() -> None:
    """Evaluating per message is what keeps the span arithmetic correct.

    Concatenating messages first and redacting afterwards would apply an offset
    computed against the joined text to an individual message, masking the wrong
    slice of the wrong message.
    """
    guard = guardrail((scoped(policy_document("PII", on_match="REDACT"),),))
    long_prefix = "x" * 500
    updated = await guard.async_pre_call_hook(auth(), None, chat(long_prefix, DIRTY), "completion")
    assert updated is not None
    assert updated["messages"][0]["content"] == long_prefix
    assert SSN_VALUE not in updated["messages"][1]["content"]


async def test_multimodal_text_parts_are_extracted_and_rewritten() -> None:
    guard = guardrail((scoped(policy_document("PII", on_match="REDACT"),),))
    data = {
        "model": "gpt-5",
        "messages": [{"role": "user", "content": [{"type": "text", "text": DIRTY}]}],
    }
    updated = await guard.async_pre_call_hook(auth(), None, data, "completion")
    assert updated is not None
    parts = updated["messages"][0]["content"]
    assert isinstance(parts, list)
    assert SSN_VALUE not in parts[0]["text"]


async def test_a_shadow_policy_leaves_the_request_alone_but_records_it() -> None:
    guard = guardrail((scoped(policy_document("PCI"), status=PolicyStatus.SHADOW),))
    assert await guard.async_pre_call_hook(auth(), None, chat(DIRTY), "completion") is None
    receipts = guard.receipts.drain()
    assert len(receipts) == 1
    assert receipts[0].shadow is True
    assert receipts[0].decision.value == "BLOCK"


async def test_a_policy_belonging_to_another_tenant_does_not_see_the_request() -> None:
    guard = guardrail((scoped(policy_document("PCI"), organization_id="org-a"),))
    assert await guard.async_pre_call_hook(auth(org_id="org-b"), None, chat(DIRTY), "completion") is None
    assert guard.receipts.drain() == ()
    with pytest.raises(HTTPException):
        await guard.async_pre_call_hook(auth(org_id="org-a"), None, chat(DIRTY), "completion")


async def test_the_post_call_hook_evaluates_the_assistant_response() -> None:
    guard = guardrail((scoped(policy_document("PCI", direction="output")),))

    class _Message:
        content = DIRTY

    class _Choice:
        message = _Message()

    class _Response:
        choices = [_Choice()]

    with pytest.raises(HTTPException):
        await guard.async_post_call_success_hook(chat(CLEAN), auth(), _Response())


async def test_the_tool_interface_evaluates_and_records_without_enforcing() -> None:
    guard = guardrail((scoped(policy_document("PCI", direction="tool")),))
    result = await guard.evaluate_tool_payload(
        user_api_key_dict=auth(),
        request_data=chat(CLEAN),
        direction=EvaluationDirection.TOOL_OUTPUT,
        content=DIRTY,
        tool_name="salesforce_query",
        tool_arguments={"soql": "SELECT ssn FROM Contact"},
    )
    assert result is not None
    assert result.decision.action.value == "BLOCK"
    assert result.enforcement_active is False
    assert result.should_block is False
    receipts = guard.receipts.drain()
    assert receipts[0].direction is EvaluationDirection.TOOL_OUTPUT
    assert receipts[0].prevented is False


async def test_receipts_are_buffered_rather_than_written_inline() -> None:
    guard = guardrail((scoped(policy_document("PCI", on_match="AUDIT")),))
    await guard.async_pre_call_hook(auth(), None, chat(DIRTY), "completion")
    assert guard.receipts.stats().buffered == 1
