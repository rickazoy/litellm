"""Streaming modes and the honesty rule (§2.9, §2.14).

The rule under test: no mode may claim to block content already sent to the
client. `observe_only` must always report detection, and `chunk_gate` must
report detection once the offending span has left the process.
"""

from __future__ import annotations

from typing import AsyncIterator

from conftest import SSN_VALUE, engine_over, policy_document, scoped

from litellm.proxy.guardrails.guardrail_hooks.wit_dlp import WitDlpGuardrail
from litellm.proxy.witos.policy_fabric.streaming import (
    DEFAULT_OVERLAP_CHARS,
    DEFAULT_WINDOW_CHARS,
    StreamState,
    can_prevent,
    disclaimer_for,
    enforcement_for,
    finish,
    offer,
)
from litellm.proxy.witos.policy_fabric.types import EnforcementKind, StreamingMode


class _Delta:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.delta = _Delta(content)


class _Chunk:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]


async def _stream(parts: tuple[str, ...]) -> AsyncIterator[_Chunk]:
    for part in parts:
        yield _Chunk(part)


def test_observe_only_declares_that_it_cannot_prevent() -> None:
    assert StreamingMode.OBSERVE_ONLY.prevents_disclosure is False
    assert StreamingMode.CHUNK_GATE.prevents_disclosure is True
    assert StreamingMode.BUFFER_FULL.prevents_disclosure is True
    assert "cannot be prevented" in disclaimer_for(StreamingMode.OBSERVE_ONLY)
    assert "already released" in disclaimer_for(StreamingMode.CHUNK_GATE)


def test_observe_only_releases_every_chunk_immediately() -> None:
    state = StreamState()
    step = offer(StreamingMode.OBSERVE_ONLY, state, "hello")
    assert step.release == "hello"
    assert step.state.buffer == ""
    assert step.state.released_chars == 5


def test_buffer_full_releases_nothing_until_the_end() -> None:
    state = StreamState()
    for part in ("a" * 5000, "b" * 5000):
        step = offer(StreamingMode.BUFFER_FULL, state, part)
        assert step.release == ""
        state = step.state
    assert state.released_chars == 0
    assert len(state.buffer) == 10000
    flushed = finish(state)
    assert len(flushed.release) == 10000


def test_chunk_gate_holds_an_overlap_so_a_span_crossing_a_boundary_stays_inspectable() -> None:
    state = StreamState()
    step = offer(StreamingMode.CHUNK_GATE, state, "x" * DEFAULT_WINDOW_CHARS)
    assert len(step.release) == DEFAULT_WINDOW_CHARS - DEFAULT_OVERLAP_CHARS
    assert len(step.state.buffer) == DEFAULT_OVERLAP_CHARS
    assert step.state.released_chars == DEFAULT_WINDOW_CHARS - DEFAULT_OVERLAP_CHARS


def test_chunk_gate_holds_everything_below_the_window() -> None:
    step = offer(StreamingMode.CHUNK_GATE, StreamState(), "short")
    assert step.release == ""
    assert step.state.released_chars == 0


def test_observe_only_never_reports_prevention_even_with_nothing_released() -> None:
    assert enforcement_for(StreamingMode.OBSERVE_ONLY, StreamState(), ()) is EnforcementKind.DETECTION
    assert enforcement_for(StreamingMode.OBSERVE_ONLY, StreamState(), ((0, 5),)) is EnforcementKind.DETECTION
    assert can_prevent(StreamingMode.OBSERVE_ONLY, StreamState()) is False


def test_a_span_already_released_is_detection_not_prevention() -> None:
    released = StreamState(buffer="tail", released_chars=100)
    assert enforcement_for(StreamingMode.CHUNK_GATE, released, ((10, 20),)) is EnforcementKind.DETECTION
    assert enforcement_for(StreamingMode.CHUNK_GATE, released, ((120, 130),)) is EnforcementKind.PREVENTION


def test_buffer_full_with_nothing_released_is_prevention() -> None:
    assert enforcement_for(StreamingMode.BUFFER_FULL, StreamState(buffer="all"), ((0, 3),)) is (
        EnforcementKind.PREVENTION
    )


async def test_observe_only_streams_the_violation_through_and_labels_it_detection() -> None:
    guardrail = WitDlpGuardrail(
        guardrail_name="wit_dlp",
        default_on=True,
        streaming_mode=StreamingMode.OBSERVE_ONLY.value,
        engine=engine_over((scoped(policy_document("PCI")),)),
    )
    parts = ("the ssn is ", SSN_VALUE, " and that is that")
    received = [
        chunk
        async for chunk in guardrail.async_post_call_streaming_iterator_hook(
            user_api_key_dict=_auth(), response=_stream(parts), request_data={}
        )
    ]
    assert len(received) == len(parts), "observe_only must never withhold a chunk"
    receipts = guardrail.receipts.drain()
    assert receipts, "the violation must still be recorded"
    assert all(receipt.decision.value == "BLOCK" for receipt in receipts)
    assert all(receipt.prevented is False for receipt in receipts)
    assert all(receipt.streaming_mode is StreamingMode.OBSERVE_ONLY for receipt in receipts)


async def test_buffer_full_blocks_before_any_chunk_reaches_the_client() -> None:
    guardrail = WitDlpGuardrail(
        guardrail_name="wit_dlp",
        default_on=True,
        streaming_mode=StreamingMode.BUFFER_FULL.value,
        engine=engine_over((scoped(policy_document("PCI")),)),
    )
    received: list[object] = []
    raised = False
    try:
        async for chunk in guardrail.async_post_call_streaming_iterator_hook(
            user_api_key_dict=_auth(), response=_stream(("the ssn is ", SSN_VALUE)), request_data={}
        ):
            received.append(chunk)
    except Exception as err:  # noqa: BLE001  # the guardrail raises an HTTPException
        raised = True
        assert "witos_dlp_policy_violation" in str(getattr(err, "detail", err))
    assert raised
    assert received == [], "buffer_full must not release anything it went on to block"
    receipts = guardrail.receipts.drain()
    assert any(receipt.prevented for receipt in receipts)


async def test_chunk_gate_blocks_a_violation_still_inside_its_window() -> None:
    guardrail = WitDlpGuardrail(
        guardrail_name="wit_dlp",
        default_on=True,
        streaming_mode=StreamingMode.CHUNK_GATE.value,
        engine=engine_over((scoped(policy_document("PCI")),)),
    )
    received: list[object] = []
    raised = False
    try:
        async for chunk in guardrail.async_post_call_streaming_iterator_hook(
            user_api_key_dict=_auth(), response=_stream(("the ssn is ", SSN_VALUE, " ok")), request_data={}
        ):
            received.append(chunk)
    except Exception:  # noqa: BLE001  # the guardrail raises an HTTPException
        raised = True
    assert raised
    assert received == []


async def test_a_clean_stream_passes_through_every_mode_untouched() -> None:
    for mode in StreamingMode:
        guardrail = WitDlpGuardrail(
            guardrail_name="wit_dlp",
            default_on=True,
            streaming_mode=mode.value,
            engine=engine_over((scoped(policy_document("PCI")),)),
        )
        parts = ("nothing ", "sensitive ", "here")
        received = [
            chunk
            async for chunk in guardrail.async_post_call_streaming_iterator_hook(
                user_api_key_dict=_auth(), response=_stream(parts), request_data={}
            )
        ]
        assert len(received) == len(parts), mode


def _auth():
    from litellm.proxy._types import UserAPIKeyAuth

    return UserAPIKeyAuth(api_key="sk-test")
