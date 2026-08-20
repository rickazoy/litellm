"""Streaming output modes and the honesty rule (§2.9).

Three modes, and one rule that outranks all of them: **no mode may claim to
block content that has already been sent to the client.** A post-call "block"
on a stream the caller has already consumed is detection. It gets recorded as
detection in the receipt, reported as detection by the API, and described as
detection in the docs. Calling it prevention would tell a security team they are
protected against an exfiltration path that is, in fact, wide open.

`buffer_full` can prevent, at the cost of time-to-first-token.
`chunk_gate` can prevent anything still inside its window, and only that.
`observe_only` can never prevent anything, by construction.

State is a value. `offer` and `finish` return a new `StreamState` rather than
mutating one, so a mid-stream decision can be reasoned about against the exact
state at that moment.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from litellm.proxy.witos.policy_fabric.types import EnforcementKind, StreamingMode

DEFAULT_WINDOW_CHARS: Final = 2048
DEFAULT_OVERLAP_CHARS: Final = 512

OBSERVE_ONLY_DISCLAIMER: Final = (
    "observe_only streams every chunk as it arrives. Violations are detected after disclosure and cannot be prevented."
)
CHUNK_GATE_DISCLAIMER: Final = (
    "chunk_gate can withhold only content still inside its inspection window. "
    "Content already released to the client is detected, not prevented."
)
BUFFER_FULL_DISCLAIMER: Final = (
    "buffer_full withholds the entire response until evaluation completes, at the cost of time-to-first-token."
)


def disclaimer_for(mode: StreamingMode) -> str:
    match mode:
        case StreamingMode.BUFFER_FULL:
            return BUFFER_FULL_DISCLAIMER
        case StreamingMode.CHUNK_GATE:
            return CHUNK_GATE_DISCLAIMER
        case StreamingMode.OBSERVE_ONLY:
            return OBSERVE_ONLY_DISCLAIMER


@dataclass(frozen=True, slots=True)
class StreamState:
    buffer: str = ""
    released_chars: int = 0

    @property
    def total_chars(self) -> int:
        return self.released_chars + len(self.buffer)


@dataclass(frozen=True, slots=True)
class StreamStep:
    state: StreamState
    release: str


def offer(
    mode: StreamingMode,
    state: StreamState,
    chunk: str,
    window_chars: int = DEFAULT_WINDOW_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> StreamStep:
    combined: Final = state.buffer + chunk
    match mode:
        case StreamingMode.OBSERVE_ONLY:
            return StreamStep(
                state=StreamState(buffer="", released_chars=state.released_chars + len(combined)),
                release=combined,
            )
        case StreamingMode.BUFFER_FULL:
            return StreamStep(state=StreamState(buffer=combined, released_chars=state.released_chars), release="")
        case StreamingMode.CHUNK_GATE:
            if len(combined) < window_chars:
                return StreamStep(state=StreamState(buffer=combined, released_chars=state.released_chars), release="")
            cut: Final = max(0, len(combined) - overlap_chars)
            return StreamStep(
                state=StreamState(buffer=combined[cut:], released_chars=state.released_chars + cut),
                release=combined[:cut],
            )


def finish(state: StreamState) -> StreamStep:
    """Flush whatever the mode was holding. Only called once evaluation allows it."""
    return StreamStep(
        state=StreamState(buffer="", released_chars=state.total_chars),
        release=state.buffer,
    )


def enforcement_for(
    mode: StreamingMode,
    state: StreamState,
    finding_offsets: Sequence[tuple[int, int]],
) -> EnforcementKind:
    """Prevention only if every offending span is still unreleased.

    Offsets are absolute positions in the concatenated response text, so a span
    starting before `released_chars` has already left the building.
    """
    if not mode.prevents_disclosure:
        return EnforcementKind.DETECTION
    if not finding_offsets:
        return EnforcementKind.PREVENTION
    if any(start < state.released_chars for start, _ in finding_offsets):
        return EnforcementKind.DETECTION
    return EnforcementKind.PREVENTION


def can_prevent(mode: StreamingMode, state: StreamState) -> bool:
    return mode.prevents_disclosure and state.released_chars == 0
