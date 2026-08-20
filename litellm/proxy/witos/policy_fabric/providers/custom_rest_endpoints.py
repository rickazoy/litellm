"""Endpoint paths for the generic REST adapter (§2.3).

Blueprint rule: vendor endpoint paths live in a per-adapter constants module,
verified against the tenant's own documentation during integration testing. They
are never inlined at the call site and never guessed, because a guessed path
produces a 404 that looks exactly like "no policies configured".

These defaults describe the shape the generic adapter expects. A connection may
override any of them, which is what makes this adapter generic rather than a
prediction about somebody's API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

DEFAULT_POLICY_LIST_PATH: Final = "/policies"
DEFAULT_POLICY_READ_PATH: Final = "/policies/{external_policy_id}"
DEFAULT_CLASSIFIER_LIST_PATH: Final = "/classifiers"
DEFAULT_HEALTH_PATH: Final = "/health"


@dataclass(frozen=True, slots=True)
class RestEndpoints:
    policy_list: str = DEFAULT_POLICY_LIST_PATH
    policy_read: str = DEFAULT_POLICY_READ_PATH
    classifier_list: str = DEFAULT_CLASSIFIER_LIST_PATH
    health: str = DEFAULT_HEALTH_PATH

    def policy_read_for(self, external_policy_id: str) -> str:
        return self.policy_read.replace("{external_policy_id}", external_policy_id)
