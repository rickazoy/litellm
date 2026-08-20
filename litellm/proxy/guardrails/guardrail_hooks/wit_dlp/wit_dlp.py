"""`wit_dlp` runtime guardrail (blueprint v2 §2.8, §2.9, §2.10).

Thin adapter: LiteLLM's hook signatures in, `PolicyFabricEngine` out. All the
policy logic lives under `litellm/proxy/witos/policy_fabric/` so it can be
tested without a proxy, a database or a network.

Message content is evaluated per message rather than as one concatenated blob,
because a redaction offset only means something relative to the string it came
from. Aggregating the verdicts afterwards preserves precedence while keeping
span arithmetic honest.

Receipts are enqueued, never awaited. A decision the database is too slow to
record is still a decision that was enforced.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping, Sequence
from typing import TYPE_CHECKING, Final

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.content_text import (
    assistant_text_from_response,
    content_to_text,
    is_all_text_parts,
    merge_rewritten_text_parts,
)
from litellm.proxy.witos.policy_fabric.evaluator import RequestScope
from litellm.proxy.witos.policy_fabric.policy_cache import PolicySetCache
from litellm.proxy.witos.policy_fabric.receipts import DecisionReceiptQueue
from litellm.proxy.witos.policy_fabric.runtime import (
    CompiledPolicyStore,
    EngineConfig,
    EngineResult,
    EvaluationRequest,
    PolicyFabricEngine,
)
from litellm.proxy.witos.policy_fabric.scope import ScopedPolicy
from litellm.proxy.witos.policy_fabric.streaming import StreamState, offer
from litellm.proxy.witos.policy_fabric.types import (
    EvaluationDirection,
    FailMode,
    StreamingMode,
)
from litellm.types.guardrails import GuardrailEventHooks

if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel
    from litellm.types.utils import ModelResponseStream

VIOLATION_TYPE: Final = "witos_dlp_policy_violation"
WARNING_HEADER: Final = "x-witos-dlp-warning"

# Live caches, so an API-side policy change can invalidate every guardrail
# instance in this process without a restart.
_LIVE_CACHES: Final[list[PolicySetCache]] = []  # mutable-ok: process-local registry of live caches


def invalidate_local_policy_caches() -> None:
    for cache in _LIVE_CACHES:
        cache.invalidate()


class WitDlpGuardrail(CustomGuardrail):
    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:  # mutable-ok: LiteLLM hook contract
        return [  # mutable-ok: LiteLLM hook contract
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.during_call,
            GuardrailEventHooks.post_call,
        ]

    @staticmethod
    def get_config_model() -> type[GuardrailConfigModel] | None:
        from litellm.types.proxy.guardrails.guardrail_hooks.wit_dlp import WitDlpGuardrailConfigModel

        return WitDlpGuardrailConfigModel

    def __init__(
        self,
        guardrail_name: str = "wit_dlp",
        event_hook: object = None,
        default_on: bool = False,
        streaming_mode: str | None = None,
        fail_mode: str | None = None,
        match_hash_salt: str | None = None,
        cache_ttl_seconds: float = 30.0,
        engine: PolicyFabricEngine | None = None,
        receipt_queue: DecisionReceiptQueue | None = None,
    ) -> None:
        super().__init__(
            guardrail_name=guardrail_name,
            supported_event_hooks=self.get_supported_event_hooks(),
            event_hook=event_hook,
            default_on=default_on,
        )
        self.guardrail_provider = "wit_dlp"
        self._config: Final = EngineConfig(
            guardrail_name=guardrail_name,
            default_fail_mode=FailMode(fail_mode) if fail_mode else FailMode.FAIL_OPEN,
            streaming_mode=StreamingMode(streaming_mode) if streaming_mode else StreamingMode.CHUNK_GATE,
            match_hash_salt=match_hash_salt,
        )
        self._cache_ttl: Final = cache_ttl_seconds
        self._receipts: Final = receipt_queue if receipt_queue is not None else DecisionReceiptQueue()
        self._injected_engine: Final = engine
        self._engine: PolicyFabricEngine | None = None  # rebind-ok: built on first use, once prisma exists

    @property
    def streaming_mode(self) -> StreamingMode:
        return self._config.streaming_mode

    @property
    def receipts(self) -> DecisionReceiptQueue:
        return self._receipts

    def _get_engine(self) -> PolicyFabricEngine | None:
        if self._injected_engine is not None:
            return self._injected_engine
        if self._engine is not None:
            return self._engine
        from litellm.proxy.proxy_server import prisma_client
        from litellm.proxy.witos.policy_fabric.presidio_bridge import build_presidio_analyzer
        from litellm.proxy.witos.policy_fabric.store import load_policy_set

        if prisma_client is None:
            return None

        async def loader(organization_id: str | None) -> tuple[ScopedPolicy, ...]:
            return await load_policy_set(prisma_client, organization_id)

        cache: Final = PolicySetCache(loader=loader, ttl_seconds=self._cache_ttl)
        _LIVE_CACHES.append(cache)
        self._engine = PolicyFabricEngine(
            cache=cache,
            store=CompiledPolicyStore(),
            config=self._config,
            presidio=build_presidio_analyzer(),
        )
        return self._engine

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: object,
        data: dict,  # mutable-ok: LiteLLM hook contract
        call_type: str,
    ) -> dict | None:  # mutable-ok: LiteLLM hook contract
        if not self.should_run_guardrail(data=data, event_type=GuardrailEventHooks.pre_call):
            return None
        engine: Final = self._get_engine()
        if engine is None:
            return None
        messages: Final = data.get("messages")
        if not isinstance(messages, list):
            return None
        scope: Final = _scope_from(user_api_key_dict, data)
        request_id: Final = _request_id(data)
        results: Final = tuple(
            [
                (index, await self._evaluate_message(engine, message, scope, request_id))
                for index, message in enumerate(messages)
                if isinstance(message, dict)
            ]
        )
        evaluated: Final = tuple(result for _, result in results if result is not None)
        self._record_all(evaluated)
        self._raise_if_blocked(evaluated)
        rewrites: Final[Mapping[int, str]] = {  # mutable-ok: LiteLLM hook contract
            index: result.rewritten_content
            for index, result in results
            if result is not None and result.should_rewrite and result.rewritten_content is not None
        }
        if not rewrites:
            return None
        return {  # mutable-ok: LiteLLM hook contract
            **data,
            "messages": [  # mutable-ok: LiteLLM hook contract
                _rewrite_message(message, rewrites[index]) if index in rewrites else message
                for index, message in enumerate(messages)
            ],
        }

    async def async_post_call_success_hook(
        self,
        data: dict,  # mutable-ok: LiteLLM hook contract
        user_api_key_dict: UserAPIKeyAuth,
        response: object,
    ) -> object:
        if not self.should_run_guardrail(data=data, event_type=GuardrailEventHooks.post_call):
            return response
        engine: Final = self._get_engine()
        if engine is None:
            return response
        text: Final = assistant_text_from_response(response)
        if not text:
            return response
        result: Final = await engine.evaluate(
            EvaluationRequest(
                request_id=_request_id(data),
                content=text,
                direction=EvaluationDirection.OUTPUT,
                scope=_scope_from(user_api_key_dict, data),
            )
        )
        self._record_all((result,))
        self._raise_if_blocked((result,))
        return response

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: AsyncGenerator[ModelResponseStream, None],
        request_data: dict,  # mutable-ok: LiteLLM hook contract
    ) -> AsyncGenerator[ModelResponseStream, None]:
        engine: Final = self._get_engine()
        if engine is None or not self.should_run_guardrail(data=request_data, event_type=GuardrailEventHooks.post_call):
            async for chunk in response:
                yield chunk
            return

        mode: Final = self._config.streaming_mode
        scope: Final = _scope_from(user_api_key_dict, request_data)
        request_id: Final = _request_id(request_data)

        if mode is StreamingMode.OBSERVE_ONLY:
            async for chunk in self._observe_only(engine, response, scope, request_id):
                yield chunk
            return

        state: StreamState = StreamState()  # rebind-ok: stream position advances per released chunk
        pending: Final[list[ModelResponseStream]] = []  # mutable-ok: chunks held back pending inspection
        pending_text: str = ""  # rebind-ok: text of the chunks currently in `pending`

        async for chunk in response:
            chunk_text: Final = _chunk_text(chunk)
            pending.append(chunk)
            pending_text += chunk_text
            step: Final = offer(mode, state, chunk_text)
            if not step.release:
                state = step.state
                continue
            verdict: Final = await self._evaluate_stream(engine, pending_text, scope, request_id, state)
            if verdict is not None and verdict.should_block:
                raise _violation(verdict)
            state = step.state
            for buffered in pending:
                yield buffered
            pending.clear()
            pending_text = ""

        if not pending:
            return
        final_verdict: Final = await self._evaluate_stream(engine, pending_text, scope, request_id, state)
        if final_verdict is not None and final_verdict.should_block:
            raise _violation(final_verdict)
        for buffered in pending:
            yield buffered

    async def _observe_only(
        self,
        engine: PolicyFabricEngine,
        response: AsyncGenerator[ModelResponseStream, None],
        scope: RequestScope,
        request_id: str,
    ) -> AsyncGenerator[ModelResponseStream, None]:
        """Release first, inspect afterwards.

        The yield happens before any evaluation on purpose. It makes it
        impossible for this mode to withhold a chunk, which is what
        `observe_only` promises, and the receipts it produces are labelled
        detection rather than prevention.
        """
        streamed: str = ""  # rebind-ok: accumulates what the client has already received
        async for chunk in response:
            yield chunk
            streamed += _chunk_text(chunk)
        if not streamed:
            return
        await self._evaluate_stream(
            engine,
            streamed,
            scope,
            request_id,
            StreamState(buffer="", released_chars=len(streamed)),
        )

    async def evaluate_tool_payload(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        request_data: dict,  # mutable-ok: LiteLLM hook contract
        direction: EvaluationDirection,
        content: str,
        tool_name: str | None = None,
        tool_arguments: Mapping[str, str] | None = None,
    ) -> EngineResult | None:
        """Tool, MCP and RAG payload evaluation (§2.10).

        The interface ships in v1 so callers and adapters can be written against
        it. Enforcement on these directions is v1.5: the engine returns
        `enforcement_active=False`, receipts are written, and nothing is blocked
        or rewritten. Claiming otherwise would be the streaming honesty problem
        in a different costume.
        """
        engine: Final = self._get_engine()
        if engine is None:
            return None
        result: Final = await engine.evaluate(
            EvaluationRequest(
                request_id=_request_id(request_data),
                content=content,
                direction=direction,
                scope=_scope_from(user_api_key_dict, request_data),
                tool_name=tool_name,
                tool_arguments=tool_arguments or {},  # mutable-ok: LiteLLM hook contract
            )
        )
        self._record_all((result,))
        return result

    async def _evaluate_message(
        self,
        engine: PolicyFabricEngine,
        message: Mapping[str, object],
        scope: RequestScope,
        request_id: str,
    ) -> EngineResult | None:
        text: Final = content_to_text(message.get("content"))
        if not text:
            return None
        return await engine.evaluate(
            EvaluationRequest(
                request_id=request_id,
                content=text,
                direction=EvaluationDirection.INPUT,
                scope=scope,
            )
        )

    async def _evaluate_stream(
        self,
        engine: PolicyFabricEngine,
        text: str,
        scope: RequestScope,
        request_id: str,
        state: StreamState,
    ) -> EngineResult | None:
        if not text:
            return None
        result: Final = await engine.evaluate(
            EvaluationRequest(
                request_id=request_id,
                content=text,
                direction=EvaluationDirection.OUTPUT,
                scope=scope,
                stream_state=state,
                streaming_mode=self._config.streaming_mode,
            )
        )
        self._record_all((result,))
        return result

    def _record_all(self, results: Sequence[EngineResult]) -> None:
        for result in results:
            for receipt in result.receipts:
                self._receipts.record(receipt)
            for failure in result.compile_failures:
                verbose_proxy_logger.warning("WIT OS DLP: skipping uncompilable policy %s", failure)

    def _raise_if_blocked(self, results: Sequence[EngineResult]) -> None:
        for result in results:
            if result.should_block:
                raise _violation(result)


def _violation(result: EngineResult) -> HTTPException:
    deciding: Final = result.decision.deciding
    return HTTPException(
        status_code=400,
        detail={  # mutable-ok: LiteLLM hook contract
            "error": {  # mutable-ok: LiteLLM hook contract
                "type": VIOLATION_TYPE,
                "policy": None if deciding is None else deciding.policy_name,
                "policy_id": None if deciding is None else deciding.policy_id,
                "source_vendor": None if deciding is None else deciding.provider,
                "policy_url": None,
                "message": (
                    (deciding.block_message if deciding is not None and deciding.block_message else None)
                    or "Blocked by a WIT OS data protection policy."
                ),
                "enforcement": result.enforcement_kind.value,
            }
        },
    )


def _rewrite_message(message: object, new_text: str) -> object:
    if not isinstance(message, dict):
        return message
    content: Final = message.get("content")
    if is_all_text_parts(content) and isinstance(content, list):
        return {  # mutable-ok: LiteLLM hook contract
            **message,
            "content": merge_rewritten_text_parts(content, new_text),
        }  # mutable-ok: LiteLLM hook contract
    return {**message, "content": new_text}  # mutable-ok: LiteLLM hook contract


def _chunk_text(chunk: object) -> str:
    choices: Final = getattr(chunk, "choices", None)
    if not isinstance(choices, list) or not choices:
        return ""
    delta: Final = getattr(choices[0], "delta", None)
    content: Final = getattr(delta, "content", None) if delta is not None else None
    return content if isinstance(content, str) else ""


def _request_id(data: Mapping[str, object]) -> str:
    metadata: Final = data.get("litellm_metadata") or data.get("metadata")
    if isinstance(metadata, Mapping):
        candidate: Final = metadata.get("litellm_call_id") or metadata.get("request_id")
        if isinstance(candidate, str) and candidate:
            return candidate
    fallback: Final = data.get("litellm_call_id")
    return fallback if isinstance(fallback, str) and fallback else "unknown"


def _scope_from(user_api_key_dict: UserAPIKeyAuth, data: Mapping[str, object]) -> RequestScope:
    metadata: Final = data.get("litellm_metadata") or data.get("metadata") or {}  # mutable-ok: LiteLLM hook contract
    application: Final = metadata.get("application") if isinstance(metadata, Mapping) else None
    model: Final = data.get("model")
    return RequestScope(
        organization_id=user_api_key_dict.org_id,
        team_id=user_api_key_dict.team_id,
        user_id=user_api_key_dict.user_id,
        key_alias=user_api_key_dict.key_alias,
        end_user_id=user_api_key_dict.end_user_id,
        application=application if isinstance(application, str) else None,
        model=model if isinstance(model, str) else None,
        model_group=model if isinstance(model, str) else None,
        provider=None,
        destination=None,
    )
