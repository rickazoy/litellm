from typing import TYPE_CHECKING, Final

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .wit_dlp import WitDlpGuardrail, invalidate_local_policy_caches

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def _raw_param(litellm_params: "LitellmParams", guardrail: "Guardrail", key: str) -> object:
    """Config values arrive from YAML, so they are read back as untyped input.

    `LitellmParams` allows extra keys, and the raw guardrail dict is the
    fallback for anything the config model does not declare.
    """
    declared: Final = getattr(litellm_params, key, None)
    if declared is not None:
        return declared
    raw: Final = guardrail.get("litellm_params")
    return raw[key] if isinstance(raw, dict) and key in raw else None


def _str_param(litellm_params: "LitellmParams", guardrail: "Guardrail", key: str) -> str | None:
    value: Final = _raw_param(litellm_params, guardrail, key)
    return value if isinstance(value, str) and value else None


def _float_param(litellm_params: "LitellmParams", guardrail: "Guardrail", key: str, default: float) -> float:
    value: Final = _raw_param(litellm_params, guardrail, key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail") -> WitDlpGuardrail:
    import litellm

    callback: Final = WitDlpGuardrail(
        guardrail_name=guardrail.get("guardrail_name", "wit_dlp"),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
        streaming_mode=_str_param(litellm_params, guardrail, "wit_dlp_streaming_mode"),
        fail_mode=_str_param(litellm_params, guardrail, "wit_dlp_fail_mode"),
        match_hash_salt=_str_param(litellm_params, guardrail, "wit_dlp_match_hash_salt"),
        cache_ttl_seconds=_float_param(litellm_params, guardrail, "wit_dlp_cache_ttl_seconds", 30.0),
    )
    litellm.logging_callback_manager.add_litellm_callback(callback)
    return callback


guardrail_initializer_registry: Final = {  # mutable-ok: guardrail registry contract
    SupportedGuardrailIntegrations.WIT_DLP.value: initialize_guardrail,
}

guardrail_class_registry: Final = {  # mutable-ok: guardrail registry contract
    SupportedGuardrailIntegrations.WIT_DLP.value: WitDlpGuardrail,
}

__all__ = (
    "WitDlpGuardrail",
    "guardrail_class_registry",
    "guardrail_initializer_registry",
    "initialize_guardrail",
    "invalidate_local_policy_caches",
)
