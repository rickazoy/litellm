from pydantic import Field

from .base import GuardrailConfigModel


class WitDlpGuardrailConfigModel(GuardrailConfigModel):
    wit_dlp_streaming_mode: str | None = Field(
        default=None,
        description=(
            "Output streaming mode: buffer_full, chunk_gate (default) or observe_only. "
            "observe_only never prevents disclosure; it detects violations after the client has "
            "already received the content."
        ),
    )
    wit_dlp_fail_mode: str | None = Field(
        default=None,
        description=(
            "What to do when a delegated evaluation cannot be completed: fail_open (default), "
            "fail_closed or observe. Always recorded on the decision receipt and always alerted."
        ),
    )
    wit_dlp_match_hash_salt: str | None = Field(
        default=None,
        description=(
            "Salt for the optional one-way correlation hash on decision receipts. Without a salt "
            "no hash is written, because an unsalted digest of a short structured value is reversible."
        ),
    )
    wit_dlp_cache_ttl_seconds: float | None = Field(
        default=None,
        description="Policy-set cache TTL. Policy changes also invalidate across pods via Redis pub/sub.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "WIT OS DLP Policy Fabric"
