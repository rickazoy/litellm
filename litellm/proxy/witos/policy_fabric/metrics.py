"""Prometheus metrics for the policy fabric (§2.14).

Registered lazily on first use, exactly like `SpendLogCleanupMetrics`, because
`prometheus_client` is an optional extra and a metrics problem must never fail a
request. Every handle stays `| None` and is null-checked at the call site.

`dlp_shadow_would_block_total` is the metric that makes shadow mode worth
running: it is the number that tells an operator what activating a policy would
have cost them, before they activate it.
"""

from __future__ import annotations

from typing import ClassVar, Final

from litellm._logging import verbose_proxy_logger

_ACTION_LABELS: Final = ("guardrail_name", "policy_id", "decision", "direction", "shadow")
_PROVIDER_LABELS: Final = ("provider", "connection_id")
_CONNECTION_LABELS: Final = ("connection_id",)


class DLPMetrics:
    _initialized: ClassVar[bool] = False
    evaluations: ClassVar[object | None] = None
    blocks: ClassVar[object | None] = None
    redactions: ClassVar[object | None] = None
    provider_latency: ClassVar[object | None] = None
    provider_errors: ClassVar[object | None] = None
    fail_open: ClassVar[object | None] = None
    fail_closed: ClassVar[object | None] = None
    policy_sync_errors: ClassVar[object | None] = None
    policy_drift: ClassVar[object | None] = None
    shadow_would_block: ClassVar[object | None] = None

    @classmethod
    def _ensure_initialized(cls) -> None:
        if cls._initialized:
            return
        cls._initialized = True
        try:
            from prometheus_client import Counter, Histogram

            cls.evaluations = Counter(  # mutable-ok: prometheus_client label tuples
                "dlp_evaluations_total", "DLP policy evaluations", labelnames=_ACTION_LABELS
            )  # mutable-ok: prometheus_client label tuples
            cls.blocks = Counter(  # mutable-ok: prometheus_client label tuples
                "dlp_blocks_total", "Requests blocked by a DLP policy", labelnames=_ACTION_LABELS
            )  # mutable-ok: prometheus_client label tuples
            cls.redactions = Counter(  # mutable-ok: prometheus_client label tuples
                "dlp_redactions_total", "Payloads redacted or masked by a DLP policy", labelnames=_ACTION_LABELS
            )
            cls.provider_latency = Histogram(
                "dlp_provider_latency_ms", "Delegated evaluation latency in ms", labelnames=_PROVIDER_LABELS
            )
            cls.provider_errors = Counter(  # mutable-ok: prometheus_client label tuples
                "dlp_provider_errors_total", "Delegated evaluation failures", labelnames=_PROVIDER_LABELS
            )
            cls.fail_open = Counter(  # mutable-ok: prometheus_client label tuples
                "dlp_fail_open_total", "Evaluations that failed open", labelnames=_CONNECTION_LABELS
            )
            cls.fail_closed = Counter(  # mutable-ok: prometheus_client label tuples
                "dlp_fail_closed_total", "Evaluations that failed closed", labelnames=_CONNECTION_LABELS
            )
            cls.policy_sync_errors = Counter(  # mutable-ok: prometheus_client label tuples
                "dlp_policy_sync_errors", "Policy sync failures", labelnames=_CONNECTION_LABELS
            )
            cls.policy_drift = Counter(  # mutable-ok: prometheus_client label tuples
                "dlp_policy_drift_total", "Upstream policy changes detected", labelnames=_CONNECTION_LABELS
            )
            cls.shadow_would_block = Counter(  # mutable-ok: prometheus_client label tuples
                "dlp_shadow_would_block_total",
                "Requests a shadow policy would have blocked had it been active",
                labelnames=_ACTION_LABELS,
            )
        except Exception as err:  # noqa: BLE001  # metrics registration must never break enforcement
            verbose_proxy_logger.warning("WIT OS DLP: could not register Prometheus metrics: %s", err)

    @classmethod
    def record_decision(
        cls,
        guardrail_name: str,
        policy_id: str,
        decision: str,
        direction: str,
        shadow: bool,
    ) -> None:
        cls._ensure_initialized()
        labels: Final = (guardrail_name, policy_id, decision, direction, str(shadow).lower())
        _increment(cls.evaluations, labels)
        if decision == "BLOCK":
            _increment(cls.shadow_would_block if shadow else cls.blocks, labels)
        elif decision in ("REDACT", "MASK") and not shadow:
            _increment(cls.redactions, labels)

    @classmethod
    def record_fail_mode(cls, connection_id: str, failed_closed: bool) -> None:
        cls._ensure_initialized()
        _increment(cls.fail_closed if failed_closed else cls.fail_open, (connection_id,))

    @classmethod
    def record_provider_error(cls, provider: str, connection_id: str) -> None:
        cls._ensure_initialized()
        _increment(cls.provider_errors, (provider, connection_id))

    @classmethod
    def record_provider_latency(cls, provider: str, connection_id: str, latency_ms: float) -> None:
        cls._ensure_initialized()
        _observe(cls.provider_latency, (provider, connection_id), latency_ms)

    @classmethod
    def record_sync_error(cls, connection_id: str) -> None:
        cls._ensure_initialized()
        _increment(cls.policy_sync_errors, (connection_id,))

    @classmethod
    def record_drift(cls, connection_id: str) -> None:
        cls._ensure_initialized()
        _increment(cls.policy_drift, (connection_id,))


def _increment(metric: object | None, labels: tuple[str, ...]) -> None:
    if metric is None:
        return
    labeller: Final = getattr(metric, "labels", None)
    if labeller is None:
        return
    labelled: Final = labeller(*labels)
    increment: Final = getattr(labelled, "inc", None)
    if increment is not None:
        increment()


def _observe(metric: object | None, labels: tuple[str, ...], value: float) -> None:
    if metric is None:
        return
    labeller: Final = getattr(metric, "labels", None)
    if labeller is None:
        return
    labelled: Final = labeller(*labels)
    observer: Final = getattr(labelled, "observe", None)
    if observer is not None:
        observer(value)
