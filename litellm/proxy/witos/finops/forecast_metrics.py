"""Prometheus instruments for the forecast side of FinOps (blueprint §1.14).

Separate from Phase 1's ``metrics.py`` only so the two work packages do not edit
one file; the registration pattern is identical, lazily registered and every
recorder a no-op when ``prometheus_client`` is absent, so a metrics problem can
never fail a forecast run.

``finops_forecast_wape`` is the instrument to alert on. Aggregation lag says the
facts are stale; this says the forecast built on them stopped being true, which
is the failure nobody notices from the outside.
"""

from typing import TYPE_CHECKING, Final, Literal, TypeAlias

from litellm._logging import verbose_proxy_logger

if TYPE_CHECKING:
    from prometheus_client import Counter as PrometheusCounter
    from prometheus_client import Gauge as PrometheusGauge
    from prometheus_client import Histogram as PrometheusHistogram

ForecastOutcomeLabel: TypeAlias = Literal[
    "forecast", "insufficient_history", "dormant", "no_history", "failed", "skipped_locked", "stopped_early"
]

_SCOPE_LABEL: Final = ("scope_type",)
_OUTCOME_LABEL: Final = ("outcome",)
_DURATION_BUCKETS: Final = (1.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0)


class ForecastMetrics:
    """Lazily-registered instruments for the forecast, runway and anomaly jobs."""

    _initialized: bool = False
    generation_seconds: "PrometheusHistogram | None" = None
    scopes: "PrometheusCounter | None" = None
    wape: "PrometheusGauge | None" = None
    bias: "PrometheusGauge | None" = None
    alerts: "PrometheusCounter | None" = None

    @classmethod
    def _ensure_initialized(cls) -> None:
        if cls._initialized:
            return
        cls._initialized = True
        try:
            from prometheus_client import Counter as PromCounter
            from prometheus_client import Gauge as PromGauge
            from prometheus_client import Histogram as PromHistogram

            cls.generation_seconds = PromHistogram(
                "finops_forecast_generation_seconds",
                "Wall-clock duration of one FinOps forecast job run",
                buckets=_DURATION_BUCKETS,
            )
            cls.scopes = PromCounter(
                "finops_forecast_scopes_total",
                "Scopes the forecast job processed, labelled by what it produced",
                labelnames=_OUTCOME_LABEL,
            )
            cls.wape = PromGauge(
                "finops_forecast_wape",
                "Backtest WAPE of the most recent forecast run, per scope type",
                labelnames=_SCOPE_LABEL,
            )
            cls.bias = PromGauge(
                "finops_forecast_bias",
                "Backtest bias of the most recent forecast run, per scope type",
                labelnames=_SCOPE_LABEL,
            )
            cls.alerts = PromCounter(
                "finops_alert_events_total",
                "FinOps alert events raised, labelled by type",
                labelnames=("alert_type",),
            )
        except Exception as e:  # noqa: BLE001  # a metrics problem must never fail a forecast run
            verbose_proxy_logger.warning("Could not register WIT OS FinOps forecast metrics: %s", e)

    @classmethod
    def record_run(cls, duration_seconds: float) -> None:
        cls._ensure_initialized()
        if cls.generation_seconds is not None:
            cls.generation_seconds.observe(duration_seconds)

    @classmethod
    def record_scope(cls, outcome: ForecastOutcomeLabel) -> None:
        cls._ensure_initialized()
        if cls.scopes is not None:
            cls.scopes.labels(outcome=outcome).inc()

    @classmethod
    def record_accuracy(cls, scope_type: str, *, wape: float | None, bias: float | None) -> None:
        cls._ensure_initialized()
        if cls.wape is not None and wape is not None:
            cls.wape.labels(scope_type=scope_type).set(wape)
        if cls.bias is not None and bias is not None:
            cls.bias.labels(scope_type=scope_type).set(bias)

    @classmethod
    def record_alert(cls, alert_type: str) -> None:
        cls._ensure_initialized()
        if cls.alerts is not None:
            cls.alerts.labels(alert_type=alert_type).inc()
