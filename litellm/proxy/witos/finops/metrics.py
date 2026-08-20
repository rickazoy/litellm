"""Prometheus instruments for FinOps aggregation (blueprint §1.14).

Registration is deferred to first use and every recorder degrades to a no-op, so
importing this module never touches the Prometheus registry and a missing
``prometheus_client`` extra can never fail an aggregation run. This mirrors
``SpendLogCleanupMetrics``, the fork's existing background-job metrics pattern.

``finops_aggregation_lag_seconds`` is the one an operator pages on: it is how far
behind live traffic the fact tables are, and every forecast built on top of them
inherits that staleness.
"""

from typing import TYPE_CHECKING, Final, Literal, TypeAlias

from litellm._logging import verbose_proxy_logger

if TYPE_CHECKING:
    from prometheus_client import Counter as PrometheusCounter
    from prometheus_client import Gauge as PrometheusGauge
    from prometheus_client import Histogram as PrometheusHistogram

CycleOutcomeLabel: TypeAlias = Literal["completed", "skipped_locked", "skipped_disabled", "failed"]

_GRAIN_LABEL: Final = ("grain",)
_OUTCOME_LABEL: Final = ("outcome",)
_DURATION_BUCKETS: Final = (0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0)


class FinOpsMetrics:
    """Lazily-registered instruments for the FinOps aggregation job."""

    _initialized: bool = False
    aggregation_lag: "PrometheusGauge | None" = None
    last_success: "PrometheusGauge | None" = None
    cycle_duration: "PrometheusHistogram | None" = None
    cycles: "PrometheusCounter | None" = None
    rows_written: "PrometheusCounter | None" = None
    missing_price_requests: "PrometheusCounter | None" = None
    unknown_model_requests: "PrometheusCounter | None" = None

    @classmethod
    def _ensure_initialized(cls) -> None:
        if cls._initialized:
            return
        cls._initialized = True
        try:
            from prometheus_client import Counter as PromCounter
            from prometheus_client import Gauge as PromGauge
            from prometheus_client import Histogram as PromHistogram

            cls.aggregation_lag = PromGauge(
                "finops_aggregation_lag_seconds",
                "Seconds between now and the newest fully aggregated bucket",
            )
            cls.last_success = PromGauge(
                "finops_last_success_timestamp",
                "Unix timestamp of the last aggregation cycle that completed",
            )
            cls.cycle_duration = PromHistogram(
                "finops_aggregation_duration_seconds",
                "Wall-clock duration of one FinOps aggregation cycle",
                buckets=_DURATION_BUCKETS,
            )
            cls.cycles = PromCounter(
                "finops_aggregation_cycles_total",
                "FinOps aggregation cycles, labelled by how the cycle ended",
                labelnames=_OUTCOME_LABEL,
            )
            cls.rows_written = PromCounter(
                "finops_aggregation_rows_written_total",
                "Fact rows inserted or recomputed by the FinOps aggregator",
                labelnames=_GRAIN_LABEL,
            )
            cls.missing_price_requests = PromCounter(
                "finops_missing_price_requests",
                "Aggregated requests that reported tokens but zero spend, so no price was applied",
            )
            cls.unknown_model_requests = PromCounter(
                "finops_unknown_model_requests",
                "Aggregated requests whose model is absent from litellm's cost map",
            )
        except Exception as e:  # noqa: BLE001 - a metrics problem must never fail an aggregation run
            verbose_proxy_logger.warning("Could not register WIT OS FinOps metrics: %s", e)

    @classmethod
    def set_aggregation_lag(cls, seconds: float) -> None:
        cls._ensure_initialized()
        if cls.aggregation_lag is not None:
            cls.aggregation_lag.set(seconds)

    @classmethod
    def set_last_success(cls, unix_timestamp: float) -> None:
        cls._ensure_initialized()
        if cls.last_success is not None:
            cls.last_success.set(unix_timestamp)

    @classmethod
    def record_cycle(cls, outcome: CycleOutcomeLabel, duration_seconds: float | None = None) -> None:
        cls._ensure_initialized()
        if cls.cycles is not None:
            cls.cycles.labels(outcome=outcome).inc()
        if duration_seconds is not None and cls.cycle_duration is not None:
            cls.cycle_duration.observe(duration_seconds)

    @classmethod
    def record_rows(cls, grain: str, rows: int) -> None:
        cls._ensure_initialized()
        if cls.rows_written is not None and rows > 0:
            cls.rows_written.labels(grain=grain).inc(rows)

    @classmethod
    def record_pricing_coverage(cls, missing_price: int, unknown_model: int) -> None:
        cls._ensure_initialized()
        if cls.missing_price_requests is not None and missing_price > 0:
            cls.missing_price_requests.inc(missing_price)
        if cls.unknown_model_requests is not None and unknown_model > 0:
            cls.unknown_model_requests.inc(unknown_model)
