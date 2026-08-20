# WIT OS FinOps — forecast engine, runway and API

Phase 2 of `BLUEPRINT_v2.md` (§1.3, §1.6–§1.11, §1.14). Phase 1 built the fact
tables; this turns them into a forecast, a runway date, a driver decomposition
and an API.

## What runs, and when

| Job | Cadence | What it does |
|---|---|---|
| `ForecastJob` | nightly | Per scope: load 365 days of daily facts, compete the model roster, simulate 500 paths, price them, compute runway against every quota, decompose the week, persist, evaluate alert rules. |
| `AnomalyDetector` | 15 min | EWMA control chart over hourly facts; two consecutive buckets past 4σ raise an event, as does a new model group billing over $5/hour. |
| `QualityJob` | nightly | Scores yesterday's actual against the one-day-ahead prediction made for it, and republishes a rolling 30-day WAPE and bias per scope. |

Nothing here runs on the inference request path.

## The refusals

The engine is designed to decline rather than to decorate.

- **Under 3 days of history**: no forecast at all. `/forecast` returns
  `state: insufficient_history` with a reason, never a band of zeros.
- **Idle 30 days**: `state: dormant`, and any previous forecast is retired so a
  stale projection cannot keep sitting on a dashboard.
- **Price book cannot reproduce the invoice**: the scope falls back to
  `ACTUAL_COST_BASED` (forecasting the cost series directly) rather than
  publishing priced usage that disagrees with the bill. The ratio is reported as
  `pricing_fidelity`.
- **No residuals to resample**: confidence drops to `low` and `has_band` is
  `false`, so P10 and P90 collapse onto P50 visibly rather than silently.

## The ladder (§1.6)

| History | Tier | Roster |
|---|---|---|
| < 3 days | `insufficient_history` | none |
| 3–6 days | `burn_rate` | burn rate only, confidence `low` |
| 7–27 days | `trend_weekday` | burn rate, seasonal naive, EWMA, robust linear, damped Holt |
| 28–89 days | `seasonal_trend` | adds the two weekday-seasonal models and weekly ETS |
| 90+ days | `full_ensemble` | the whole roster plus the inverse-WAPE ensemble |

## API

All routes are under `/witos/finops`, all reads come from persisted rows, and all
apply two checks: the key's `witos_scopes` permission and the entity clamp copied
from the fork's spend endpoints.

| Method | Path | Permission |
|---|---|---|
| GET | `/overview` | `finops:view` |
| GET | `/timeseries` | `finops:view` |
| GET | `/forecast` | `finops:view` |
| GET | `/runway` | `finops:view` |
| GET | `/drivers` | `finops:view` |
| GET | `/model-mix` | `finops:view` |
| GET | `/forecast-quality` | `finops:view` |
| GET/POST/PATCH/DELETE | `/quotas` | `finops:view` / `finops:manage_budget` |
| GET/POST/DELETE | `/scenarios`, POST `/scenarios/evaluate` | `finops:run_scenarios` |
| GET/POST/PATCH | `/alerts`, GET `/alert-events`, PATCH `/alert-events/{id}` | `finops:view` / `finops:manage_alerts` |
| GET | `/report?format=csv` | `finops:view` |
| GET | `/stream` | `finops:view` |
| POST | `/admin/rebuild`, GET `/admin/status` | `finops:manage` |

Quotas mirrored from LiteLLM budgets (`source_type = litellm_budget`) are
read-only here and return 409 on write: edit the budget in LiteLLM and the next
mirror run carries it across.

## CFO export columns

Fixed and append-only. Downstream spreadsheets have formulas against these
positions, so new columns go on the end and existing ones never move or change
meaning.

```
scope_type, scope_id, period_end, actual_mtd_usd, projected_eom_p50_usd,
projected_eom_p90_usd, budget_usd, projected_variance_usd,
projected_variance_pct, budget_consumed_pct, exhaustion_p50, exhaustion_p90,
prob_overrun_this_period, survives_cycle, forecast_algorithm, forecast_regime,
forecast_wape, forecast_quality_score, generated_at, pricing_snapshot_hash
```

## Webhooks

Alert channels of `type: webhook` are signed `HMAC-SHA256` over
`<timestamp>.<body>` and carry `X-WITOS-Signature: sha256=<hex>`,
`X-WITOS-Timestamp` and `X-WITOS-Event-Id`. The secret is named by the channel
and read from `WITOS_WEBHOOK_SECRET_<NAME>` at delivery time; it is never stored
in the database. A channel naming a secret that is not set is refused rather than
sent unsigned.

## Optional dependency

`statsmodels` is optional. Where it is absent the two exponential-smoothing
candidates report themselves unavailable and do not compete; every other model is
stdlib arithmetic. The winning algorithm is recorded on each forecast row, so
which roster ran is always visible.

## Prometheus

`finops_forecast_generation_seconds`, `finops_forecast_scopes_total`,
`finops_forecast_wape`, `finops_forecast_bias`, `finops_alert_events_total`,
alongside Phase 1's aggregation instruments.
