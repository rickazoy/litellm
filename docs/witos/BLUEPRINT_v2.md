# WIT OS × LiteLLM Fork — Engineering Blueprint v2.0 (MERGED / AUTHORITATIVE)
## Feature 1: AI FinOps Intelligence (predictive cost, usage & runway)
## Feature 2: DLP Policy Federation (Cyera, Purview, Nightfall, Netskope, custom)

**Version:** 2.0 · **Supersedes:** v1.0 · **Owner:** Rick Azoy, CAIO/CISO, WITONE
**Purpose:** Build-ready specification to feed directly into Claude Code. This version merges the v1.0 blueprint (LiteLLM internals grounding, concrete schemas, security controls) with the strongest elements of the second-opinion review: usage-based forecasting with driver decomposition, probabilistic runway via Monte Carlo, a generic quota model, four-mode DLP federation with capability discovery, shadow-first deployment, streaming honesty modes, and tool/RAG-context DLP.

---

# 0. GROUND RULES (read first, apply everywhere)

## 0.1 What exists in LiteLLM — reuse, never duplicate

- **ORM/DB:** Prisma + PostgreSQL, schema at `litellm/proxy/schema.prisma`, tables prefixed `LiteLLM_`.
- **Entity hierarchy:** Organization → Team → (Project) → Key, plus Users/End-Users. Budgets attach at every level via shared `LiteLLM_BudgetTable` (`max_budget`, `soft_budget`, `budget_duration`, `budget_reset_at`, TPM/RPM).
- **Spend capture:** per-request `LiteLLM_SpendLogs` (indexed `api_key`, `user`, `team_id`, `startTime`), written **asynchronously** by `DBSpendUpdateWriter` — late-arriving rows are normal and must be handled (see §1.5 watermark). Retention cleanup (`spend_log_cleanup.py`) purges old SpendLogs — **therefore all forecasting depends on our aggregate tables, never raw SpendLogs at read time.**
- **Budget enforcement:** pre-call checks + Redis/in-memory budget-reservation pattern.
- **Pricing:** LiteLLM's cost calculator + `litellm.model_cost` map is the ONLY pricing source. **Do not create a competing price catalog.**
- **Spend API pattern:** `spend_management_endpoints.py` — copy its role-based scope filtering.
- **Admin UI:** Next.js/React at `ui/litellm-dashboard/` (Tremor/AntD, TanStack Query, generated OpenAPI types).
- **Guardrails:** `config.yaml` `guardrails:` list, hooks `pre_call`/`during_call`/`post_call`, `CustomGuardrail` base class. DLP federation compiles/executes at this layer.
- **Background jobs:** follow existing loop patterns; multi-pod safety required.

## 0.2 Mandatory Claude Code preflight (before writing any code)

1. Inspect THIS fork (not upstream assumptions) and produce a repo-specific implementation map identifying: current `schema.prisma`, SpendLogs model/repository, budget models, spend endpoints, cost calculator, dashboard routing + API client patterns (`networking.ts`), RBAC implementation, guardrail registry + `CustomGuardrail` base, background-job patterns, Redis abstraction, secret-manager abstraction.
2. **LICENSING BOUNDARY:** Do not copy, port, depend on, or reproduce code from `litellm/enterprise` (or any enterprise-licensed tree) unless explicitly authorized. Implement everything on the MIT-side using public proxy/models/dashboard patterns.
3. Do not modify the inference hot path for FinOps. Zero forecasting code on the request path.
4. Implement database → domain → API contracts first; connect UI last. No UI-first mocks.
5. At each stage: run existing LiteLLM tests, lint/format, UI tests; preserve backward compatibility; document every migration.

## 0.3 Fork layout & merge hygiene

All new code namespaced to keep upstream merges clean — only additive files + registrations:

```
litellm/proxy/witos/
├── shared/        # scheduler (job lease), alert_dispatcher, webhooks (HMAC), rbac, metrics
├── finops/        # aggregator, forecasting, quotas, runway, drivers, scenarios, router
└── policy_fabric/ # canonical model, connectors, sync, compiler, runtime guardrail, router
ui/litellm-dashboard/src/components/witos/{finops/, dlp/}
witos-sdk/         # TS client + hooks + embeddable components + EN/ES i18n
tests/witos/       # unit, integration, contract, backtest-regression
docs/witos/        # finops.md, wit_dps_spec.md, connector_dev_guide.md, security.md
```

## 0.4 Shared-control-plane rule

LiteLLM fork = system of record and enforcement point (it sees every transaction). WIT OS = executive/security experience consuming `/witos/finops/*` and `/witos/dlp/*` via `witos-sdk`. **WIT OS must never compute a different number.** Same forecast ID, same figures, different visualization. WIT OS backend holds scoped virtual keys; WIT OS frontend never talks to the proxy directly (WIT OS API gateway proxies with tenant→org/team mapping).

## 0.5 RBAC (both features)

New permissions enforced via key metadata scopes, checked in every router:

```
finops:view  finops:manage  finops:manage_budget  finops:manage_alerts  finops:run_scenarios
dlp:view  dlp:manage_connections  dlp:import  dlp:approve  dlp:enforce  dlp:view_decisions  dlp:test
```
A CFO key gets `finops:view + run_scenarios + manage_alerts` with no DLP or model-management access. A SecOps key gets `dlp:*` without financial visibility. Team-scoped keys see only their entities (reuse spend-endpoint filtering pattern).

---

# PART I — AI FINOPS INTELLIGENCE

## 1.1 Product definition — the questions it must answer

Not a token chart. The system answers: What will we spend this month / next month (P50/P90)? What changed since yesterday's forecast, and **why** (volume vs model mix vs prompt size vs caching)? When does each team/key/customer exhaust budget or token quota, with what probability? Which business unit exceeds budget first? What happens if adoption grows 20%, if 30% of expensive-model traffic migrates down, if cache hit rate rises? How accurate have our forecasts been?

## 1.2 Architecture

```
App ─▶ LiteLLM Gateway ─▶ Provider          (request path: UNTOUCHED)
              │
              ▼
      LiteLLM_SpendLogs ──(async writer)──▶ daily spend tables
              │
   incremental aggregation ~5 min (watermark + lateness window)
              ▼
 ┌─────────────────────────────────────────────┐
 │ FinOps Aggregation  (hourly+daily facts,    │
 │ model mix, quota state)                     │
 ├─────────────────────────────────────────────┤
 │ Forecast Engine (usage-based: tokens/req/   │
 │ mix → priced via litellm cost calc; model   │
 │ competition + backtest; Monte Carlo bands;  │
 │ regime detection; drivers; scenarios)       │
 ├─────────────────────────────────────────────┤
 │ FinOps API  /witos/finops/*  + SSE + webhooks│
 └───────────┬───────────────────┬─────────────┘
       LiteLLM Admin UI      WIT OS (LEDGER → Cost Intelligence)
       (operator view)       (executive view, via witos-sdk)
```

## 1.3 Core forecasting principle (v2 change): forecast USAGE, price it AFTERWARD

For token-priced workloads (`forecast_strategy = TOKEN_BASED`), forecast these series **separately per scope**:

```
prompt_tokens · completion_tokens · request_count · model_mix (share per model_group) · cache_read_tokens
```
Then compute projected cost deterministically:
```
future_cost ≈ Σ_models ( pred_input_tokens_m × input_price_m
                       + pred_output_tokens_m × output_price_m
                       + cache costs + provider-specific usage costs )
```
using LiteLLM's cost calculator with a **pricing snapshot hash** stored per forecast (so price-map changes are detectable and forecasts reproducible).

For non-token workloads (speech, transcription, image, video, search, rerank, provider services): `forecast_strategy = ACTUAL_COST_BASED` — forecast the relevant usage units and the historical cost relationship; never force tokens.

**Why:** separates "usage grew" from "mix got more expensive," enables driver decomposition (§1.10), and makes scenarios (§1.11) exact repricings instead of approximations.

## 1.4 Data model (additive Prisma models)

```prisma
// ---------- Fact tables ----------
model WITOS_FinOpsUsageHourly {
  id                 String   @id @default(uuid())
  bucket_start_utc   DateTime
  scope_type         String   // global|organization|team|user|key|end_user|model|model_group|tag
  scope_id           String
  scope_label        String?
  organization_id    String?
  team_id            String?
  model              String?
  model_group        String?
  provider           String?
  call_type          String?
  request_count      Int
  successful_requests Int
  failed_requests    Int
  prompt_tokens      BigInt
  completion_tokens  BigInt
  total_tokens       BigInt
  cache_read_tokens  BigInt   @default(0)
  cache_creation_tokens BigInt @default(0)
  spend_usd          Decimal  @db.Decimal(18, 8)
  avg_latency_ms     Int?
  p95_latency_ms     Int?
  breakdown          Json?    // {model_group: {tokens, spend, requests}} for drill-down
  created_at         DateTime @default(now())
  updated_at         DateTime @updatedAt
  @@unique([bucket_start_utc, scope_type, scope_id])
  @@index([scope_type, scope_id, bucket_start_utc])
}

model WITOS_FinOpsUsageDaily {
  // identical logical schema at daily grain (bucket_date instead of bucket_start_utc)
  // daily = 30–365d forecasting source; hourly = intraday burn + anomaly detection
  id String @id @default(uuid())
  bucket_date        DateTime
  scope_type         String
  scope_id           String
  scope_label        String?
  organization_id    String?
  team_id            String?
  model_group        String?
  provider           String?
  request_count      Int
  successful_requests Int
  failed_requests    Int
  prompt_tokens      BigInt
  completion_tokens  BigInt
  total_tokens       BigInt
  cache_read_tokens  BigInt   @default(0)
  cache_creation_tokens BigInt @default(0)
  spend_usd          Decimal  @db.Decimal(18, 8)
  breakdown          Json?
  created_at         DateTime @default(now())
  updated_at         DateTime @updatedAt
  @@unique([bucket_date, scope_type, scope_id])
  @@index([scope_type, scope_id, bucket_date])
}
```
**Tag cardinality rule:** never aggregate every tag combination. Maintain tag-scope aggregates ONLY for admin-approved FinOps dimensions (`environment, application, cost_center, business_unit, project, customer`) configured in `WITOS_FinOpsConfig.approved_tag_dimensions`.

```prisma
// ---------- Quotas: entitlements, NOT rate limits ----------
// LiteLLM TPM is a rate limit — it does not mean "100M tokens remaining."
// This table represents true entitlements so runway statements are honest.
model WITOS_FinOpsQuota {
  quota_id      String   @id @default(uuid())
  scope_type    String
  scope_id      String
  metric        String   // usd | total_tokens | prompt_tokens | completion_tokens | requests
  limit_value   Decimal  @db.Decimal(24, 4)
  period        String   // day|week|month|quarter|year|contract|custom
  period_start  DateTime
  period_end    DateTime?
  reset_at      DateTime?
  source_type   String   // litellm_budget | manual | provider_contract | customer_contract | imported
  source_id     String?  // e.g. LiteLLM_BudgetTable.budget_id when mirrored
  soft_threshold_pct Decimal? @db.Decimal(5,2)
  created_at    DateTime @default(now())
  updated_at    DateTime @updatedAt
  @@index([scope_type, scope_id])
}
```
A nightly mirror job auto-materializes quotas from `LiteLLM_BudgetTable` (`source_type=litellm_budget`, kept in sync, read-only in UI); manual/contract quotas are user-managed. All runway math runs against this table only.

```prisma
// ---------- Persisted forecasts ----------
model WITOS_FinOpsForecast {
  forecast_id     String   @id @default(uuid())
  scope_type      String
  scope_id        String
  generated_at    DateTime @default(now())
  forecast_start  DateTime
  forecast_end    DateTime
  metric          String   // spend_usd | prompt_tokens | completion_tokens | total_tokens | requests
  strategy        String   // TOKEN_BASED | ACTUAL_COST_BASED
  algorithm       String
  algorithm_version String
  history_days    Int
  training_points Int
  trend_regime    String   // stable|accelerating|decelerating|step_change|volatile|insufficient_history
  wape            Decimal? @db.Decimal(8,4)
  bias            Decimal? @db.Decimal(8,4)
  quality_score   Int?     // 0-100 composite
  pricing_snapshot_hash String?
  // per-day arrays: {date, p10, p50, p90} plus model-mix vector for TOKEN_BASED:
  forecast_json   Json
  // denormalized headlines for fast dashboards:
  proj_eom        Decimal? @db.Decimal(18,4)
  proj_eoq        Decimal? @db.Decimal(18,4)
  burn_rate_daily Decimal? @db.Decimal(18,4)
  is_latest       Boolean  @default(true)
  @@index([scope_type, scope_id, metric, is_latest])
  @@index([generated_at])
}

// ---------- Runway results (per quota) ----------
model WITOS_FinOpsRunway {
  id                String   @id @default(uuid())
  computed_at       DateTime @default(now())
  quota_id          String
  scope_type        String
  scope_id          String
  consumed          Decimal  @db.Decimal(24,4)
  remaining         Decimal  @db.Decimal(24,4)
  exhaustion_p50    DateTime?
  exhaustion_p90    DateTime?   // "as early as"
  prob_exhaust_before_reset Decimal? @db.Decimal(5,4)  // e.g. 0.78
  prob_overrun_this_period  Decimal? @db.Decimal(5,4)
  survives_cycle    Boolean
  is_latest         Boolean  @default(true)
  @@index([quota_id, is_latest])
  @@index([scope_type, scope_id, is_latest])
}

// ---------- Scenarios ----------
model WITOS_FinOpsScenario {
  scenario_id  String   @id @default(uuid())
  name         String
  description  String?
  scope_type   String
  scope_id     String
  assumptions_json Json  // see §1.11
  created_by   String
  created_at   DateTime @default(now())
  updated_at   DateTime @updatedAt
}

// ---------- Alerts ----------
model WITOS_FinOpsAlertRule {
  alert_id     String  @id @default(uuid())
  name         String
  scope_type   String
  scope_id     String            // "*" for all
  condition_type String          // forecast_exceeds_budget | budget_exhaustion_within |
                                 // token_exhaustion_within | spend_growth_above | usage_growth_above |
                                 // forecast_changed_above | anomaly_detected | forecast_quality_below
  threshold    Decimal @db.Decimal(12,4)
  lookahead_days Int?
  channels     Json              // [{type:slack|webhook|email, ...}] webhook ⇒ HMAC-signed
  cooldown_min Int     @default(240)
  enabled      Boolean @default(true)
  last_triggered_at DateTime?
}

model WITOS_FinOpsAlertEvent {
  id          String   @id @default(uuid())
  created_at  DateTime @default(now())
  alert_id    String?
  alert_type  String
  severity    String   // info|warning|critical
  scope_type  String
  scope_id    String
  title       String
  body        Json     // structured: observed, expected, z, probability, runway dates, drivers
  status      String   @default("open")  // open|acked|resolved|muted
  acked_by    String?
  notified_via Json?
  @@index([status, created_at])
}

// ---------- Multi-pod job lease ----------
model WITOS_BackgroundJobLease {
  job_name    String   @id
  owner_id    String
  lease_until DateTime
  heartbeat_at DateTime
}
```

## 1.5 Aggregation engine (`finops/aggregator.py`)

- Runs every ~5 min (configurable). **Watermark + lateness window** (async spend writer means late rows are real):
```
effective_start = watermark - lateness_window   # default 2h overlap
query raw rows ≥ effective_start → determine affected hourly buckets
→ recompute those buckets from source (idempotent) → UPSERT → advance watermark
```
- Daily table rolled from hourly at day close + corrected on late arrivals (same overlap logic).
- Hour-granularity source: existing daily spend aggregate tables where sufficient; SpendLogs for the trailing window before retention purge. Pure SQL aggregation — never Python row loops.
- **Multi-pod safety:** every job acquires `WITOS_BackgroundJobLease` (compare-and-set with `lease_until`, heartbeat renewal); skip-if-held. Use pg advisory lock as the CAS primitive if simpler in this repo.
- Backfill CLI: `python -m litellm.proxy.witos.finops.backfill --days 90`.

## 1.6 Forecast engine (`finops/forecasting.py`, `forecast_models.py`)

No LLMs, no heavy ML in the core loop — hierarchical classical time series. Deps: `numpy`, `pandas`, `statsmodels`. No Prophet in v1 (cmdstan bloat); `ForecastModel` interface allows adding later.

**Candidate roster (model competition):**
```
SeasonalNaive · EWMA · RobustLinear (Theil–Sen) · DayOfWeekSeasonalTrend
RecentWeightedSeasonalTrend · Holt (damped) · HoltWintersETS (weekly, damped) · Ensemble (inverse-WAPE weighted)
```

**Selection — rolling-origin backtest:** train 1–30 → predict 31–37; train 1–37 → predict 38–44; walk forward. Score **WAPE** (primary — robust to zero-volume days, unlike MAPE), MAE, and **bias**. Best model or weighted ensemble wins; ties → simpler model.

**Short-history ladder (deterministic, honest):**
```
<3d   insufficient_history — no forecast, explicit UI state
3–7d  burn-rate only, confidence=LOW, banner: "Only N days of history"
7–28d trend + weekday adjustment
28–90d seasonal + trend ensemble
90d+  full ensemble + robust seasonality
```
Never display fake precision.

**Regime-change detection:** compare `recent_7d_avg` vs `previous_7d_avg` vs `recent_28d_avg` using robust z-score (MAD) / CUSUM. On regime change (`accelerating|decelerating|step_change`), down-weight pre-change history and tag `trend_regime` on the forecast. AI workloads step-change constantly — this is not optional.

**Confidence intervals — residual-bootstrap Monte Carlo:** for each future day, `simulated = point_forecast + sampled_historical_residual`; run ~500 paths; report P10/P50/P90 per day and cumulative. These same paths power runway probabilities (§1.7) for free.

**Zero/gap handling:** missing day = 0 (valid signal). Entities inactive 30+ days → `dormant`, skipped.

**Forecast quality telemetry:** nightly, score yesterday's actual vs prior 1-day-ahead prediction → rolling 30d WAPE + bias per scope → stored, exposed via API and footer UI. Backtest-regression test suite: committed synthetic-series baselines; CI fails if WAPE regresses > 2 pts.

## 1.7 Runway & exhaustion (`finops/runway.py`)

For every quota row:
```
remaining = limit_value − consumed_in_current_period
for each Monte Carlo path: cumulative += simulated_daily_usage; record first date ≥ remaining
→ exhaustion_p50, exhaustion_p90 ("as early as"),
  prob_exhaust_before_reset, prob_overrun_this_period, survives_cycle
```
Cycle-aware: respects `reset_at` / `budget_reset_at`; if reset lands first → `survives_cycle=true` + projected % consumed at reset. Runs for USD quotas AND token quotas. Output statement style (the CFO sentence): *"78% probability of exceeding the $100,000 monthly budget. Median exhaustion Aug 27; conservative (P90) Aug 24."*

## 1.8 Anomaly detection

Every 15 min on hourly facts: EWMA (α=0.3) + EW variance per scope on spend & requests; flag `z ≥ 4` for 2 consecutive buckets → `anomaly_detected` event with top contributing model/key. Separate **new-model-cost** detector: first appearance of a model in a scope with hourly spend > $5 (configurable) → `info` alert.

## 1.9 Driver decomposition (`finops/drivers.py`) — deterministic, never LLM-invented

For any scope + comparison window, decompose forecast/spend delta into additive contributions:
```
Δspend = f(Δrequest_volume, Δmodel_mix_share, Δavg_prompt_tokens, Δavg_completion_tokens,
           Δcache_hit_rate, Δprice_map, residual)
```
Method: hold-one-factor-constant counterfactual repricing over the fact tables (compute spend with factor X frozen at baseline, attribute the difference; residual line keeps it honest). Output:
```
Forecast +17.3% vs last week:
  +9.1%  request volume
  +5.7%  expensive-model share
  +4.2%  average prompt size
  −1.7%  caching improvement
  +0.0%  price changes    (residual −0.0%)
```
An LLM MAY optionally render these structured drivers as CFO prose; the math is the source of truth.

## 1.10 Scenario engine (`finops/scenarios.py`)

Assumptions schema (stored in `WITOS_FinOpsScenario.assumptions_json`, also accepted stateless):
```json
{
  "usage_growth_pct": 15,
  "new_users": 150,
  "model_migrations": [{"from": "claude-opus-4-8", "to": "claude-sonnet-4-6", "traffic_pct": 30}],
  "cache_hit_rate": 0.35,
  "prompt_token_delta_pct": -10,
  "pricing_adjustments": {"gpt-5": {"input_delta_pct": -20, "output_delta_pct": -20}},
  "new_workloads": [{"daily_spend_usd": 40, "start_date": "2026-09-01"}]
}
```
Evaluation = replay forecast token/mix series with transforms applied, reprice via cost calculator (exact, because forecasting is usage-based), re-run Monte Carlo. Output: baseline vs scenario chart + delta table (EOM, savings $, savings %, new runway dates). Presets: baseline / optimistic / expected / high-growth / custom. Deterministic given `pricing_snapshot_hash` (reproducibility test required).

## 1.11 API contract (`/witos/finops/*`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/witos/finops/overview` | Fleet headline: MTD, proj EOM (P50/P90), top runway risks, open alerts. |
| GET | `/witos/finops/timeseries?scope_type&scope_id&metric&grain` | Fact-table series for charts. |
| GET | `/witos/finops/forecast?scope_type&scope_id&horizon_days=90` | Latest forecast: actuals + P10/P50/P90 daily + headlines + quality + drivers. |
| GET | `/witos/finops/runway?scope_type&scope_id` | Per-quota runway incl. probabilities. |
| GET | `/witos/finops/drivers?scope_type&scope_id&window=7d` | Decomposition table. |
| GET | `/witos/finops/model-mix?scope_type&scope_id` | Mix history + forecast shares. |
| GET | `/witos/finops/forecast-quality?scope_type&scope_id` | WAPE/bias history, regime, quality score. |
| GET/POST/PATCH/DELETE | `/witos/finops/quotas` | Quota CRUD (litellm_budget rows read-only). |
| GET/POST/DELETE | `/witos/finops/scenarios` · POST `/scenarios/evaluate` | Saved + stateless what-if. |
| GET/POST/PATCH | `/witos/finops/alerts` (rules) · GET `/witos/finops/alert-events` · PATCH `/alert-events/{id}` | Alerting. |
| GET | `/witos/finops/report?format=csv&period=month` | CFO export: per-team actual MTD, proj EOM P50/P90, variance vs budget, runway, overrun probability. Fixed documented columns. |
| GET | `/witos/finops/stream` | SSE: new alert events + forecast refreshes. |
| POST | `/witos/finops/admin/rebuild` · GET `/admin/status` | Ops: rebuild aggregates/forecasts; watermark/lag status. |

Example response shape for `/forecast` (scope=team finance-ai): actuals `{period_spend, period_tokens}`; forecast `{spend_p50, spend_p90, tokens_p50, budget_exhaustion_p50: "2026-08-28", budget_exhaustion_p90: "2026-08-25", prob_overrun: 0.78}`; quality `{history_days, wape: 0.084, bias, trend: "accelerating", quality_score: 91}`; drivers `[...]`.

**Webhooks:** alert channels of `type:webhook` signed HMAC-SHA256 (`X-WITOS-Signature`, secret from env `WITOS_WEBHOOK_SECRET_<NAME>`); payload = event + latest forecast headline. WIT OS ingests into its event bus.

## 1.12 LiteLLM Admin UI — "Cost Intelligence" (`/cost-intelligence`)

Components under `src/components/witos/finops/`: `FinOpsPage, FinOpsOverview, ForecastChart, BudgetRunway, ForecastDrivers, ScopeSelector, ModelMixChart, ForecastQuality, ScenarioBuilder, ScenarioResults, FinOpsAlerts, QuotaManager`.

- **Filters:** Organization / Team / Key / User / Model / Tag-dimension / Period.
- **Executive cards:** MTD Spend · Projected EOM (P50, with P90 sub-line) · Budget · Projected Variance · Budget Remaining · Days of Runway · Tokens Remaining · Forecast Confidence.
- **Primary chart:** solid actuals, dashed P50, shaded P10–P90 band, budget line, hard vertical **TODAY** marker + budget-reset marker. Never blur actuals and projections. Toggle spend/tokens/requests.
- **Runway table:** per quota — limit, consumed, %, burn/day, P50 & P90 exhaustion, overrun probability, status pill (🟢 survives cycle / 🟡 <30d / 🔴 <7d). CSV export.
- **Drivers panel:** signed contribution list (as §1.9). **Model-mix** treemap + forecast shares.
- **Scenario builder drawer:** assumption controls → dual chart + savings delta + "copy JSON."
- **Alerts panel** + rule editor. **Quality footer** always visible: "Engine: ETS(weekly, damped) · 30d WAPE 7.8% · bias +1.9% · regime: accelerating · computed 02:10 UTC."
- **Honest empty states:** <7d history → burn-rate view + low-confidence banner; no quota → "Set budget →" deep link.

## 1.13 WIT OS integration

`witos-sdk` (TypeScript): OpenAPI-generated client + hooks (`useFinOpsOverview, useForecast, useRunway, useDrivers, useScenarioEval, useFinOpsStream`) + embeddable headless-styled components (`<ForecastChart/> <RunwayGauge/> <BurnCard/> <DriversList/>`) themed via CSS vars (WITONE tokens: CYAN `#2FB4E9`, ORANGE `#F07B2A`, DEEP `#05070D`). Full **EN/ES i18n** bundle — every CFO-facing label ships Spanish.

WIT OS navigation: `LEDGER → AI Spend · Forecast · Budget Runway · Optimization · Scenarios · Chargeback`. Chargeback/markup (`markup_pct` per tenant) is **display-side only** in v1; invoice-grade billing needs its own ledger (explicit non-goal).

## 1.14 FinOps NFRs & tests

- Aggregation ≤ 60s per cycle at 10M SpendLogs/day; forecast job ≤ 5 min for 2,000 scopes (vectorized, small process pool); API p95 < 300 ms (reads hit persisted forecasts); scenario eval ≤ 2s. Money = `Decimal`, USD internal, FX display-only (`WITOS_FinOpsConfig.fx_rate`).
- **Required test matrix:** late SpendLog records · duplicate events · zero-cost models · unknown prices · custom pricing · provider price change mid-period · cache tokens · new model introduction · new team with no history · 400% spike (regime) · weekend seasonality · month boundary · budget reset · quota reset · DST/UTC · deleted user/team · multi-pod lease contention · aggregation worker crash mid-cycle · forecast rebuild idempotency · scenario reproducibility (pricing hash).
- **Prometheus metrics:** `finops_aggregation_lag_seconds, finops_last_success_timestamp, finops_forecast_generation_seconds, finops_forecast_wape, finops_forecast_bias, finops_missing_price_requests, finops_unknown_model_requests`.

---

# PART II — DLP POLICY FEDERATION

## 2.1 Product definition

Not "policy import." **Federation:** some vendors expose policy definitions that can be mirrored locally; others expose only runtime evaluation APIs and remain source of truth. One canonical model, four integration modes, capability discovery per connector, shadow-first deployment, full provenance and audit.

```
Cyera │ Purview │ Nightfall │ Netskope │ Custom REST/WIT-DPS
   └────────── Vendor Adapters (capability negotiation) ──────────┘
                        │
              Canonical WIT-DPS Policy Model  (AST, classifiers, versions)
                        │
        ┌───────────────┴───────────────┐
  Local Enforcement (compiled)    Delegated Evaluation (vendor runtime)
        └───────────────┬───────────────┘
                 Decision Aggregator (action precedence)
                        │
              wit_dlp Guardrail (pre_call / post_call / tool)
                        │
        Prompt · Response · Tool args/results · RAG context
```

## 2.2 Four integration modes (per policy, declared by connector capabilities)

- **MIRROR** — vendor exposes the policy definition; we import, normalize, compile, enforce locally (zero runtime vendor dependency).
- **DELEGATE** — vendor is source of truth; we send content/context to its runtime evaluation API; it returns allow/block/redact/risk + matched policy.
- **HYBRID** — import classification/scope metadata locally; delegate complex decisions.
- **OBSERVE** — evaluate and log only, no enforcement. **Mandatory first stage for every imported policy** (deployment safety).

## 2.3 Capability discovery (never invent vendor endpoints)

Every adapter implements `get_capabilities()`; UI and sync behavior are driven by the result, from this closed set:
```
POLICY_LIST · POLICY_READ · POLICY_VERSION · CLASSIFIER_LIST
REALTIME_INPUT_EVALUATION · REALTIME_OUTPUT_EVALUATION
REDACTION · WEBHOOKS · POLICY_PUSH
```
`UnsupportedCapability` is a first-class exception; unsupported features are hidden, not stubbed with fakes. All vendor endpoint paths live in per-adapter `*_endpoints.py` constants verified against the customer tenant's docs during integration testing — never inline, never guessed.

## 2.4 Canonical model — WIT-DPS v2

Policy document (JSON, JSON-Schema validated, stored + versioned):

```json
{
  "wit_dps_version": "2.0",
  "policy_id": "witdps_7f3a...",
  "source": {
    "vendor": "cyera", "connection_id": "conn_abc",
    "external_policy_id": "cyera-policy-4412", "external_policy_version": "17",
    "external_url": "https://.../policies/4412",
    "imported_at": "2026-08-19T12:00:00Z",
    "external_policy_hash": "sha256:...", "canonical_policy_hash": "sha256:...",
    "unmapped": []
  },
  "name": "PCI - Cardholder Data",
  "severity": "critical",
  "mode": "mirror",
  "priority": 100,
  "direction": "both",
  "scope": {"entities": [{"type": "team", "id": "*"}], "models": ["*"], "applications": ["*"]},
  "condition": {
    "operator": "ALL",
    "conditions": [
      {"type": "data_class", "class": "PCI.CREDIT_CARD", "min_confidence": 0.9},
      {"operator": "NOT", "condition": {"type": "team", "value": "payments-security"}}
    ]
  },
  "actions": {
    "on_match": "BLOCK",
    "redact_strategy": "mask",
    "block_message": "Blocked by policy 'PCI - Cardholder Data' (source: Cyera).",
    "alert_channels": ["slack:secops"],
    "log_payload": "metadata_only"
  },
  "exceptions": [{"type": "key_alias", "value": "pci-approved-service"}]
}
```

**Condition AST:** operators `ALL / ANY / NOT`; leaf types (closed set v1):
```
data_class · sensitivity · regex · dictionary · keyword · sensitivity_label
identity · group · team · organization · application · model · model_group
provider · destination · file_type · tool · tool_argument · classification_source
```
Tiny recursive-descent evaluator, depth-capped, size-capped, no `eval`. (`prompt_intent` reserved for v2.)

**Canonical classifier taxonomy** (`classifier_registry.py`), vendor classifications normalized into:
```
PII.EMAIL PII.SSN PII.PASSPORT PII.DOB PII.PHONE ...
PHI.MEDICAL_RECORD PHI.DIAGNOSIS ...
PCI.CREDIT_CARD PCI.CVV
FINANCIAL.BANK_ACCOUNT FINANCIAL.ROUTING_NUMBER FINANCIAL.IBAN
SECRET.API_KEY SECRET.PASSWORD SECRET.PRIVATE_KEY
IP.SOURCE_CODE IP.TRADE_SECRET
CUSTOM.<vendor>.<name>
```
Mapping rows preserve `{vendor_classifier_id, vendor_name, canonical_class, confidence_translation, metadata}` — **never discard vendor identity.**

**Canonical actions + deterministic precedence** (aggregator resolves multiple simultaneous matches; explicit `priority` can override):
```
BLOCK > REQUIRE_APPROVAL > REDACT > MASK > WARN > AUDIT > ALLOW
```
(`REQUIRE_APPROVAL` = v1.5: hold request, emit approval event; ships behind a flag.)

## 2.5 Data model (Prisma)

```prisma
model WITOS_DLPConnection {
  connection_id  String   @id @default(uuid())
  provider       String   // cyera|purview|nightfall|netskope|custom_rest|custom_witdps
  name           String
  description    String?
  base_url       String
  region         String?
  tenant_external_id String?
  auth_type      String   // api_key|oauth2_client_credentials|bearer
  secret_reference String // env/secret-manager key name — NEVER credentials in DB
  capabilities_json Json? // discovered, cached
  sync_mode      String   @default("manual_approve") // auto_apply|manual_approve
  sync_interval_min Int   @default(60)
  fail_mode      String   @default("fail_open")      // fail_open|fail_closed|observe (per-policy override allowed)
  timeout_ms     Int      @default(800)
  privacy_json   Json?    // §2.11 data-sharing toggles
  enabled        Boolean  @default(true)
  status         String   @default("active")
  last_sync_at   DateTime?
  last_success_at DateTime?
  last_error     String?
  created_by     String
  created_at     DateTime @default(now())
  updated_at     DateTime @updatedAt
}

model WITOS_DLPPolicy {
  policy_id      String   @id @default(uuid())
  name           String
  description    String?
  source         String   // local|cyera|purview|nightfall|netskope|custom
  connection_id  String?
  external_policy_id String?
  external_policy_version String?
  mode           String   // mirror|delegate|hybrid|observe
  status         String   @default("imported") // imported|draft|shadow|active|disabled|stale|conflict|superseded
  priority       Int      @default(100)
  direction      String   @default("both")     // input|output|both|tool
  canonical_policy_json Json
  external_policy_hash  String?
  canonical_policy_hash String
  compile_status String?  // compiled_local|delegated|hybrid|failed
  compile_error  String?
  approved_by    String?
  activated_at   DateTime?
  last_synced_at DateTime?
  created_at     DateTime @default(now())
  updated_at     DateTime @updatedAt
  @@unique([connection_id, external_policy_id])
  @@index([status])
}

model WITOS_DLPPolicyVersion {   // immutable audit history
  policy_version_id String  @id @default(uuid())
  policy_id      String
  version        Int
  source_version String?
  canonical_json Json
  external_json_sanitized Json?   // vendor doc minus secrets/PII
  hash           String
  change_note    String?          // "Cyera v18: action log→block on detector d3"
  created_by     String?          // human or "sync:conn_abc"
  created_at     DateTime @default(now())
  @@unique([policy_id, version])
}

model WITOS_DLPSyncRun {
  id           String   @id @default(uuid())
  connection_id String
  started_at   DateTime @default(now())
  finished_at  DateTime?
  status       String   // running|success|partial|failed
  stats        Json     // {fetched, created, updated, unchanged, stale_marked, failed_mapping, pending_review}
  error_detail Json?
  @@index([connection_id, started_at])
}

model WITOS_DLPDecision {          // decision receipts
  decision_id   String   @id @default(uuid())
  created_at    DateTime @default(now())
  request_id    String            // correlates to LiteLLM_SpendLogs.request_id
  scope_json    Json              // {org, team, key_alias, user, model, application}
  policy_id     String
  policy_version Int
  provider      String?
  external_policy_id String?
  direction     String            // input|output|tool_input|tool_output|rag_context
  decision      String            // ALLOW|AUDIT|WARN|REDACT|MASK|BLOCK|REQUIRE_APPROVAL
  risk_score    Decimal? @db.Decimal(5,2)
  matched_classifiers Json        // [{class:"PII.SSN", count:1, offsets:[...], confidence}] — NEVER raw values
  matched_rule_ids Json?
  vendor_request_id String?
  evaluation_latency_ms Int?
  fail_mode_triggered Boolean @default(false)
  redaction_performed Boolean @default(false)
  shadow        Boolean  @default(false)   // decision from shadow policy — not enforced
  match_hash    String?  // optional one-way hash for correlation only
  @@index([policy_id, created_at])
  @@index([request_id])
}
```
**Hard rules:** never store raw sensitive findings (no `SSN=123-45-6789`) — classifier + count + offsets (+ optional one-way hash). Never store credentials — `secret_reference` only, resolved through the repo's secret-manager abstraction; API never echoes secrets.

## 2.6 Adapter contract (`policy_fabric/providers/base.py`)

```python
class DLPProviderAdapter(ABC):
    provider: str
    async def test_connection(self) -> ConnectorHealth: ...
    async def get_capabilities(self) -> set[Capability]: ...
    async def list_policies(self, since: datetime | None) -> list[VendorPolicyRef]:
        raise UnsupportedCapability
    async def get_policy(self, external_policy_id: str) -> dict:
        raise UnsupportedCapability
    async def list_classifiers(self) -> list[VendorClassifier]:
        raise UnsupportedCapability
    async def evaluate_input(self, ctx: EvalContext) -> VendorDecision:
        raise UnsupportedCapability
    async def evaluate_output(self, ctx: EvalContext) -> VendorDecision:
        raise UnsupportedCapability
    def map_to_witdps(self, raw: dict) -> MappingResult: ...   # canonical + unmapped[] (never silently dropped)
    async def sync(self) -> SyncStats: ...
```

### v1 adapters, build order & mode posture

1. **`custom_witdps.py` / `custom_rest.py`** (week 1) — direct WIT-DPS JSON via `POST /witos/dlp/policies/import` or pull-URL; the universal escape hatch (n8n-scriptable). Capabilities: POLICY_LIST/READ.
2. **`cyera.py`** (flagship) — **capability discovery first:** AI Firewall runtime API available → enable DELEGATE (per-request allow/block, ms-latency; timeout default 800 ms); classification API available → import classification metadata (HYBRID); policy-definition read API exposed for the tenant → enable MIRROR; otherwise delegated remains source of truth. Cyera's **learned classifications** (GenAI-based, customer-specific — often 20–40% of a customer's categories) can never compile locally → always `data_class` leaves resolved via DELEGATE. All paths in `cyera_endpoints.py`, verified against tenant docs.
3. **`purview.py`** — **DELEGATE-first** via protection-scope/content-processing APIs where available; sensitivity labels imported as `sensitivity_label` leaves (HYBRID); MIRROR only if the tenant's API genuinely exposes policy-read (do not assume Graph exposes full DLP rule bodies).
4. **`nightfall.py`** — HYBRID default (detection-rule APIs are strong); MIRROR where policy-management APIs allow.
5. **`netskope.py`** stub + mapping tables + `docs/witos/connector_dev_guide.md` for field engineers.

## 2.7 Sync workflow & drift management

```
CONNECT → DISCOVER CAPABILITIES → FETCH → NORMALIZE → VALIDATE → DIFF → SHADOW → APPROVE → ACTIVE
```
- **Never** does a newly imported policy go straight to production blocking — even with `sync_mode=auto_apply`, first activation of any BLOCK/REQUIRE_APPROVAL policy requires human approval; imports land in `shadow`.
- **Diff by normalized hash:** same → no-op; different → new immutable `WITOS_DLPPolicyVersion` + diff UI + re-shadow-or-approve workflow (an upstream change from log→block must not silently start blocking).
- **Stale handling:** missing from vendor listing → mark `stale`, keep enforcing per config, alert; delete only after configurable grace period (default 7 days / 3 consecutive syncs). Never delete on one failed listing.
- **Shadow analytics:** shadow policies evaluate on live traffic, write `shadow=true` decisions, and the review UI shows "would have blocked N requests / affected teams X,Y in the last 7d" before anyone clicks Activate.

## 2.8 Runtime enforcement — `wit_dlp` guardrail

```yaml
guardrails:
  - guardrail_name: wit-enterprise-dlp
    litellm_params:
      guardrail: wit_dlp
      mode: [pre_call, post_call]
      default_on: true
```

**Flow per request:** resolve identity/scope (org, team, user, key, application tag, model) → resolve applicable active+shadow policies from a short-lived Redis policy-set cache (`dlp:policyset:{org}:{team}:{key}:{version}`, invalidated on any policy change via pubsub — hot-reload, **no proxy restart**) → evaluate:
- **Local (mirrored/hybrid) detectors:** regex/dictionary/keyword compiled with **google-re2** (linear time; imported patterns are untrusted input — patterns re2 rejects fail compile and are flagged in review, never fall back to backtracking `re`). `data_class` leaves resolvable locally map through Presidio recognizers (reuse LiteLLM's Presidio integration).
- **Delegated leaves:** batched per vendor per request; lazy short-circuit evaluation of the AST (cheap local first; vendor calls only if the condition can still flip). Per-connection circuit breaker (5 failures/30s → trip → apply `fail_mode`, alert, `dlp_fail_open_total`/`dlp_fail_closed_total` metrics).
- **Decision aggregator:** precedence §2.4 across all matched policies → single action; write decision receipt async (existing DB-writer queue pattern — never block the request on the log insert).

**Actions:** AUDIT/WARN log+continue (WARN adds response header + UI toast); REDACT/MASK rewrite content spans then continue; BLOCK → `HTTPException(400, {"error": {"type": "witos_dlp_policy_violation", "policy", "source_vendor", "policy_url"}})`.

**Fail-mode matrix:** per-connection default + per-policy override; recommended default = fail-closed for `critical`-severity BLOCK policies, fail-open+alert otherwise. Configurable, never hardcoded, always alerted.

**Latency budget:** local-only policies ≤ 5 ms p95 added; delegated calls only when a policy demands them; attachments scanned up to 1 MB (larger → policy-configurable skip-or-block).

## 2.9 Streaming output (explicit, honest)

`STREAM_OUTPUT_DLP_MODE` per scope/policy set:
- **`buffer_full`** — hold complete answer, evaluate, release if allowed. Strongest guarantee, worst TTFT.
- **`chunk_gate`** — buffer N-token windows with 512-token overlap, inspect, release approved chunks; on BLOCK mid-stream, terminate with error event. Balanced default.
- **`observe_only`** — stream normally, log violations. No output prevention.

**Honesty rule (UI + docs + API):** no mode may claim to block content already sent to the client. Post-call "block" on a completed stream is detection, not prevention — label it as such everywhere.

## 2.10 Tool / agent / RAG DLP (design now, ship v1.5)

`direction` already includes `tool_input | tool_output | rag_context`. Evaluate tool arguments before execution, tool/MCP results and retrieved RAG context **before they enter model context** (the Salesforce-returns-2,500-SSNs case), and agent memory writes. Hook points: LiteLLM's tool-call interception in pre/post hooks; guardrail receives structured tool payloads, applies same policy resolution with `direction` filter. This is the control point that matters most for WIT OS agentic workflows — the interface ships in v1 even if only prompt/response enforcement is active.

## 2.11 Privacy & data-sharing controls (per connection, `privacy_json`)

```
send_full_content_to_provider  YES/NO   (NO → send hashes/excerpts per vendor support)
send_identity                  YES/NO
send_application_id            YES/NO
send_conversation_context      YES/NO
store_vendor_findings          YES/NO
store_local_content            YES/NO   (default NO — metadata_only receipts)
```
Customers will ask whether the DLP integration itself exfiltrates data. Delegated calls: TLS-only, documented per-vendor egress allowlist, payload minimized to inspection-necessary content + opaque correlation ID.

## 2.12 DLP API (`/witos/dlp/*`)

```
GET/POST /connections · GET/PATCH/DELETE /connections/{id}
POST /connections/{id}/test · /discover · /sync
GET /sync-runs?connection_id=
GET /policies?status=&provider= · GET /policies/{id} · /versions · /diff?from=&to=
POST /policies/{id}/shadow · /activate · /disable        (RBAC: dlp:approve / dlp:enforce)
POST /policies/import                                     (WIT-DPS JSON; dlp:import)
POST /test                                                (test bench, §2.14; dlp:test)
GET /decisions?policy_id=&from=&shadow= · GET /decisions/{id}
GET /decisions/summary                                    (by policy/team/model/action, trend)
GET /classifiers · GET /providers                         (registry + capability matrix)
```

## 2.13 UI — "Data Protection" (LiteLLM Admin) + WIT OS

Nav: **Overview · Policies · Providers · Classifiers · Decisions · Testing.**

- **Providers:** vendor cards (status, mode, policy count, classifier count, last sync, runtime latency p50, error rate) + capability badges (Realtime Enforcement / Policy Import / Classifications / Redaction / Output Scanning). Add-connection wizard: provider → base URL/tenant → auth (secret_reference name, validated against secret manager) → **Discover** → capability review → privacy toggles → sync mode → fail mode.
- **Policies:** columns Policy / Source / Mode / Scope / Action / State / Version / Last Synced / Drift. Detail: human-rendered WIT-DPS (not raw JSON), condition tree visual, local/delegated chips per leaf, unmapped-elements warning panel, immutable version timeline + diff viewer, shadow analytics ("would have blocked 14 requests, 3 teams, last 7d"), Shadow/Activate/Disable buttons with audit trail.
- **Decisions:** trend chart, action mix, top policies/teams/models, shadow-vs-enforced toggle, drill to masked receipts.
- **Testing (test bench, prominent):** input text + scope pickers (org/team/application/model/direction) → dry-run against active+shadow → final decision card + per-policy verdicts, matched canonical classifiers, source vendor, per-stage latency, local-vs-external evaluation counts.
- **WIT OS:** SDK gains `useDlpPolicies, useDecisionSummary, <ViolationTrend/> <PolicyStatusBoard/> <ProviderHealth/>`; decision webhooks merge into WIT OS SOC alert stream (Policy Fabric = the data-layer control point in RA-AI-001's UPIA/XPIA architecture alongside AIRS + Arctic Wolf).

## 2.14 DLP NFRs, security & tests

- **Security (non-negotiable):** secret references only; re2 everywhere for imported patterns; AST depth/size caps; mapping treated as a parsing security boundary; approval events audit-logged (who/version/when); receipt masking enforced at write time; strict tenant separation in every query; OAuth token caching + rotation-safe.
- **Required test matrix:** provider timeout · provider 401 · OAuth renewal · rate limiting · policy deletion vs stale grace · policy drift diff · duplicate external policy · classifier mapping conflict · shadow mode isolation (never enforces) · fail open · fail closed · circuit breaker · Unicode + multibyte offsets · attachments · streaming (all 3 modes) · tool-call payloads · redaction correctness (span math) · multiple simultaneous BLOCK policies (precedence) · vendor unavailable · policy-cache invalidation · tenant isolation · secret rotation.
- **Prometheus metrics:** `dlp_evaluations_total, dlp_blocks_total, dlp_redactions_total, dlp_provider_latency_ms, dlp_provider_errors_total, dlp_fail_open_total, dlp_fail_closed_total, dlp_policy_sync_errors, dlp_policy_drift_total, dlp_shadow_would_block_total`.

---

# PART III — DELIVERY PLAN (Claude Code work packages)

Each phase = independently mergeable PR set with tests. Preflight (§0.2) precedes Phase 1.

**Phase 1 — FinOps foundation:** Prisma models (§1.4) + migrations · job-lease scheduler · aggregation engine with watermark/lateness · quota mirror job · backfill CLI · metrics.
**Phase 2 — Forecast engine + API:** usage-based forecasting (roster, backtest/WAPE, history ladder, regime detection, Monte Carlo) · runway probabilities · drivers · scenarios · anomaly job · full `/witos/finops/*` + OpenAPI · alert rules/events + HMAC webhooks.
**Phase 3 — FinOps UIs:** Cost Intelligence page (§1.12) · witos-sdk + i18n · CFO CSV report · SSE.
**Phase 4 — Policy Fabric core:** DLP Prisma models · WIT-DPS v2 schema + JSON-Schema validator + AST evaluator · classifier registry · compiler (re2/Presidio/delegated plans) · `wit_dlp` guardrail (pre/post, actions, precedence, receipts, streaming modes) · custom_witdps/custom_rest connector · import + test-bench + decisions endpoints.
**Phase 5 — Cyera + federation workflow:** cyera adapter (capability discovery, delegate runtime, hybrid classifications, conditional mirror, circuit breaker) · sync engine with versioning/diff/stale-grace · shadow analytics · review/approve UI flow.
**Phase 6 — DLP UIs + adapters:** Data Protection pages · WIT OS SDK DLP additions · purview (delegate-first) · nightfall (hybrid) · netskope stub + connector dev guide · tool/RAG direction activation.

**Ship-order rationale:** FinOps forecast + runway + drivers first (immediate CFO value, no vendor dependencies); DLP canonical engine + adapter contract + shadow mode before any vendor adapter (if the abstraction is right, every new vendor is an adapter, not an integration project).

**Definition of done (global):** OpenAPI documented · RBAC-tested per §0.5 · unit+integration+contract+backtest-regression green · hot-path latency delta < 5 ms p95 (local-only policies) · zero secrets/raw findings in DB · upstream-merge clean (additive only) · EN/ES strings for all customer-facing labels · Prometheus metrics live · migrations documented.

**Env vars:** `WITOS_FINOPS_ENABLED, WITOS_DLP_ENABLED, WITOS_WEBHOOK_SECRET_<NAME>, WITOS_DLP_SECRET_<CONNECTION>, WITOS_DELEGATED_TIMEOUT_MS_DEFAULT=800, WITOS_STREAM_DLP_MODE_DEFAULT=chunk_gate, WITOS_FINOPS_LATENESS_WINDOW_H=2`.

Both feature flags default **off**, and off is inert rather than idle: `witos/registration.py` mounts nothing, schedules nothing, and never imports `finops/` or `policy_fabric/` at all, so a build with the flags unset is indistinguishable from one without this code on the request path. That is deliberately what the first deploy of a new image is tested under, separating "can we run our own image" from "does the new code work". Only an explicit `true` (any casing) enables a subsystem; `1`, `yes` and `on` do not.

Job cadences are the blueprint's, overridable per deployment: `WITOS_FINOPS_AGGREGATION_INTERVAL_MIN=5, WITOS_FINOPS_ANOMALY_INTERVAL_MIN=15, WITOS_FINOPS_QUOTA_MIRROR_HOUR_UTC=1, WITOS_FINOPS_FORECAST_HOUR_UTC=2, WITOS_FINOPS_QUALITY_HOUR_UTC=3`. The nightly three are ordered by what they read: the quota mirror writes the entitlements runway is computed against, and quality scores the forecasts the middle job published. An unparseable or out-of-range value logs a warning and falls back to the default, because a typo in an env var must not stop the gateway from starting.

---

# PART IV — OPEN DECISIONS (confirm before Phase 4/5)

1. **Cyera tenant API surface** — capability discovery handles the unknowns, but the endpoint constants file must be filled from the actual tenant's docs during integration testing.
2. **Fail-mode defaults per customer** — blueprint default: fail-closed for critical BLOCK, fail-open+alert otherwise. Gov clients likely want broader fail-closed; latency-sensitive tenants the opposite.
3. **WIT OS tenant→org mapping** source of truth (WIT OS DB vs LiteLLM org metadata).
4. **Chargeback scope** — display-side markup only (v1) vs invoice-grade ledger (separate project).
5. **REQUIRE_APPROVAL UX** — where the approval queue lives (LiteLLM UI vs WIT OS SOC) before enabling the action.
6. **Strategic end state (v3, do not build yet, do not preclude):** policy-and-cost-aware routing — per request: is it permitted → which models satisfy policy → which permissible model meets quality/SLA → which stays within projected budget. Both engines' outputs are designed to be consumable by the router later; keep interfaces clean for it.
