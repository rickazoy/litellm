# WIT OS fork map — preflight findings (blueprint §0.2)

Verified against **this fork at `main`**, not upstream assumptions.

| Blueprint assumption | Reality in this fork |
|---|---|
| Prisma schema | `litellm/proxy/schema.prisma` ✓ |
| Spend endpoints (RBAC scope-filtering pattern to copy) | `litellm/proxy/spend_tracking/spend_management_endpoints.py` ✓ |
| Cost calculator (ONLY pricing source) | `litellm/cost_calculator.py` ✓ |
| Guardrail base class | `litellm/integrations/custom_guardrail.py` ✓ |
| Admin dashboard | `ui/litellm-dashboard/` ✓ |

## LICENSING BOUNDARY — read before writing a line

The enterprise-licensed tree is **`enterprise/` at the repository root** (211 files),
not `litellm/enterprise` as the blueprint states. Do not import from it, copy from
it, or reproduce its patterns. Everything WIT OS builds is MIT-side, under
`litellm/proxy/witos/`.

## Merge hygiene

`upstream` remote points at BerriAI/litellm. All WIT OS code is **additive** and
namespaced under `litellm/proxy/witos/` plus `ui/litellm-dashboard/src/components/witos/`.
The only edits permitted outside that namespace are registrations (router include,
guardrail registry entry, Prisma model additions), each of which must be a single
identifiable hunk so an upstream merge conflict is trivial to resolve.

---

# Preflight findings, DLP Policy Fabric (Phase 4)

Verified by reading this fork and by running the result against it.
# Preflight findings, FinOps Phase 1

Verified by reading this fork, and by running the aggregation SQL against a real
PostgreSQL 16 instance.

## Prisma schema: three copies, root is source of truth

`schema.prisma` at the repo root is authoritative. `litellm/proxy/schema.prisma`
and `litellm-proxy-extras/litellm_proxy_extras/schema.prisma` are byte-identical
copies, enforced by `.github/workflows/check-schema-sync.yml`. Every model
addition goes into all three.

The datasource block is named `client`, not `db`, so native database attributes
are written `@client.Decimal(5, 2)`. `@db.Decimal(...)` as the blueprint spells
it fails `prisma validate` here.

## Guardrail registration is directory-scanned, not hardcoded

`guardrail_registry.py:107-249` walks every subdirectory of
`litellm/proxy/guardrails/guardrail_hooks/` that has an `__init__.py`, imports
it, and merges any `guardrail_initializer_registry` / `guardrail_class_registry`
it exports into the module-level dicts. A new guardrail therefore touches
`guardrail_registry.py` not at all. The only shared-file edits are the
`SupportedGuardrailIntegrations` enum member and, for the admin UI to render its
fields, the config model import plus its entry in the `LitellmParams` base list
(`litellm/types/guardrails.py`). Both are single hunks.

`CustomGuardrail` dispatches to `apply_guardrail` only when a subclass overrides
it (`uses_apply_guardrail_interface`, `custom_guardrail.py:634`). `wit_dlp` does
not, so it gets the native lifecycle hooks it needs for streaming.
`get_config_model()` returning `None` is silent: the guardrail still works and
simply never appears in the UI's field discovery (`guardrail_endpoints.py:1943`).

## Presidio is HTTP, not in-process

`_OPTIONAL_PresidioPIIMasking.analyze_text(text, presidio_config, request_data)`
(`guardrail_hooks/presidio.py:275`) posts to a configured analyzer service and
returns `PresidioAnalyzeResponseItem` dicts with `entity_type` / `start` / `end`
/ `score`. Offsets are code points. The policy fabric reuses that call through
`presidio_bridge.py` rather than adding a second recogniser set, and finds a
configured instance by scanning `litellm.callbacks`.

## Redis pub/sub already exists; cluster mode silently degrades

`proxy/common_utils/config_sync_pubsub.py` is the pattern: reach the
coordination Redis through `proxy_server.redis_usage_cache` (assigned only at
startup, so import it late), call `init_async_client()`, and narrow the result
to `redis.asyncio.Redis`. A `RedisCluster` client has no usable `pubsub()` here
and the helper returns `None`, so any invalidation built on this path falls back
to TTL expiry under cluster Redis. The DLP policy cache keeps a TTL floor for
exactly that reason.

## Secrets

`get_secret_str(name)` in `litellm/secret_managers/main.py:139` is sync, returns
`str | None`, consults the configured manager when one exists and falls back to
`os.environ`. There is no async variant at that layer.

## Off-request-path DB writes

There is no single generic write queue. `gateway_request_tracking.py` is the
closest reusable shape and the one the decision-receipt queue follows: a
synchronous `record` that never awaits, plus a drain called by a scheduled
flush. `BaseUpdateQueue.add_update` awaits `queue.put`, so it blocks the caller
once the queue is full, which is the wrong shape for a guardrail.

## Prometheus

`db_transaction_queue/spend_log_cleanup_metrics.py` is the template: lazily
registered instruments, `prometheus_client` imported inside the function because
it is an optional extra, every handle nullable, `_initialized` set before the
try so a failed registration is not retried per call. `/metrics` scrapes the
default registry, so a counter registered this way needs no callback wiring.

## Lint gates a new module has to clear

`ruff.toml` (base), `ruff-strict.toml` (bans `typing.Any`, `typing.cast`,
`typing.List/Dict/Set`, `TypeGuard`), `scripts/check_type_discipline.py`
(LIT001/2 no mutable collections, LIT010 `Final` on every assignment plus
`TypeAlias` on implicit type aliases, LIT011 no parameter rebinding, and note
that `del param` to mark an argument unused counts as rebinding), and
basedpyright in strict mode. `ReadOnly` comes from `typing_extensions`. Line
length is 120.
copies, enforced by `.github/workflows/check-schema-sync.yml` and regenerated by
`sync-schema.yml`. Any model addition must be made at the root and copied to both.

The datasource block is named `client`, not `db`, so native database attributes
are written `@client.Decimal(18, 8)`. `@db.Decimal(...)` as spelled in the
blueprint fails `prisma validate` in this fork.

## Migrations

Live in `litellm-proxy-extras/litellm_proxy_extras/migrations/<timestamp>_<name>/migration.sql`,
applied in lexical order. Every statement in the tree uses `IF NOT EXISTS`, and
Prisma truncates index names to 63 characters keeping the `_key` / `_idx` suffix
(so `WITOS_FinOpsQuota_source_type_scope_type_scope_id_metric_pe_key`, not the
naive truncation Postgres would produce). Prisma's `@default(uuid())` is
client-side only: the column carries no database default, so raw-SQL inserts must
supply an id (`gen_random_uuid()::text`, core since PostgreSQL 13).

## LiteLLM_SpendLogs: the aggregation source

`schema.prisma:611`. Indexed on `startTime`, `(startTime, request_id)`,
`end_user`, `session_id`. `request_id` is the primary key, so duplicate events
cannot exist at source; idempotency is entirely about repeated aggregation.

- `startTime` / `endTime` / `created_at` are `TIMESTAMP(3)` **without** time
  zone, holding UTC. Bind window bounds as `$1::timestamptz AT TIME ZONE 'UTC'`,
  the idiom `spend_management_endpoints.py` already uses.
- `created_at` exists but is **not indexed**, which is why the watermark tracks
  `startTime` (indexed) plus a lateness window rather than insert time.
- `spend` is `DOUBLE PRECISION`. Cast to `numeric` in SQL before it becomes money.
- `status` is `'success' | 'failure'`, and NULL on rows written before it existed.
- `metadata` and `request_tags` are `JSONB`.
- **Cache tokens are not columns.** `spend_tracking_utils.py:390` nests them at
  `metadata -> additional_usage_values -> cache_read_input_tokens` /
  `cache_creation_input_tokens`. Read them with a `jsonb_typeof(...) = 'number'`
  guard: the value is untrusted.
- There is no per-row `provider` beyond `custom_llm_provider`, and no
  `model_group` guarantee (it defaults to `''`).

## The spend writer, and why lateness is real

`litellm/proxy/db/db_spend_update_writer.py`. Rows are queued in memory and
flushed on a timer, `PROXY_BATCH_WRITE_AT` (default 10s, `litellm/constants.py:1523`),
jittered by up to 5s in `proxy_server.py:8838`. Retention (`spend_log_cleanup.py`)
deletes old rows outright, which is why forecasting must read the WIT OS fact
tables and never raw spend logs.

## Budgets are attached two different ways

`LiteLLM_BudgetTable` is referenced by `budget_id` from organization, project,
key, end user, tag, team membership and organization membership. **Teams and
users carry `max_budget` / `budget_duration` / `budget_reset_at` inline on their
own rows and have no `budget_id` at all** (`schema.prisma:127`, `:244`). A quota
mirror that read only `LiteLLM_BudgetTable` would silently omit the two scopes
that matter most.

`tpm_limit` / `rpm_limit` / `max_parallel_requests` live on the same tables and
are rate limits, never entitlements. They are deliberately not mirrored.

## Background jobs and multi-pod safety

APScheduler `AsyncIOScheduler`, wired in `proxy_server.py` around line 8809.
Existing jobs elect a leader with `PodLockManager`
(`litellm/proxy/db/db_transaction_queue/pod_lock_manager.py`), a **Redis** `SET NX EX`
lock that no-ops when Redis is absent. WIT OS jobs use a database lease instead
(`WITOS_BackgroundJobLease`) so a Redis-less deployment still gets single-writer
guarantees. `LiteLLM_CronJob` exists with a similar `ttl` lease shape but is not
used by the newer jobs.

`now()` is `timestamptz`; these columns are naive `TIMESTAMP(3)`. Writing a bare
`now()` into one converts through the session's `TimeZone`, so two pods with
different session zones would disagree about when a lease expires. Every WIT OS
statement writes `(now() AT TIME ZONE 'UTC')`.

## Prometheus metrics pattern

`spend_log_cleanup_metrics.py` is the template: a class of lazily-registered
instruments, registration deferred to first use, every recorder a no-op when
`prometheus_client` is absent. WIT OS FinOps metrics follow it exactly.

## Raw SQL through Prisma

`prisma_client.db.query_raw(sql, *args)` / `.execute_raw(...)` with `$n`
placeholders. Python sequences bind to `$n::text[]` (`_ORG_SPEND_REPORT_SQL` in
`spend_management_endpoints.py` already does this). Typed row handling is by
`TypedDict` plus a generic `_query_raw` helper; WIT OS validates rows with a
pydantic `TypeAdapter` at the same boundary.

## RBAC scope filtering (for the Phase 2 API)

`spend_management_endpoints.py:1595` `_resolve_spend_report_scope` is the pattern
to copy: a non-admin caller is clamped to its own identity and a mismatch is a
403; `_is_admin_view_safe` gates the admin bypass; `_resolve_org_spend_report_scope`
adds the org-admin case. Scope columns are interpolated only from a module-level
`frozenset`, never from caller input, with values bound as parameters.

## Testing against a real database

`pytest-postgresql` and `psycopg` are declared dev dependencies but no test in
this fork used them before now. `tests/witos/finops/conftest.py` applies the whole
committed migration chain to an empty database and runs the real SQL against it.
Point `WITOS_TEST_DATABASE_URL` at a local server, or leave it unset and the
fixtures probe `127.0.0.1:5432`, skipping the suite when nothing answers.

The tests bind through psycopg rather than Prisma, so `$n` placeholders are
rewritten to `%s` in the test executor. One consequence leaks into production
code: psycopg binds a Postgres array from a Python `list` and a composite type
from a `tuple`, so the approved-tag-dimension parameter is passed as a list.

## Lint gates a new module has to clear

`ruff.toml` (base), `ruff-strict.toml` (strict gate, `typing.Any`, `typing.cast`,
`typing.List/Dict/Set` and `TypeGuard` are banned imports),
`scripts/check_type_discipline.py` (LIT001/2 no mutable collections, LIT010
`Final` on every assignment, LIT011 no parameter rebinding, LIT012 `ReadOnly`
TypedDict fields), and `basedpyright` in strict mode with `reportAny` and
`reportExplicitAny` as errors. `ReadOnly` comes from `typing_extensions`, not
`typing`, at this repo's 3.12 target. Python line length is 120.
