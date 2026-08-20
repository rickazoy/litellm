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
