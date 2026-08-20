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
