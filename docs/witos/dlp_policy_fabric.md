# WIT OS DLP Policy Fabric

Phase 4 of blueprint v2: the canonical policy model, the compiler, and the
`wit_dlp` runtime guardrail. No vendor adapter beyond the custom pair ships
here, which is deliberate. The abstraction gets proved against the escape hatch
first, so every vendor after it is an adapter rather than an integration project.

## The four federation modes

**MIRROR** means the vendor exposes its policy definition, we import it,
normalise it, compile it and enforce it locally, with no runtime dependency on
the vendor at request time. **DELEGATE** means the vendor stays the source of
truth and we call its evaluation API per request. **HYBRID** imports the
classification and scope metadata locally and delegates the decisions that only
the vendor can make, which is what learned, customer-specific classifiers
require. **OBSERVE** evaluates and logs without enforcing.

## Why shadow is mandatory first

Every imported policy lands in `shadow`, whatever the connection's `sync_mode`
says, and `POST /witos/dlp/policies/{id}/activate` refuses to promote a policy
that has not run in shadow. A policy whose action is BLOCK needs `dlp:approve`
on top of `dlp:enforce`.

The reason is that an imported policy is a stranger. Nobody has yet seen what it
matches against this tenant's real traffic, the vendor's own semantics may not
survive translation exactly, and a policy that looked like an audit rule
upstream can arrive as a blocking rule after a vendor-side edit. Shadow answers
"what would this have done to us last week" with data rather than with
confidence. A shadow verdict is partitioned out of the aggregation before an
action is chosen, so there is no code path in which it changes a response.

## The streaming honesty rule

`buffer_full` holds the whole answer, evaluates, then releases. It can prevent
disclosure, at the cost of time to first token. `chunk_gate`, the default,
buffers a window with an overlap so a span crossing a chunk boundary is still
inspectable, and can prevent disclosure of anything still inside that window.
`observe_only` streams every chunk as it arrives and evaluates afterwards.

No mode may claim to block content that has already been sent to the client.
A post-call block on a stream the caller has already consumed is detection, and
it is recorded, reported and documented as detection. `StreamingMode.prevents_disclosure`
is the rule in code, `WITOS_DLPDecision.prevented` is the rule in the database,
and the `enforcement` field on `GET /witos/dlp/decisions` is the rule in the API.
`observe_only` returns `false` from all three, always. Telling a security team
they are protected against an exfiltration path that is in fact wide open is
worse than telling them the path exists.

## Security posture

**Findings.** A decision receipt records that something was found, never what was
found. `MatchedClassifier` has no field capable of holding matched text, and the
only constructor takes `Finding` objects carrying offsets. Masking is structural
and happens at write time, not at read time, because receipts outlive requests
and get exported. The optional correlation hash is emitted only when
`wit_dlp_match_hash_salt` is set: an unsalted SHA-256 of a nine-digit number is a
lookup table, not a one-way function.

**Credentials.** A connection stores `secret_reference`, the name of a key in the
proxy's secret manager. The value is resolved at call time through
`get_secret_str` and never becomes a column, a response field or a log line. The
API reports whether a reference resolves, never what it resolves to.

**Vendor input.** Mapping is treated as a parsing security boundary. The JSON
Schema sets `additionalProperties: false` everywhere, the AST parser is depth
capped at 12 and size capped at 256 nodes, and an unrecognised vendor action is
an error rather than a guess. Anything a mapping cannot express is reported in
`MappingResult.unmapped` rather than dropped.

**Tenancy.** Policies and receipts are loaded with an organization filter rather
than filtered after loading, so a policy belonging to another tenant is never in
memory to be leaked.

## Why google-re2 and not `re`

Imported patterns are untrusted input. Python's `re` is a backtracking engine, so
a crafted pattern turns the guardrail into a denial of service against the proxy
that runs it. re2 is linear time by construction and refuses the constructs that
make backtracking explosive: backreferences and lookaround.

The consequence is deliberate. A pattern re2 rejects fails the compile, the
policy does not install, and the reason lands in the review queue with the leaf's
node id. It is never retried with `re`, and it is never silently skipped, because
a policy that quietly stops matching is worse than one that visibly fails to
install. `tests/witos/dlp/test_compiler_re2.py` asserts at the source level that
nothing under `policy_fabric/` imports `re`.

## Enable it

```yaml
guardrails:
  - guardrail_name: wit-enterprise-dlp
    litellm_params:
      guardrail: wit_dlp
      mode: [pre_call, post_call]
      default_on: true
      wit_dlp_streaming_mode: chunk_gate
      wit_dlp_fail_mode: fail_open
```

`google-re2` is required for local detectors and ships in the `witos` extra.
Without it the compiler fails every local pattern with an actionable message
rather than degrading to a weaker engine.

## Migrations

**Migrations are not applied automatically and must be applied by hand.** The
Prisma models live in all three copies of `schema.prisma`, and the SQL is
`litellm-proxy-extras/litellm_proxy_extras/migrations/20260819210000_witos_dlp_policy_fabric/migration.sql`.
It is additive: five new tables, no existing table altered.

## What is not here yet

Vendor adapters for Cyera, Purview, Nightfall and Netskope, the sync engine with
drift diffing and the stale grace period, shadow analytics in the UI, the
approval queue that `REQUIRE_APPROVAL` needs before it can be activated, and
enforcement on the `tool_input`, `tool_output` and `rag_context` directions. The
direction interface ships now so callers can be written against it: those
directions evaluate and write receipts, and `enforcement_active` comes back
false.
