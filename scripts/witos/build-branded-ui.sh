#!/usr/bin/env bash
# Build the WIT OS-branded admin UI from THIS FORK's dashboard source.
#
# Why this replaced the old script
# --------------------------------
# The previous build (in the witone-llm-gateway deployment repo) cloned upstream
# LiteLLM at a pinned tag and branded that. Correct while we had no fork and no
# UI of our own. It is now actively dangerous: the fork's dashboard carries the
# WIT OS pages (Cost Intelligence, Data Protection), and a fresh upstream clone
# does not. Building from upstream would silently ship a UI missing every page
# we added — the branding would be right and the product would be gone.
#
# Two rules this script exists to enforce:
#
#   1. Build from the fork's own ui/litellm-dashboard, never a clone.
#   2. Never mutate that source. Branding is applied to a throwaway copy, so
#      `git status` stays clean and upstream merges stay trivial. A source tree
#      with branding baked in would conflict on every single upstream merge.
#
# Usage:  scripts/witos/build-branded-ui.sh
# Output: build/witos-branded-ui/out   (gitignored; deploy this directory)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$REPO_ROOT/ui/litellm-dashboard"
WORK="$REPO_ROOT/build/witos-branded-ui"

[ -d "$SRC" ] || { echo "FATAL: $SRC not found — are you in the fork?" >&2; exit 1; }

echo "==> staging a throwaway copy (source is never modified)"
rm -rf "$WORK"; mkdir -p "$(dirname "$WORK")"
# Copy source only. node_modules and previous builds are regenerated.
rsync -a --exclude node_modules --exclude .next --exclude out "$SRC/" "$WORK/"

cd "$WORK"
echo "==> npm install"
npm install --no-audit --no-fund

echo "==> applying WIT OS branding to the copy"
python3 - << 'PY'
import os, re

phrases = [
    ("LiteLLM Proxy Admin UI", "WIT OS AI Gateway · Admin Console"),
    ("LiteLLM Dashboard", "WIT OS AI Gateway"),
    ("LiteLLM Proxy", "WIT OS AI Gateway"),
]
url_pats = [
    (re.compile(r'https://docs\.litellm\.ai[^"\'\)\s]*'), "https://witos.ai"),
    (re.compile(r'https://github\.com/BerriAI[^"\'\)\s]*'), "https://witos.ai"),
]
# Case-sensitive, and \b on BOTH sides on purpose: it must not touch
# identifiers such as LiteLLMModelNameField or LiteLLMParams, which have a word
# character immediately after "LiteLLM" and so never match.
word = re.compile(r"\bLiteLLM\b")
# The sweep can still land on an unquoted object key (`LiteLLM:`). Requote it,
# because `WIT OS AI Gateway:` is a syntax error.
danger = re.compile(r'(?<!["\'`\w>/-])WIT OS AI Gateway\s*:')
# A bare `LiteLLM` used as a declared identifier would become invalid too.
# There is none today; fail loudly rather than emit broken JS if one appears.
ident = re.compile(r"\b(?:const|let|var|function|class|interface|type|enum)\s+LiteLLM\b")

changed = 0
for root, _, files in os.walk("src"):
    for f in files:
        if not f.endswith((".tsx", ".ts", ".jsx", ".js", ".css", ".html")):
            continue
        p = os.path.join(root, f)
        s = open(p, encoding="utf-8", errors="ignore").read()
        if ident.search(s):
            raise SystemExit(
                f"FATAL: {p} declares a bare `LiteLLM` identifier. The word sweep "
                "would rename it to an invalid one. Add an exclusion before shipping."
            )
        o = s
        for a, b in phrases:
            s = s.replace(a, b)
        for pat, b in url_pats:
            s = pat.sub(b, s)
        s = word.sub("WIT OS AI Gateway", s)
        s = danger.sub('"WIT OS AI Gateway":', s)
        if s != o:
            open(p, "w", encoding="utf-8").write(s)
            changed += 1
print(f"    files rebranded: {changed}")
PY

echo "==> building"
npm run build

echo "==> verifying the output is branded AND still complete"
grep -q "WIT OS AI Gateway" out/index.html \
  || { echo "FAIL: branding missing from out/index.html" >&2; exit 1; }
grep -qi "litellm proxy admin" out/index.html \
  && { echo "FAIL: upstream branding survived" >&2; exit 1; } || true

# The regression this script exists to prevent: our pages must be in the build.
# Skipped while the WIT OS pages do not exist yet, asserted the moment they do.
if [ -d "$SRC/src/components/witos" ]; then
  for page in cost-intelligence data-protection; do
    grep -rqi "$page" out/ \
      || { echo "FAIL: WIT OS page '$page' missing from the build" >&2; exit 1; }
  done
  echo "    WIT OS pages present in output"
else
  echo "    note: no WIT OS UI components yet (Phase 3/6) — page assertions skipped"
fi

echo
echo "OK — branded static export at: $WORK/out"
echo "Deploy: tar this directory to the gateway host and bind-mount it over the"
echo "image's baked export, exactly as the old ui-out mount did."
