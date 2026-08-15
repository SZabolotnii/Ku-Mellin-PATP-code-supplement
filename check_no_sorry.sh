#!/usr/bin/env bash
# Guard: the Lean layer must contain no incomplete or unsafe proof.
#
# Two checks, because they answer two different questions.
#
#   1. Source scan — no `sorry` / `admit` / `native_decide` / `#exit` and no
#      custom `axiom` is WRITTEN in declaration or proof position. Comments and
#      docstrings may legitimately name these tokens (Lean/PATP/Audit.lean
#      discusses `sorry` in prose), so the patterns anchor on syntax rather
#      than on the bare word. The previous version grepped the word anywhere
#      and reported failure the moment the audit file was added.
#
#   2. Axiom audit — and, more strongly, that nothing DEPENDS on such a thing.
#      A theorem can contain no `sorry` anywhere in its text and still reach
#      `sorryAx` through an import; check 1 cannot see that, and neither can
#      the exit code of `lake build`, which is 0 even with sorries in the tree.
#      Lean/PATP/Audit.lean makes the build print every dependency; this
#      verifies what it printed.
#
# Check 2 needs a Lean toolchain and will fetch/build Mathlib on a cold cache.
# Pass --no-build to run the source scan alone.
set -euo pipefail
cd "$(dirname "$0")"

LIB=PATP
STANDARD='propext|Classical.choice|Quot.sound'

echo "== 1/2  source scan =="
if grep -rnE '^[[:space:]]*axiom[[:space:]]|\bnative_decide\b|^#exit|(:=|by|<;>)[[:space:]]*(sorry|admit)\b|^[[:space:]]*(sorry|admit)[[:space:]]*$' \
     Lean/ --include='*.lean'; then
  echo "FAIL: forbidden construct in Lean/ (see the line above)"
  exit 1
fi
echo "OK: no sorry/admit/axiom/native_decide/#exit in declaration position"

if [ "${1:-}" = "--no-build" ]; then
  echo "SKIPPED 2/2 (--no-build): axiom dependencies were NOT verified."
  exit 0
fi

echo "== 2/2  axiom audit =="
LOG=$(mktemp)
trap 'rm -f "$LOG"' EXIT

if ! lake build "$LIB" 2>&1 | tee "$LOG"; then
  echo "FAIL: lake build $LIB did not succeed"
  exit 1
fi

if grep -q "declaration uses 'sorry'" "$LOG"; then
  echo "FAIL: the build reports declarations using 'sorry'"
  exit 1
fi

AUDITED=$(grep -c 'depends on axioms:' "$LOG" || true)
if [ "$AUDITED" -eq 0 ]; then
  echo "FAIL: no '#print axioms' output — nothing was certified."
  echo "      Lean/PATP/Audit.lean must be imported from Lean/PATP.lean."
  exit 1
fi

if grep 'depends on axioms:' "$LOG" | grep -vE "\[($STANDARD)(, ($STANDARD))*\]$"; then
  echo "FAIL: a declaration depends on a non-standard axiom (see the line above)"
  exit 1
fi

echo "OK: $AUDITED declarations audited, all within [propext, Classical.choice, Quot.sound]"
