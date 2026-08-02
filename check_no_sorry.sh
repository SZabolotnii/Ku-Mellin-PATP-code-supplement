#!/usr/bin/env bash
# Guard: fail if any Lean source contains an incomplete/unsafe proof token.
# (plain `decide` is kernel-checked and allowed; `native_decide` is not.)
set -euo pipefail
cd "$(dirname "$0")"
if grep -rniE 'sorry|axiom|admit|native_decide|#exit' Lean/ ; then
  echo "FAIL: forbidden token found in Lean/"
  exit 1
fi
echo "OK: Lean/ is free of sorry/axiom/admit/native_decide/#exit"
