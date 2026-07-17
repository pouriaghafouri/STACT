#!/usr/bin/env bash
set -euo pipefail

BIN="${BIN:-./build/tact_bench}"
FULL="${FULL:-0}"
COMMON=(
  --engine sparse_tc
  --layout dense
  --functions sin,cos,tan
  --terms 3
  --iterations 1
  --threshold 5e-3
  --min -0.78539816339
  --max 0.78539816339
  --profile-sections
)

if [[ ! -x "$BIN" ]]; then
  echo "Error: benchmark executable not found or not executable: $BIN" >&2
  echo "Build the project first, or set BIN=/path/to/tact_bench." >&2
  exit 1
fi

run_case() {
  local n="$1"
  local repeat="$2"
  echo "================ sparse_tc: N=${n}, repeat=${repeat} ================"
  "$BIN" "${COMMON[@]}" --n "$n" --stress-repeat "$repeat"
}

run_case 80 1
run_case 80 2
run_case 80 2000
run_case 160 2000
run_case 83 2000

if [[ "$FULL" == "1" ]]; then
  run_case 10000000 2000
else
  echo "Small sparse_tc tests passed. Run FULL=1 $0 for N=10000000."
fi
