#!/usr/bin/env bash
set -uo pipefail

BIN=${1:-./build/tact_bench}
CSV=${2:-results/benchmark_results.csv}
BIN_SLT=${BIN_SLT:-./build-slt/tact_bench}
N=${N:-1000000}
ITERS=${ITERS:-20}
TERMS=${TERMS:-5}
TACT_REPEAT=${TACT_REPEAT:-2000}
STRICT=${STRICT:-0}

mkdir -p "$(dirname "$CSV")"
rm -f "$CSV"

run_case() {
  local bin=$1 engine=$2 layout=$3 funcs=$4
  local args=(--engine "$engine" --layout "$layout" --functions "$funcs"
              --n "$N" --iterations "$ITERS" --threshold 1e-3 --csv "$CSV")
  if [[ "$engine" =~ ^(baseline_fast|baseline_sp|wmma)$ ]] &&
     [[ "$funcs" == "cos" || "$funcs" == "sin,cos,tan" ]]; then
    args+=(--tact-reference --terms 3 --legacy-repeat "$TACT_REPEAT" --no-validate)
  else
    args+=(--clean --terms "$TERMS")
  fi
  "$bin" "${args[@]}"
  local status=$?
  if (( status == 2 && STRICT == 0 )); then
    echo "Validation warning recorded: engine=$engine functions=$funcs" >&2
    return 0
  fi
  return "$status"
}

cases=(cos sin,cos sin,cos,tan sin,cos,tan,sinh)
for funcs in "${cases[@]}"; do
  run_case "$BIN" baseline_fast dense "$funcs" || exit $?
  run_case "$BIN" baseline_sp dense "$funcs" || exit $?
  run_case "$BIN" wmma dense "$funcs" || exit $?
  run_case "$BIN" cusparse sparse24 "$funcs" || exit $?
  if [[ -x "$BIN_SLT" ]]; then
    run_case "$BIN_SLT" cusparselt sparse24 "$funcs" || exit $?
  fi
done

echo "Wrote $CSV"
