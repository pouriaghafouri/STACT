#!/usr/bin/env bash
set -uo pipefail

# One benchmark model only: the original TACT stress workload.  The core
# operation is repeated STRESS_REPEAT times; no clean/alternate mode exists.
BIN=${BIN:-${1:-./build/tact_bench}}
BIN_SLT=${BIN_SLT:-./build-slt/tact_bench}
CSV=${CSV:-${2:-results/benchmark_results.csv}}
OUTDIR=${OUTDIR:-results}
LOGDIR=${LOGDIR:-$OUTDIR/logs}

N=${N:-1000000}
ITERATIONS=${ITERATIONS:-20}
WARMUP=${WARMUP:-0}
TERMS=${TERMS:-3}
STRESS_REPEAT=${STRESS_REPEAT:-2000}
THRESHOLD=${THRESHOLD:-1e-3}
CUSPARSELT_THRESHOLD=${CUSPARSELT_THRESHOLD:-5e-3}
INPUT_MIN=${INPUT_MIN:--0.78539816339}
INPUT_MAX=${INPUT_MAX:-0.78539816339}
VALIDATE=${VALIDATE:-0}
STRICT=${STRICT:-0}
PROFILE_SECTIONS=${PROFILE_SECTIONS:-0}

mkdir -p "$(dirname "$CSV")" "$LOGDIR"

HEADER='case_label,case_description,func_count,functions,method,engine,layout,terms,n,iterations,stress_repeat,threshold,status,avg_ms,min_ms,max_ms,throughput_mvalues_s,failures,max_abs_err,max_rel_err,init_ms,build_power_ms,matmul_ms,postprocess_ms,kernel_ms,total_profiled_ms,h2d_ms,d2h_ms,scalar_ms,fused_kernel_ms,log_file'
printf '%s\n' "$HEADER" > "$CSV"

csv_field() {
  local value=${1//\"/\"\"}
  printf '"%s"' "$value"
}

append_row() {
  local first=1 value
  for value in "$@"; do
    if (( first )); then first=0; else printf ',' >> "$CSV"; fi
    csv_field "$value" >> "$CSV"
  done
  printf '\n' >> "$CSV"
}

record_without_run() {
  local label=$1 description=$2 count=$3 funcs=$4 method=$5 engine=$6
  local layout=$7 threshold=$8 status=$9 log=${10}
  local empty_fields=("" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "")
  append_row "$label" "$description" "$count" "$funcs" "$method" "$engine" \
    "$layout" "$TERMS" "$N" "$ITERATIONS" "$STRESS_REPEAT" "$threshold" \
    "$status" "${empty_fields[@]}" "$log"
}

sfu_supports_case() {
  case "$1" in
    cos|sin,cos|sin,cos,tan) return 0 ;;
    *) return 1 ;;
  esac
}

run_case() {
  local label=$1 description=$2 count=$3 funcs=$4 method=$5
  local engine=$6 layout=$7 bin=$8 threshold=$9
  local stem="${label}_${engine}"
  local log="$LOGDIR/${stem}.log"
  local native_csv="$LOGDIR/${stem}.native.csv"
  rm -f "$native_csv"

  if [[ "$engine" == sfu ]] && ! sfu_supports_case "$funcs"; then
    printf 'unsupported: no old-TACT SFU kernel for %s\n' "$funcs" > "$log"
    record_without_run "$label" "$description" "$count" "$funcs" "$method" \
      "$engine" "$layout" "$threshold" unsupported "$log"
    echo "unsupported: $label / $method"
    return 0
  fi

  if [[ ! -x "$bin" ]]; then
    printf 'unavailable binary: %s\n' "$bin" > "$log"
    record_without_run "$label" "$description" "$count" "$funcs" "$method" \
      "$engine" "$layout" "$threshold" unavailable "$log"
    echo "unavailable: $label / $method"
    return 0
  fi

  local args=(
    --engine "$engine"
    --layout "$layout"
    --functions "$funcs"
    --terms "$TERMS"
    --n "$N"
    --warmup "$WARMUP"
    --iterations "$ITERATIONS"
    --stress-repeat "$STRESS_REPEAT"
    --threshold "$threshold"
    --min "$INPUT_MIN"
    --max "$INPUT_MAX"
    --csv "$native_csv"
  )
  if (( VALIDATE )); then args+=(--validate); else args+=(--no-validate); fi
  if (( PROFILE_SECTIONS )); then
    args+=(--profile-sections)
  else
    args+=(--no-profile-sections)
  fi

  "$bin" "${args[@]}" > "$log" 2>&1
  local exit_code=$?
  local status=ok
  if (( exit_code == 2 )); then
    status=validation_failed
  elif (( exit_code == 3 )); then
    status=unsupported
  elif (( exit_code != 0 )); then
    status=failed
  fi

  if [[ -s "$native_csv" ]]; then
    local row
    row=$(tail -n 1 "$native_csv")
    local n_engine n_layout n_funcs n_terms n_n n_warmup n_iters n_stress
    local n_values n_outputs init h2d kernel d2h avg min max scalar fused
    local build_power matmul postprocess profiled throughput max_abs max_rel failures n_threshold
    IFS=',' read -r n_engine n_layout n_funcs n_terms n_n n_warmup n_iters n_stress \
      n_values n_outputs init h2d kernel d2h avg min max scalar fused build_power \
      matmul postprocess profiled throughput max_abs max_rel failures n_threshold <<< "$row"

    append_row "$label" "$description" "$count" "$funcs" "$method" \
      "$engine" "$layout" "$n_terms" "$n_n" "$n_iters" "$n_stress" \
      "$threshold" "$status" "$avg" "$min" "$max" "$throughput" \
      "$failures" "$max_abs" "$max_rel" "$init" "$build_power" "$matmul" \
      "$postprocess" "$kernel" "$profiled" "$h2d" "$d2h" "$scalar" \
      "$fused" "$log"
  else
    record_without_run "$label" "$description" "$count" "$funcs" "$method" \
      "$engine" "$layout" "$threshold" "$status" "$log"
  fi

  echo "$status: $label / $method (log: $log)"
  if (( STRICT )) && [[ "$status" != ok ]]; then
    return "$exit_code"
  fi
  return 0
}

case_labels=(single dual triple quad)
case_descriptions=(
  "single cosine"
  "dual sine cosine"
  "triple sine cosine tangent"
  "quad sine cosine tangent hyperbolic-sine"
)
case_counts=(1 2 3 4)
case_functions=(cos sin,cos sin,cos,tan sin,cos,tan,sinh)

for index in "${!case_labels[@]}"; do
  label=${case_labels[$index]}
  description=${case_descriptions[$index]}
  count=${case_counts[$index]}
  funcs=${case_functions[$index]}

  run_case "$label" "$description" "$count" "$funcs" SP sp dense \
    "$BIN" "$THRESHOLD" || exit $?
  run_case "$label" "$description" "$count" "$funcs" SFU sfu dense \
    "$BIN" "$THRESHOLD" || exit $?
  run_case "$label" "$description" "$count" "$funcs" "Tensor Core" wmma dense \
    "$BIN" "$THRESHOLD" || exit $?
  # cuSPARSE uses the canonical sequential K layout; it does not need 2:4.
  run_case "$label" "$description" "$count" "$funcs" cuSPARSE cusparse dense \
    "$BIN" "$THRESHOLD" || exit $?
  # Only cuSPARSELt uses the paired 2:4 K permutation.
  run_case "$label" "$description" "$count" "$funcs" cuSPARSELt cusparselt sparse24 \
    "$BIN_SLT" "$CUSPARSELT_THRESHOLD" || exit $?
done

echo "Wrote benchmark CSV to $CSV"
echo "Wrote per-run logs to $LOGDIR"
