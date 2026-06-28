#!/usr/bin/env bash
set -euo pipefail
BIN=${1:-./build/tact_bench}
CSV=${2:-results.csv}
N=${N:-1000000}
ITERS=${ITERS:-20}
TERMS=${TERMS:-5}

rm -f "$CSV"
for engine in baseline_fast baseline_sp wmma cusparse; do
  layout=dense
  if [[ "$engine" == "cusparse" ]]; then layout=sparse24; fi
  "$BIN" --engine "$engine" --layout "$layout" --functions sin,cos,tan \
    --terms "$TERMS" --n "$N" --iterations "$ITERS" --threshold 1e-3 --csv "$CSV"
done

echo "Wrote $CSV"
