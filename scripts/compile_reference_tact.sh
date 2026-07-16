#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE_DIR=${SOURCE_DIR:-$ROOT/tact}
OUTDIR=${1:-$ROOT/build-reference}
ARCH=${ARCH:-sm_86}

mkdir -p "$OUTDIR"

compile() {
  local source=$1 output=$2
  shift 2
  if [[ ! -f "$SOURCE_DIR/$source" ]]; then
    echo "missing reference source: $SOURCE_DIR/$source" >&2
    return 1
  fi
  nvcc -O0 -arch="$ARCH" "$@" "$SOURCE_DIR/$source" -o "$OUTDIR/$output"
}

compile cosine_tensor_core.cu cosine_tensor_core
compile TripleTranscendCore.cu TripleTranscendCore
compile cosine_SFU.cu cosine_SFU --use_fast_math
compile TripleTranscendSFU.cu TripleTranscendSFU --use_fast_math
compile cosine_SP.cu cosine_SP
compile TripleTranscendSP.cu TripleTranscendSP

echo "Reference binaries written to $OUTDIR"
