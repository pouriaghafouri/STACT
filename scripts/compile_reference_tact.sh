#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
out=${1:-"$root/build-reference"}
mkdir -p "$out"

# Keep these at -O0 and preserve the distinct compiler modes used by the old
# experiments: only SFU sources receive --use_fast_math.
nvcc -O0 -arch=sm_86 "$root/tact/cosine_tensor_core.cu" -o "$out/cosine_tensor_core"
nvcc -O0 -arch=sm_86 "$root/tact/TripleTranscendCore.cu" -o "$out/TripleTranscendCore"
nvcc -O0 -arch=sm_86 --use_fast_math "$root/tact/cosine_SFU.cu" -o "$out/cosine_SFU"
nvcc -O0 -arch=sm_86 --use_fast_math "$root/tact/TripleTranscendSFU.cu" -o "$out/TripleTranscendSFU"
nvcc -O0 -arch=sm_86 "$root/tact/cosine_SP.cu" -o "$out/cosine_SP"
nvcc -O0 -arch=sm_86 "$root/tact/TripleTranscendSP.cu" -o "$out/TripleTranscendSP"

echo "Reference binaries written to $out"
