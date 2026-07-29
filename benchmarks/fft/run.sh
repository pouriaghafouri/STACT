#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "$0")"

nvcc -o FFT_Parallel_Tensor FFT_Parallel_Tensor.cu -O0 -arch=sm_86
nvcc -o FFT_Parallel FFT_Parallel.cu -O0 -arch=sm_86
nvcc -o FFT_Parallel_SFU FFT_Parallel_SFU.cu -O0 -arch=sm_86 --use_fast_math
nvcc -o FFT_Parallel_SFU_changed FFT_Parallel_SFU_changed.cu -O0 -arch=sm_86 --use_fast_math -Xptxas -O1
