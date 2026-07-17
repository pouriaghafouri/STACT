# STACT — Sparse Tensor Accelerated Computation of Transcendentals

STACT is a CUDA/C++ research benchmark for evaluating TACT polynomial
approximations of transcendental functions on GPU. The benchmark compares five
execution paths:

- **SP** — standard single-precision CUDA math
- **SFU** — CUDA fast intrinsics used by the reference TACT programs
- **WMMA** — the original dense 16x16 Tensor Core TACT implementation
- **cuSPARSE** — generic CSR SpMM baseline
- **Sparse Tensor Core PTX** — a custom 2:4 structured-sparse TACT engine using
  direct `mma.sp.sync` instructions

The main reference workload is:

```text
functions       = sin,cos,tan
terms           = 3
N               = 10000000
stress-repeat   = 2000
input range     = [-pi/4, pi/4]
threshold       = 5e-3
```

Validation is always enabled and cannot be disabled.

## Requirements

- Linux
- CMake 3.22+
- CUDA Toolkit 12.x
- An Ampere-or-newer NVIDIA GPU for `sparse_tc`
- The examples below target `sm_86`

The custom engine executes the Ampere warp-level instruction:

```text
mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
```

## Build

```bash
rm -rf build

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86

cmake --build build -j
ctest --test-dir build --output-on-failure
```

The executable is:

```text
build/tact_bench
```

The default build keeps the legacy CUDA translation units at `-O0`, matching
the original stress benchmark, and compiles only `sparse_tc_engine.cu` at
`-O3`. CMake removes its target-wide Release optimization flag before applying
these per-source choices, so nvcc receives exactly one optimization level per
translation unit. The following warning must no longer appear:

```text
nvcc warning : incompatible redefinition for option 'optimize'
```

To inspect the actual nvcc command for the custom engine, use a single-job
verbose build:

```bash
rm -rf build
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --verbose -j1 2>&1 | tee build_verbose.log
grep 'sparse_tc_engine.cu' build_verbose.log
```

The `sparse_tc_engine.cu` command should contain `-O3` and must not contain
`-O0`. Legacy engine commands should contain `-O0` when `TACT_CUDA_O0` is left
at its default value.

## Custom sparse Tensor Core engine

Run the new engine with:

```bash
./build/tact_bench \
  --engine sparse_tc \
  --layout dense \
  --functions sin,cos,tan \
  --terms 3 \
  --n 10000000 \
  --iterations 1 \
  --stress-repeat 2000 \
  --threshold 5e-3 \
  --min -0.78539816339 \
  --max 0.78539816339 \
  --profile-sections
```

Accepted aliases are:

```text
sparse_tc
sparse-tc
custom_sparse
mma_sp
```

### What this engine does

The WMMA coefficient matrix is sparse, but its original slot-major K ordering
is not always legal for Ampere 2:4 structured sparsity. The new engine applies
an exact K permutation:

```text
original logical K: slot * terms + term
sparse-TC K:        term * inputs_per_row + slot
```

The same permutation is applied to the power operand, so the mathematical
product does not change. For all supported TACT configurations, this term-major
ordering places at most two nonzero coefficients in each four-wide K group.
No coefficient is removed, approximated, or changed.

The engine then:

1. transposes and compresses the FP16 coefficient matrix as a legal 2:4 sparse
   Tensor Core operand;
2. generates the same FP16-rounded power tile as WMMA;
3. loads sparse coefficients, powers, metadata, and FP32 accumulators once;
4. performs the stress loop using direct sparse Tensor Core PTX;
5. writes final function outputs directly, including the odd-function
   multiplication by `x`.

No CSR matrix, cuSPARSE descriptor, large intermediate matrix, or separate
postprocess kernel is used by `sparse_tc`.

### Exact meaning of `stress-repeat`

One sparse PTX instruction has shape `m16n8k16`, so two instructions are needed
to update a full logical 16x16 TACT tile:

```text
mma.sp for output columns 0..7
mma.sp for output columns 8..15
```

The kernel contains:

```text
for repeat = 0 .. stress_repeat-1:
    sparse MMA for columns 0..7
    sparse MMA for columns 8..15
```

Therefore, with `--stress-repeat 2000`:

```text
logical updates per output      = 2000
physical mma.sp instructions    = 4000 per logical tile
algebraic repeat collapse       = none
A/B reload inside stress loop   = none
C global-memory traffic in loop = none
```

The loop count is runtime data and the instructions are emitted through
`asm volatile`. The custom translation unit is compiled at `-O3` so its
register-only mainloop is optimized, while the volatile instructions prevent
the 2000 repetitions from being replaced by a multiplication by 2000.

### Timing interpretation

`kernel avg ms` measures the complete fused custom kernel, matching the style
of the WMMA measurement. Power-tile construction and final output stores occur
once per tile. Only the two `mma.sp.sync` instructions are inside the 2000-step
stress loop.

The project does not claim a speedup before measurement. `sparse_tc` is the
path with the strongest architectural chance of improving on WMMA because it
combines:

- hardware 2:4 sparse Tensor Core execution;
- register-resident accumulation;
- no library-call boundary inside the recurrence;
- no global result matrix between repetitions.

Actual speed depends on GPU clocks, register allocation, occupancy, and the
instruction throughput of the target GPU.

## Testing `sparse_tc`

The tests below intentionally grow from one logical tile to the full workload.
For `sin,cos,tan` with three terms, one logical tile contains 80 inputs.

### 1. One tile, repeat counts 1, 2, and 2000

```bash
for R in 1 2 2000; do
  echo "================ sparse_tc: N=80 R=${R} ================"
  ./build/tact_bench \
    --engine sparse_tc \
    --layout dense \
    --functions sin,cos,tan \
    --terms 3 \
    --n 80 \
    --iterations 1 \
    --stress-repeat "$R" \
    --threshold 5e-3 \
    --min -0.78539816339 \
    --max 0.78539816339 \
    --profile-sections
done
```

### 2. More than one tile

```bash
./build/tact_bench \
  --engine sparse_tc \
  --layout dense \
  --functions sin,cos,tan \
  --terms 3 \
  --n 160 \
  --iterations 1 \
  --stress-repeat 2000 \
  --threshold 5e-3 \
  --min -0.78539816339 \
  --max 0.78539816339 \
  --profile-sections
```

### 3. Tail handling

`N=83` creates one complete 80-input tile plus a partial tile.

```bash
./build/tact_bench \
  --engine sparse_tc \
  --layout dense \
  --functions sin,cos,tan \
  --terms 3 \
  --n 83 \
  --iterations 1 \
  --stress-repeat 2000 \
  --threshold 5e-3 \
  --min -0.78539816339 \
  --max 0.78539816339 \
  --profile-sections
```

### 4. Full correctness and performance run

```bash
./build/tact_bench \
  --engine sparse_tc \
  --layout dense \
  --functions sin,cos,tan \
  --terms 3 \
  --n 10000000 \
  --iterations 1 \
  --stress-repeat 2000 \
  --threshold 5e-3 \
  --min -0.78539816339 \
  --max 0.78539816339 \
  --profile-sections
```

Every valid invocation must exit with status zero and report both validation
gates as zero failures:

```text
failures:      0
sparse-TC/WMMA parity:
  failures:    0
```

The first `failures` line checks every output against the independent host
FP16-rounded TACT model. The parity line independently compares every output
against the WMMA engine. A failure in either gate returns exit status `2`.

### 5. Direct WMMA comparison

Run both engines with identical arguments and compare `kernel avg ms`.

```bash
echo "================ WMMA ================"
./build/tact_bench \
  --engine wmma --layout dense \
  --functions sin,cos,tan --terms 3 \
  --n 10000000 --iterations 1 --stress-repeat 2000 \
  --threshold 5e-3 \
  --min -0.78539816339 --max 0.78539816339 \
  --profile-sections

echo "================ sparse_tc ================"
./build/tact_bench \
  --engine sparse_tc --layout dense \
  --functions sin,cos,tan --terms 3 \
  --n 10000000 --iterations 1 --stress-repeat 2000 \
  --threshold 5e-3 \
  --min -0.78539816339 --max 0.78539816339 \
  --profile-sections
```

For a less noisy timing comparison after correctness has passed, repeat the
same commands with `--iterations 5`. Validation remains mandatory.

A convenience script runs all small tests and can optionally run the full
10-million-input case:

```bash
bash scripts/test_sparse_tc.sh
FULL=1 bash scripts/test_sparse_tc.sh
```

## Full comparison commands

### SP

```bash
./build/tact_bench \
  --engine baseline_sp \
  --functions sin,cos,tan \
  --terms 3 --n 10000000 --iterations 1 \
  --stress-repeat 2000 --threshold 5e-3 \
  --min -0.78539816339 --max 0.78539816339
```

### SFU

```bash
./build/tact_bench \
  --engine baseline_fast \
  --functions sin,cos,tan \
  --terms 3 --n 10000000 --iterations 1 \
  --stress-repeat 2000 --threshold 5e-3 \
  --min -0.78539816339 --max 0.78539816339
```

### Dense Tensor Core / WMMA

```bash
./build/tact_bench \
  --engine wmma --layout dense \
  --functions sin,cos,tan \
  --terms 3 --n 10000000 --iterations 1 \
  --stress-repeat 2000 --threshold 5e-3 \
  --min -0.78539816339 --max 0.78539816339
```

### Custom sparse Tensor Core PTX

```bash
./build/tact_bench \
  --engine sparse_tc --layout dense \
  --functions sin,cos,tan \
  --terms 3 --n 10000000 --iterations 1 \
  --stress-repeat 2000 --threshold 5e-3 \
  --min -0.78539816339 --max 0.78539816339 \
  --profile-sections
```

### cuSPARSE baseline

```bash
./build/tact_bench \
  --engine cusparse --layout csr \
  --functions sin,cos,tan \
  --terms 3 --n 10000000 --iterations 1 \
  --stress-repeat 2000 --threshold 5e-3 \
  --min -0.78539816339 --max 0.78539816339 \
  --profile-sections
```

## Mandatory validation

SP and SFU use a deterministic device replay. Matrix engines are checked
against an independent host model that rounds powers and coefficients to FP16
before FP32 accumulation.

Both sparse engines also receive a complete output parity check against WMMA:

- `cuSPARSE/WMMA parity`
- `sparse-TC/WMMA parity`

All `N * number_of_functions` outputs are checked. A validation or parity
failure returns exit status `2`.

The host acceptance rule is unchanged:

```text
absolute_error <= threshold OR relative_error <= threshold
```

## Full sweep

The benchmark script now includes `sparse_tc`:

```bash
bash scripts/run_benchmarks.sh
```

Override experiment settings with environment variables:

```bash
N=10000000 ITERATIONS=3 PROFILE_SECTIONS=1 \
  bash scripts/run_benchmarks.sh
```

Results are written to:

```text
results/benchmark_results.csv
```

## Source files relevant to the new engine

```text
src/sparse_tc_engine.cu     direct mma.sp implementation
include/tact/common.cuh     engine enum
include/tact/engines.cuh    engine factory declaration
src/common.cu               CLI parsing and layout rules
src/runner.cu               engine selection and WMMA parity validation
src/main.cu                 result reporting
CMakeLists.txt              source registration and per-file -O3
README.md                   commands and design notes
```
