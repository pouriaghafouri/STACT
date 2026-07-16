# STACT — Sparse Tensor Accelerated Computation of Transcendentals

STACT is a CUDA research benchmark for comparing four implementations of the
same transcendental-function workloads:

- **SP** — standard single-precision CUDA math functions
- **SFU** — the fast intrinsic kernels used by the reference TACT programs
- **Tensor Core** — the WMMA implementation of TACT
- **cuSPARSE** — CSR sparse matrix multiplication using the same TACT
  coefficient and power matrices

The SP, SFU, and WMMA kernels follow the reference programs in `tact/`.

## Supported benchmark cases

| Case | Functions |
|---|---|
| Single | `cos` |
| Dual | `sin,cos` |
| Triple | `sin,cos,tan` |
| Quad | `sin,cos,tan,sinh` |

SFU is available only for the single, dual, and triple cases. The quad case is
not run with SFU because this project does not define an SFU implementation for
`sinh`.

The reference TACT polynomial configuration is:

```text
terms = 3
stress-repeat = 2000
```

## Requirements

- Linux
- CMake 3.22 or newer
- CUDA Toolkit with cuSPARSE
- An NVIDIA GPU supporting the selected CUDA architecture

The commands below target `sm_86`.

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

## Reference program compilation

Compile the standalone programs under `tact/` with the following styles.

Tensor Core:

```bash
nvcc -O0 -arch=sm_86 tensor_core.cu -o tensor_core
```

SFU:

```bash
nvcc -O0 -arch=sm_86 --use_fast_math SFU.cu -o SFU
```

SP:

```bash
nvcc -O0 -arch=sm_86 SP.cu -o SP
```

Replace the generic source and output names with the actual filenames in
`tact/`.

## Standard experiment configuration

The examples use:

```text
N              = 10000000
terms          = 3
stress-repeat  = 2000
iterations     = 1
input range    = [-pi/4, pi/4]
threshold      = 5e-3
```

`N=10000000` is the documented default experiment size. It avoids the much
larger intermediate allocations produced by `N=100000000` on common 4 GB and
6 GB GPUs.

## One example for each method

### SP

```bash
./build/tact_bench \
  --engine baseline_sp \
  --functions sin,cos,tan \
  --terms 3 \
  --n 10000000 \
  --iterations 1 \
  --stress-repeat 2000 \
  --threshold 5e-3 \
  --min -0.78539816339 \
  --max 0.78539816339
```

### SFU

```bash
./build/tact_bench \
  --engine baseline_fast \
  --functions sin,cos,tan \
  --terms 3 \
  --n 10000000 \
  --iterations 1 \
  --stress-repeat 2000 \
  --threshold 5e-3 \
  --min -0.78539816339 \
  --max 0.78539816339
```

### Tensor Core / WMMA

```bash
./build/tact_bench \
  --engine wmma \
  --layout dense \
  --functions sin,cos,tan \
  --terms 3 \
  --n 10000000 \
  --iterations 1 \
  --stress-repeat 2000 \
  --threshold 5e-3 \
  --min -0.78539816339 \
  --max 0.78539816339
```

### cuSPARSE

```bash
./build/tact_bench \
  --engine cusparse \
  --layout csr \
  --functions sin,cos,tan \
  --terms 3 \
  --n 10000000 \
  --iterations 1 \
  --stress-repeat 2000 \
  --threshold 5e-3 \
  --min -0.78539816339 \
  --max 0.78539816339
```

## Copy-paste all methods

```bash
echo "================ SP ================"
./build/tact_bench \
  --engine baseline_sp \
  --functions sin,cos,tan \
  --terms 3 \
  --n 10000000 \
  --iterations 1 \
  --stress-repeat 2000 \
  --threshold 5e-3 \
  --min -0.78539816339 \
  --max 0.78539816339

echo "================ SFU ================"
./build/tact_bench \
  --engine baseline_fast \
  --functions sin,cos,tan \
  --terms 3 \
  --n 10000000 \
  --iterations 1 \
  --stress-repeat 2000 \
  --threshold 5e-3 \
  --min -0.78539816339 \
  --max 0.78539816339

echo "============ Tensor Core ==========="
./build/tact_bench \
  --engine wmma \
  --layout dense \
  --functions sin,cos,tan \
  --terms 3 \
  --n 10000000 \
  --iterations 1 \
  --stress-repeat 2000 \
  --threshold 5e-3 \
  --min -0.78539816339 \
  --max 0.78539816339

echo "============= cuSPARSE ============="
./build/tact_bench \
  --engine cusparse \
  --layout csr \
  --functions sin,cos,tan \
  --terms 3 \
  --n 10000000 \
  --iterations 1 \
  --stress-repeat 2000 \
  --threshold 5e-3 \
  --min -0.78539816339 \
  --max 0.78539816339
```

## Mandatory validation

Validation is always enabled. There is no enable/disable validation flag.
Every command exits with status `2` when validation fails.

The output reports:

- number of validated function outputs
- maximum absolute error
- maximum relative error
- total failure count
- per-function maximum errors and failure counts

Validation is performed outside the timed benchmark region.

### SP and SFU validation

SP and SFU use a full second device replay from the original input and compare
all `N * F` outputs. This checks deterministic execution, indexing, state
updates, and output placement without using host `libm` as a false reference
for a 2000-step CUDA state chain.

### Tensor Core validation

WMMA compares every output against an independent host implementation of the
TACT polynomial. The host model applies the same FP16 rounding to powers and
coefficients and evaluates:

```text
C = 1 + stress_repeat * (A * B)
```

### cuSPARSE validation

cuSPARSE has two mandatory full-output checks:

1. comparison against the independent host FP16-rounded TACT model
2. direct output parity comparison against the trusted WMMA implementation

A valid cuSPARSE run must report zero failures in both sections.

## Why the previous cuSPARSE path was very slow

The previous implementation called `cusparseSpMM()` 2000 times. For the triple
case with `N=10000000`, the result matrix contains about 30 million FP32 values.
Each repetition therefore read and wrote a very large result matrix from global
memory. WMMA does not have that cost because its accumulator remains inside the
warp fragment during the 2000 `mma_sync` operations.

The repeated sparse recurrence is linear:

```text
C_0     = 1
C_(r+1) = B^T * A^T + C_r
```

Therefore:

```text
C_R = 1 + R * (B^T * A^T)
```

The cuSPARSE engine now computes the same final algebra with one SpMM call:

```text
alpha = stress_repeat
beta  = 1
C     = alpha * (B^T * A^T) + beta * C
```

It also omits zero-only K rows and output rows from the cuSPARSE descriptors and
buffers. This removes the 2000 large global-memory result updates that caused
the multi-second runtime. The mandatory host validation and WMMA parity check
verify the final result.

This optimization means `stress-repeat` is an **effective algebraic repeat
count** for cuSPARSE rather than 2000 physical library calls. The program prints
this explicitly in the cuSPARSE kernel path.

## Matrix formulation

The logical Tensor Core operation is:

```text
C = A * B
```

- `A` contains FP16 input powers.
- `B` contains FP16 Taylor coefficients.
- `C` uses FP32 accumulation.

cuSPARSE computes the transposed equivalent:

```text
C^T = B^T * A^T
```

The coefficient matrix is stored as CSR. For the triple three-term case:

```text
inputs per row = 5
used K rows    = 15
used C rows    = 15
nonzeros       = 45
```

The final postprocess uses the same function-column mapping and odd-function
multiplication by `x` as WMMA.

## Full benchmark sweep and CSV

Run all single, dual, triple, and quad cases:

```bash
bash scripts/run_benchmarks.sh
```

The default sweep uses `N=10000000`, one measured iteration, three terms, and a
stress count of 2000. Override settings with environment variables:

```bash
N=10000000 ITERATIONS=3 PROFILE_SECTIONS=1 \
  bash scripts/run_benchmarks.sh
```

Results are written to:

```text
results/benchmark_results.csv
```

The CSV includes timing, throughput, mandatory validation results, and the
additional cuSPARSE/WMMA parity results.

## Plotting

```bash
python3 scripts/plot_results.py \
  results/benchmark_results.csv \
  --outdir results/plots
```

The script plots successful SP, SFU, Tensor Core, and cuSPARSE rows. Unsupported
SFU cases and validation failures are listed in `status_summary.txt`.

## Section profiling

Add:

```bash
--profile-sections
```

For cuSPARSE this reports:

- power-matrix construction and C initialization
- the single algebraically scaled SpMM
- postprocessing

The profile pass is separate from the main latency measurement.

## Meaning of `N`

`N` is the number of input values in one benchmark iteration. For three
functions and `N=10000000`, one iteration produces:

```text
3 * 10000000 = 30000000 function outputs
```

Approximate major allocations for the triple case include:

- input: 40 MB
- output: 120 MB
- cuSPARSE power matrix: about 60 MB
- cuSPARSE result matrix: about 120 MB

Host-side validation also stores output and reference arrays. Keep sufficient
system RAM as well as GPU memory available.

## Debugging

Use a small `N` with `--debug`:

```bash
./build/tact_bench \
  --engine cusparse \
  --layout csr \
  --functions sin,cos,tan \
  --terms 3 \
  --n 64 \
  --iterations 1 \
  --stress-repeat 2000 \
  --threshold 5e-3 \
  --min -0.78539816339 \
  --max 0.78539816339 \
  --debug \
  --debug-values 8 \
  --debug-rows 2
```

The debug dump includes the logical power matrix, coefficient matrix, expected
intermediate values, CSR metadata, final outputs, and validation errors.

## Notes

- Use `terms=3` for direct compatibility with the reference cosine and triple
  TACT kernels.
- Taylor accuracy depends on term count and input range.
- Tangent error grows on wider intervals.
- cuSPARSE uses `CUSPARSE_SPMM_CSR_ALG2` with row-major dense matrices.
- Increase `--iterations` only after the one-iteration commands pass mandatory
  validation.
