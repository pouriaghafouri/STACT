# TACT benchmark project

TACT expresses truncated transcendental-function series as a shared input-power
matrix multiplied by a coefficient matrix. The project compares five CUDA
engines:

1. `sfu`: CUDA fast intrinsics.
2. `sp`: standard single-precision CUDA math.
3. `wmma`: dense Tensor Core WMMA.
4. `cusparse`: the same matrix product with CSR coefficients.
5. `cusparselt`: the same matrix product with 2:4 structured coefficients.

Supported functions are `exp`, `ln1p`, `sin`, `cos`, `tan`, `sinh`, `cosh`,
and `tanh`. One fused call may contain up to four functions from either the
even-power group `{sin,cos,tan,sinh,cosh,tanh}` or the mixed-power group
`{exp,ln1p}`.

### SFU support is intentionally narrower

The SFU engine accepts only functions with an explicit fast intrinsic in this
project:

| Function | SFU implementation |
|---|---|
| `exp` | `__expf` |
| `ln1p` | `__logf(1+x)` |
| `sin` | `__sinf` |
| `cos` | `__cosf` |
| `tan` | `__tanf` |

`sinh`, `cosh`, and `tanh` are not SFU-supported. A fused SFU request is
accepted only when every requested function appears in the table. The engine
returns a clear error otherwise; it never substitutes an SP function. SP and
the three matrix engines support all eight implemented functions.

## Shared matrix design

Every matrix engine uses one canonical logical coefficient matrix `B` with
shape `16 x 16`:

- Tile/power matrix `A` packs several input values per row.
- Each input slot owns `terms` K positions.
- Function columns use a stable physical order; triple fusion uses
  `cos`, `sin`, then `tan`, matching TACT.
- WMMA consumes row-major physical `B` (`B` itself in dense mode, or the
  paired K permutation in `sparse24` mode).
- cuSPARSE consumes `CSR(B^T)` and computes `B^T * A^T`.
- cuSPARSELt stores `B` as its structured operand and uses transposed
  row-major descriptors.

For `sparse24`, the K positions are permuted so every coefficient column has at
most two nonzeros in each group of four. The identical permutation is applied
to `A`, so the mathematical product is unchanged. The implementation retains
both canonical logical `B` and physical `B24`, plus a full logical-K to
physical-K permutation. Debug mode verifies that their dot products agree.

## Build

WMMA and cuSPARSE:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
ctest --test-dir build --output-on-failure
```

CUDA `-O0` is enabled by default for stable TACT stress measurements. Use
`-DTACT_CUDA_O0=OFF` for optimized clean-mode experiments. Fast math is applied
only to `src/sfu_kernels.cu`; it never affects SP or the matrix engines.

cuSPARSELt:

```bash
export CUSPARSELT_ROOT=/path/to/cusparselt
cmake -S . -B build-slt -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DTACT_ENABLE_CUSPARSELT=ON
cmake --build build-slt -j
```

cuSPARSELt requires Ampere or newer sparse Tensor Cores.

### Reference source builds

The sources under `tact/` remain the behavioral reference. Compile all six
reference programs with:

```bash
bash scripts/compile_reference_tact.sh
```

The equivalent commands use `-O0 -arch=sm_86`; only SFU receives
`--use_fast_math`:

```bash
cd tact
nvcc -O0 -arch=sm_86 cosine_tensor_core.cu -o cosine_tensor_core
nvcc -O0 -arch=sm_86 TripleTranscendCore.cu -o TripleTranscendCore
nvcc -O0 -arch=sm_86 --use_fast_math cosine_SFU.cu -o cosine_SFU
nvcc -O0 -arch=sm_86 --use_fast_math TripleTranscendSFU.cu -o TripleTranscendSFU
nvcc -O0 -arch=sm_86 cosine_SP.cu -o cosine_SP
nvcc -O0 -arch=sm_86 TripleTranscendSP.cu -o TripleTranscendSP
```

## TACT and clean modes

`cos` and `sin,cos,tan` automatically select the canonical three-term,
2,000-step TACT workload unless a different term count or `--clean` is given.
The explicit form is:

```bash
./build/tact_bench --engine wmma --layout dense \
  --functions sin,cos,tan --terms 3 --tact --steps 2000 \
  --n 10000000 --iterations 50 --no-validate
```

The scalar TACT kernels retain the original chained behavior and global writes:

- SFU: `sin`, `D=cos`, then `tan` from the updated D.
- SP: `sin`, `tan`, then `D=cos`.
- Both use 64-thread blocks and write during every stress step.

WMMA initializes its accumulator to one and repeats MMA `steps` times. Sparse
engines compute the equivalent aggregate `1 + steps*(A*B)` by passing
`alpha=steps` to their sparse matmul and applying the same post-processing.
This preserves the logical TACT result without issuing thousands of library
kernel launches.

Use clean mode for a normal one-pass approximation and validation:

```bash
./build/tact_bench --engine wmma --clean --terms 5 \
  --functions sin,cos,tan --n 1000000 --iterations 20 --threshold 1e-3
./build/tact_bench --engine cusparse --layout sparse24 --clean --terms 5 \
  --functions sin,cos,tan --n 1000000 --iterations 20 --threshold 1e-3
./build-slt/tact_bench --engine cusparselt --layout sparse24 --clean --terms 5 \
  --functions sin,cos,tan --n 1000000 --iterations 20 --threshold 5e-3
```

Single, dual, triple, and quad clean fusion are supported:

```bash
./build/tact_bench --engine wmma --clean --terms 5 --functions cos --threshold 1e-3
./build/tact_bench --engine wmma --clean --terms 5 --functions sin,cos --threshold 1e-3
./build/tact_bench --engine wmma --clean --terms 5 --functions sin,cos,tan --threshold 1e-3
./build/tact_bench --engine wmma --clean --terms 3 --functions sin,cos,tan,sinh --threshold 1e-3
```

The same four shapes are supported by SP, WMMA, cuSPARSE, and cuSPARSELt.
SFU supports the first three; the example quad is rejected because `sinh` has
no supported fast intrinsic. Requests mixing the even and mixed power groups,
such as `sin,exp`, are rejected rather than computed with an invalid shared A.

`N` is the number of input values per engine run. A run requesting `F`
functions produces exactly `N * F` values in function-major output layout.

## Examples by engine

All commands below can be copied from the repository root. Use `--clean`
when comparing mathematical outputs and `--tact` when reproducing the old
TACT stress workload. The `--terms` option controls the matrix approximation;
SP and SFU evaluate the requested CUDA functions directly but accept the same
option so one command configuration can be shared by all engines.

### SP

SP uses standard CUDA single-precision functions such as `sinf`, `cosf`,
`tanf`, and `sinhf`. It supports every implemented function and every clean
fusion size.

```bash
# Single cosine
./build/tact_bench --engine sp --clean --functions cos --terms 5 \
  --n 1000000 --warmup 5 --iterations 20 --validate

# Dual sine and cosine
./build/tact_bench --engine sp --clean --functions sin,cos --terms 5 \
  --n 1000000 --warmup 5 --iterations 20 --validate

# Triple sine, cosine, and tangent
./build/tact_bench --engine sp --clean --functions sin,cos,tan --terms 5 \
  --n 1000000 --warmup 5 --iterations 20 --validate

# Quad including standard sinhf
./build/tact_bench --engine sp --clean --functions sin,cos,tan,sinh --terms 5 \
  --n 1000000 --warmup 5 --iterations 20 --validate

# Old TripleTranscendSP-style stress kernel
./build/tact_bench --engine sp --tact --functions sin,cos,tan --terms 3 \
  --steps 2000 --n 10000000 --warmup 1 --iterations 50 --no-validate
```

### SFU

SFU uses only the explicitly supported fast intrinsics: `__expf`,
`__logf(1+x)`, `__sinf`, `__cosf`, and `__tanf`. There is no standard-math
fallback inside this engine.

```bash
# Single fast cosine
./build/tact_bench --engine sfu --clean --functions cos --terms 5 \
  --n 1000000 --warmup 5 --iterations 20 --validate

# Supported mixed-basis pair
./build/tact_bench --engine sfu --clean --functions exp,ln1p --terms 5 \
  --min -0.1 --max 0.1 --n 1000000 --iterations 20 --validate

# Supported triple fast-intrinsic case
./build/tact_bench --engine sfu --clean --functions sin,cos,tan --terms 5 \
  --n 1000000 --warmup 5 --iterations 20 --validate

# Old TripleTranscendSFU-style stress kernel
./build/tact_bench --engine sfu --tact --functions sin,cos,tan --terms 3 \
  --steps 2000 --n 10000000 --warmup 1 --iterations 50 --no-validate
```

For example, this request is intentionally rejected because `sinh` has no
supported SFU intrinsic in TACT:

```bash
./build/tact_bench --engine sfu --clean --functions sin,cos,tan,sinh
# Error: SFU engine does not support function sinh
```

### Tensor Core / WMMA

The `wmma` engine builds the powers and coefficient tile inside one fused CUDA
kernel and executes the series through `wmma::mma_sync`. Dense layout is the
canonical old-TACT arrangement.

```bash
# Single Tensor Core cosine
./build/tact_bench --engine wmma --layout dense --clean \
  --functions cos --terms 5 --n 1000000 --iterations 20 \
  --threshold 1e-3 --validate

# Dual Tensor Core fusion
./build/tact_bench --engine wmma --layout dense --clean \
  --functions sin,cos --terms 5 --n 1000000 --iterations 20 \
  --threshold 1e-3 --validate

# Triple Tensor Core fusion
./build/tact_bench --engine wmma --layout dense --clean \
  --functions sin,cos,tan --terms 5 --n 1000000 --iterations 20 \
  --threshold 1e-3 --validate

# Quad Tensor Core fusion
./build/tact_bench --engine wmma --layout dense --clean \
  --functions sin,cos,tan,sinh --terms 5 --n 1000000 \
  --iterations 20 --threshold 1e-3 --validate

# Old TripleTranscendCore-style stress kernel: 1 + 2000 * (A x B)
./build/tact_bench --engine wmma --layout dense --tact \
  --functions sin,cos,tan --terms 3 --steps 2000 \
  --n 10000000 --warmup 1 --iterations 50 --no-validate
```

WMMA can also consume the paired physical K permutation for layout debugging:

```bash
./build/tact_bench --engine wmma --layout sparse24 --clean \
  --functions sin,cos,tan --terms 5 --n 64 --warmup 0 --iterations 1 \
  --threshold 1e-3 --debug --validate
```

### cuSPARSE

The `cusparse` engine builds the same physical power matrix as WMMA, converts
the coefficient operand to `CSR(B^T)`, and computes `B^T * A^T`. It is included
in the regular `build/tact_bench` binary. Both K orderings are supported: use
`dense` for sequential K rows or `sparse24` for the same permutation used by
cuSPARSELt.

```bash
# Dense logical K ordering
./build/tact_bench --engine cusparse --layout dense --clean \
  --functions sin,cos,tan --terms 5 --n 1000000 \
  --iterations 20 --threshold 1e-3 --validate

# 2:4-compatible physical K ordering, still stored as ordinary CSR
./build/tact_bench --engine cusparse --layout sparse24 --clean \
  --functions sin,cos,tan,sinh --terms 5 --n 1000000 \
  --iterations 20 --threshold 1e-3 --validate

# Sparse-library version of the canonical triple stress workload
./build/tact_bench --engine cusparse --layout sparse24 --tact \
  --functions sin,cos,tan --terms 3 --steps 2000 \
  --n 10000000 --warmup 1 --iterations 50 --no-validate
```

### cuSPARSELt

cuSPARSELt must be enabled at build time and requires `--layout sparse24`.
TACT verifies the generated 2:4 matrix, compresses it during engine setup, and
uses sparse Tensor Cores for the structured matrix multiplication.

```bash
# Single structured-sparse cosine
./build-slt/tact_bench --engine cusparselt --layout sparse24 --clean \
  --functions cos --terms 5 --n 1000000 \
  --iterations 20 --threshold 5e-3 --validate

# Dual structured-sparse fusion
./build-slt/tact_bench --engine cusparselt --layout sparse24 --clean \
  --functions sin,cos --terms 5 --n 1000000 \
  --iterations 20 --threshold 5e-3 --validate

# Triple structured-sparse fusion
./build-slt/tact_bench --engine cusparselt --layout sparse24 --clean \
  --functions sin,cos,tan --terms 5 --n 1000000 \
  --iterations 20 --threshold 5e-3 --validate

# Quad structured-sparse fusion
./build-slt/tact_bench --engine cusparselt --layout sparse24 --clean \
  --functions sin,cos,tan,sinh --terms 5 --n 1000000 \
  --iterations 20 --threshold 5e-3 --validate

# Structured-sparse version of the triple stress workload
./build-slt/tact_bench --engine cusparselt --layout sparse24 --tact \
  --functions sin,cos,tan --terms 3 --steps 2000 \
  --n 10000000 --warmup 1 --iterations 50 --no-validate
```

### Direct five-engine triple comparison

These commands use identical clean-mode inputs and iteration counts. The
regular engines come from `build/tact_bench`; only cuSPARSELt uses the
separately configured binary.

```bash
./build/tact_bench --engine sp --layout dense --clean \
  --functions sin,cos,tan --terms 5 --n 10000000 --iterations 50 --validate

./build/tact_bench --engine sfu --layout dense --clean \
  --functions sin,cos,tan --terms 5 --n 10000000 --iterations 50 --validate

./build/tact_bench --engine wmma --layout dense --clean \
  --functions sin,cos,tan --terms 5 --n 10000000 --iterations 50 \
  --threshold 1e-3 --validate

./build/tact_bench --engine cusparse --layout sparse24 --clean \
  --functions sin,cos,tan --terms 5 --n 10000000 --iterations 50 \
  --threshold 1e-3 --validate

./build-slt/tact_bench --engine cusparselt --layout sparse24 --clean \
  --functions sin,cos,tan --terms 5 --n 10000000 --iterations 50 \
  --threshold 5e-3 --validate
```

For the historical performance ordering, replace the clean options
(`--clean --terms 5 --validate`) with
`--tact --terms 3 --steps 2000 --no-validate` in the five commands. Keep WMMA
on `dense`; the sparse engines may use `sparse24`.

## Timing

Every measured iteration reports:

- `HtoD avg ms`: input transfer.
- `kernel avg ms`: all engine work required to produce outputs.
- `DtoH avg ms`: one synchronous copy per physical output group.
- `total avg ms`: HtoD + kernel + DtoH.
- `init ms`: reusable coefficients, descriptors, compression, and workspace.

Matrix-engine kernel timing includes power construction, matmul, post-processing,
and writes. Section profiling reports those phases separately and can be disabled
with `--no-profile-sections`.

## Validation and debugging

Clean mode validates every output and reports global and per-function
absolute/relative errors and failure counts. SFU is the reference only when it
supports the complete function set; otherwise validation automatically uses
SP. TACT mode disables validation by default because its scalar chain and
matrix accumulation are intentionally different workloads.

For layout inspection:

```bash
./build/tact_bench --engine wmma --clean --functions sin,cos,tan \
  --terms 3 --n 64 --iterations 1 --warmup 0 --debug \
  --debug-values 8 --debug-rows 2 --threshold 1e-3
```

Debug output includes inputs, function order, logical and physical slot-to-K
mapping, logical A/B/C, physical A24/B24, the maximum logical/physical dot
difference, outputs, references, errors, CSR entries, and 2:4 validity.

## Sweeps and plots

```bash
bash scripts/run_benchmarks.sh
python3 scripts/plot_results.py results/benchmark_results.csv --outdir results/plots
```

Plot generation requires Python 3 and `matplotlib`; the benchmark sweep itself
has no Python dependency.

The sweep covers single, dual, triple, and quad fusion across SP, SFU, WMMA,
cuSPARSE, and cuSPARSELt. Its CSV records `ok`, `validation_failed`, `failed`,
`unsupported`, and `unavailable` rows and links each row to a log. Unsupported
SFU entries are never plotted as timings.

Configuration is controlled by `BIN`, `BIN_SLT`, `OUTDIR`, `CSV`, `N`,
`ITERATIONS`, `WARMUP`, `INNER_REPEAT`, `TERMS`, `THRESHOLD`,
`CUSPARSELT_THRESHOLD`, `INPUT_MIN`, `INPUT_MAX`, `VALIDATE`, `MODE`,
`TACT_STEPS`, and `STRICT`. Defaults use clean mode and the conservative range
`[-pi/4, pi/4]`. The plotting script produces latency, throughput, SFU/SP
speedup, section-time plots, and a status summary.

## Limitations

- Taylor accuracy depends on the input interval and number of terms.
- FP16 powers and coefficients add rounding error.
- Tangent error grows rapidly near its poles.
- cuSPARSELt may be slower than cuSPARSE for a tiny coefficient matrix.
- cuSPARSELt output is FP16 in the supported descriptor configuration, whereas
  WMMA and cuSPARSE accumulate into FP32 outputs.
- SFU results exist only for the five explicitly supported fast intrinsics.
- Incompatible even/mixed power groups cannot share one fused Tile A.
- cuSPARSELt requires a supported GPU and an installed cuSPARSELt library.
