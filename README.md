# TACT benchmark project

TACT rewrites truncated Taylor/Maclaurin approximations as products between a
shared input-power tile and sparse coefficient columns. It benchmarks those
products against scalar GPU transcendental functions using:

1. `baseline_fast`: CUDA fast intrinsics / SFU-style path when available.
2. `baseline_sp`: CUDA libdevice single-precision path.
3. `wmma`: TACT-style Tensor Core MMA with dense 16x16 WMMA tiles.
4. `cusparse`: generic cuSPARSE SpMM using a sparse coefficient matrix.
5. `cusparselt`: cuSPARSELt 2:4 structured-sparse Tensor Core path.

Supported functions:

```text
exp, ln1p, sin, cos, tan, sinh, cosh, tanh
```

For one fused call (up to four functions), use either only the even-power group:

```text
sin, cos, tan, sinh, cosh, tanh
```

or only the mixed-power group:

```text
exp, ln1p
```

## Build

WMMA/cuSPARSE build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
```

`TACT_REFERENCE_OPT_LEVEL=ON` is the default and appends CUDA `-O0`, matching
the historical TACT compilation model used for the SFU/SP/WMMA stress
comparison. Configure with `-DTACT_REFERENCE_OPT_LEVEL=OFF` for optimized clean
one-pass experiments.

cuSPARSELt build:

```bash
export CUSPARSELT_ROOT=/path/to/cusparselt
cmake -S . -B build-slt -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DTACT_ENABLE_CUSPARSELT=ON
cmake --build build-slt -j
```

cuSPARSELt requires Ampere or newer sparse Tensor Cores. The code checks this at runtime.

### Standalone historical references

The old sources under `tact/` deliberately use `-O0`; only SFU references use
fast math. Compile them in the original modes with:

```bash
cd tact
nvcc -O0 -arch=sm_86 cosine_tensor_core.cu -o cosine_tensor_core
nvcc -O0 -arch=sm_86 TripleTranscendCore.cu -o TripleTranscendCore
nvcc -O0 -arch=sm_86 --use_fast_math cosine_SFU.cu -o cosine_SFU
nvcc -O0 -arch=sm_86 --use_fast_math TripleTranscendSFU.cu -o TripleTranscendSFU
nvcc -O0 -arch=sm_86 cosine_SP.cu -o cosine_SP
nvcc -O0 -arch=sm_86 TripleTranscendSP.cu -o TripleTranscendSP
```

Equivalently, `bash scripts/compile_reference_tact.sh` writes all six binaries
to `build-reference/`. The regular CMake target intentionally does not enable
`--use_fast_math` globally: `baseline_fast` calls fast intrinsics explicitly,
while `baseline_sp` retains standard single-precision libdevice behavior. The
historical SFU and SP kernels are also kept in separate source files so only
the historical SFU file receives `--use_fast_math`. Relocatable device code is
disabled, matching the standalone old builds and their generated kernel
resource profiles.

## Example runs

Baseline fast/SFU-style:

```bash
./build/tact_bench --engine baseline_fast --functions sin,cos,tan --n 10000000 --iterations 50
./build/tact_bench --engine baseline_sp --functions sin,cos,tan --n 10000000 --iterations 50
```

WMMA dense:

```bash
./build/tact_bench --engine wmma --layout dense --functions sin,cos,tan \
  --n 10000000 --iterations 50
```

The three commands above automatically select the old three-term, 2000-step
TACT workload so their kernel times are directly comparable. Use `--clean`
with an explicit term count for a one-pass approximation/validation run.

cuSPARSE generic sparse:

```bash
./build/tact_bench --engine cusparse --layout sparse24 --functions sin,cos,tan \
  --terms 5 --n 10000000 --iterations 20 --threshold 1e-3
```

cuSPARSELt 2:4 structured sparse:

```bash
./build-slt/tact_bench --engine cusparselt --layout sparse24 --functions sin,cos,tan \
  --terms 5 --n 10000000 --iterations 50 --threshold 1e-3
```

For small `n`, use `--inner-repeat` to reduce timing noise:

```bash
./build/tact_bench --engine baseline_fast --functions sin --n 100000 \
  --iterations 50 --inner-repeat 100
```

The printed `outputs/it` already includes `inner_repeat`.

Single, dual, triple, and quad fusion examples are respectively:

```bash
./build/tact_bench --engine wmma --clean --terms 5 --functions cos --n 4096 --iterations 3
./build/tact_bench --engine wmma --clean --terms 5 --functions sin,cos --n 4096 --iterations 3
./build/tact_bench --engine wmma --clean --terms 5 --functions sin,cos,tan --n 4096 --iterations 3
./build/tact_bench --engine wmma --clean --terms 5 --functions sin,cos,tan,sinh --n 4096 --iterations 3
```

`N` is the number of input `x` values processed by each engine run. With `F`
requested functions, each run writes `N * F` outputs; a timed iteration writes
`N * F * inner_repeat` outputs.

### Historical TACT benchmark mode

The standalone references perform a 2000-step internal stress workload. The
scalar kernels chain their result from one loop step to the next and perform
their global stores inside every loop iteration. The old triple kernels also
use intentionally different evaluation orders: SFU evaluates `sin`, updates D
with `cos`, then evaluates `tan` from the updated D; SP evaluates `sin` and
`tan` from the old D, then updates D with `cos`. The old WMMA kernels initialize
C to one and accumulate the same MMA 2000 times. For `cos` and `sin,cos,tan`,
baseline SFU/SP and WMMA select this mode automatically when no non-reference
term count is supplied. It can also be selected explicitly:

```bash
./build/tact_bench --engine baseline_fast --functions cos --terms 3 \
  --tact-reference --legacy-repeat 2000 --no-validate
./build/tact_bench --engine baseline_sp --functions sin,cos,tan --terms 3 \
  --tact-reference --legacy-repeat 2000 --no-validate
./build/tact_bench --engine wmma --layout dense --functions sin,cos,tan --terms 3 \
  --tact-reference --legacy-repeat 2000 --no-validate
```

Use `--clean` for normal one-pass mathematical evaluation. Cross-validating the
historical WMMA stress accumulator against the chained historical SFU workload
is not a mathematical correctness test, so `--no-validate` is recommended for
historical timing reproduction and is selected automatically in implicit TACT
mode. TACT mode also restores the reference initial values:
scalar kernels use `pi*(2*i+1)`, single-cosine WMMA uses `5`, and triple WMMA
uses `1`.

## Validation

Validation is on by default in clean mode. Every method compares every output with
`baseline_fast`. The report includes global and per-function maximum absolute
error, maximum relative error, and threshold failure counts. A non-zero failure
count returns exit code `2`.

Implicit historical TACT mode disables validation because its WMMA accumulation
and chained scalar workload intentionally do not compute the same values. Pass
`--validate` explicitly only when inspecting that expected difference.

In clean mode, use `--no-validate` for raw timing sweeps.

The default ranges are conservative for FP16 powers and truncated polynomials.
For layout checks involving tangent, a useful explicit interval is
`--min -0.785398 --max 0.785398`; adjust `--threshold` for the chosen term count.

## Debugging matrix construction

`--debug` dumps input/output/reference samples, absolute errors, slot-to-K
mapping, logical A/B/C matrices, and final outputs. Sparse engines additionally
show CSR metadata or a structured 2:4 validity report. Control its size or send
it to a file with:

```bash
./build/tact_bench --engine wmma --functions sin,cos,tan --terms 3 --n 64 \
  --iterations 1 --warmup 0 --debug --debug-values 8 --debug-rows 2
./build/tact_bench --engine cusparse --layout sparse24 --functions sin,cos,tan \
  --n 64 --iterations 1 --warmup 0 --debug --debug-file debug.txt
```

## Benchmark sweep and plots

The sweep covers single, dual, triple, and quad cases. Validation exit code `2`
is recorded and does not stop the sweep unless `STRICT=1`; other errors still
stop it. If `build-slt/tact_bench` exists, cuSPARSELt is included automatically.

```bash
bash scripts/run_benchmarks.sh
python3 scripts/plot_results.py results/benchmark_results.csv --outdir results/plots
```

Set `BIN_SLT`, `N`, `ITERS`, `TERMS`, or `TACT_REPEAT` in the environment to
override sweep defaults. Cosine and triple SFU/SP/WMMA cases use historical TACT
mode; dual, quad, and sparse cases use clean mode. Plotting requires Matplotlib.
Plot labels use `sfu`, `wmma (tact)`, `cuSparse`, and `cuSparseLt` (with `sp`
included as the additional SP baseline).

## Important cuSPARSELt notes

The cuSPARSELt backend uses row-major physical matrices with `opA=T` and `opB=T`, matching NVIDIA's current sample style. It stores the sparse coefficient matrix physically as `K x N`; after `opA=T`, it becomes the logical `N x K` coefficient matrix. The dense power matrix is stored physically as `rows x K`; after `opB=T`, it becomes the logical `K x rows` power matrix.

This avoids the unsupported row-major `NN` descriptor combination that can produce `operation not permitted` at `cusparseLtMatmulDescriptorInit`.

cuSPARSELt also limits each C/D dimension. The backend chunks the row dimension so large `n` values, such as 10 million inputs for triple fusion, do not exceed the descriptor limit.

## Timing and validation notes

Every measured iteration reports the historical timing breakdown:

- `HtoD avg ms`: copying the `N` input values to the GPU.
- `kernel avg ms`: the complete engine work needed to produce all outputs.
- `DtoH avg ms`: copying all `N * F` outputs back to the host; historical
  triple mode uses three separate synchronous copies in old order: cosine/D,
  sine, then tangent.
- `total avg ms`: `HtoD + kernel + DtoH`.

The kernel phase includes the full per-run execution path for each engine:

- `baseline_fast` / `baseline_sp`: one kernel that computes every requested
  function for all `N` inputs and writes all outputs. In historical mode its
  64-thread kernel performs the old stores during each of the 2000 stress steps.
- `wmma`: the fused WMMA kernel, including tile power construction, coefficient loading, MMA, post-processing, and output writes.
- `cusparse`: power-matrix construction, cuSPARSE SpMM, post-processing, and output writes.
- `cusparselt`: chunked power-matrix construction, cuSPARSELt matmul, post-processing, and output writes.

`init ms` is reported separately for one-time setup work such as descriptors,
static coefficient matrices, CSR construction, and cuSPARSELt compression. It is
not included in `total avg ms`.

Section timing is enabled by default. Use `--no-profile-sections` for lower-overhead raw timing.

The reported sections mean:

- `baseline kernel`: all scalar function evaluations and writes.
- `fused WMMA kernel`: power/tile construction, coefficient load, MMA,
  post-processing, and writes in the fused kernel.
- `build power matrix`: per-run dense input-power construction for a sparse engine.
- `matrix multiply`: SpMM or structured sparse matmul.
- `postprocess/write`: leading constants, odd-function multiply-by-x, and output writes.
- `init ms`: one-time reusable coefficient construction, descriptors, workspace,
  compression, and related setup; it is outside repeated `avg ms` timing.

Clean-mode validation is reported both globally and per function. A large error
in `sin,cos,tan` on a wide range is generally caused by the truncated tangent
series near `pi/2`. Use explicit `--min` and `--max` for controlled stress tests.

## Limitations

- Taylor accuracy depends on the requested range and number of terms; tangent
  error grows quickly on wider intervals and near its poles.
- FP16 power/coefficient storage introduces additional rounding.
- cuSPARSELt can be slower than cuSPARSE for these tiny, highly sparse
  coefficient matrices.
- cuSPARSELt requires an installed cuSPARSELt library and supported NVIDIA GPU
  architecture (Ampere/SM80 or newer for this implementation).
