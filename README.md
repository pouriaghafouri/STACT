# TACT benchmark project

This project benchmarks transcendental-function evaluation on GPUs using:

1. `baseline_fast`: CUDA fast intrinsics / SFU-style path when available.
2. `baseline_sp`: CUDA libdevice single-precision path.
3. `wmma`: TACT-style Tensor Core MMA with dense 16x16 WMMA tiles.
4. `cusparse`: generic cuSPARSE SpMM using a sparse coefficient matrix.
5. `cusparselt`: cuSPARSELt 2:4 structured-sparse Tensor Core path.

Supported functions:

```text
exp, ln1p, sin, cos, tan, sinh, cosh, tanh
```

For one fused call, use either only the even-power group:

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
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j
```

cuSPARSELt build:

```bash
export CUSPARSELT_ROOT=/path/to/cusparselt
cmake -S . -B build-slt -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DTACT_ENABLE_CUSPARSELT=ON
cmake --build build-slt -j
```

cuSPARSELt requires Ampere or newer sparse Tensor Cores. The code checks this at runtime.

## Example runs

Baseline fast/SFU-style:

```bash
./build/tact_bench --engine baseline_fast --functions sin,cos,tan --n 10000000 --iterations 50
```

WMMA dense:

```bash
./build/tact_bench --engine wmma --layout dense --functions sin,cos,tan \
  --terms 5 --n 10000000 --iterations 50 --threshold 1e-5
```

cuSPARSE generic sparse:

```bash
./build/tact_bench --engine cusparse --layout sparse24 --functions sin,cos,tan \
  --terms 5 --n 1000000 --iterations 20 --threshold 1e-3
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

## Validation

Validation is on by default. Every method compares its output with `baseline_fast` for every requested output value. The program prints max absolute error, max relative error, and the number of failures at the configured threshold. A non-zero failure count returns exit code `2`.

Use `--no-validate` only for raw timing sweeps.

## Important cuSPARSELt notes

The cuSPARSELt backend uses row-major physical matrices with `opA=T` and `opB=T`, matching NVIDIA's current sample style. It stores the sparse coefficient matrix physically as `K x N`; after `opA=T`, it becomes the logical `N x K` coefficient matrix. The dense power matrix is stored physically as `rows x K`; after `opB=T`, it becomes the logical `K x rows` power matrix.

This avoids the unsupported row-major `NN` descriptor combination that can produce `operation not permitted` at `cusparseLtMatmulDescriptorInit`.

cuSPARSELt also limits each C/D dimension. The backend chunks the row dimension so large `n` values, such as 10 million inputs for triple fusion, do not exceed the descriptor limit.

## v4 timing and validation notes

Timing now includes the full per-run execution path for each engine:

- `baseline_fast` / `baseline_sp`: one kernel that computes every requested function for all `N` inputs and writes all outputs.
- `wmma`: the fused WMMA kernel, including tile power construction, coefficient loading, MMA, post-processing, and output writes.
- `cusparse`: power-matrix construction, cuSPARSE SpMM, post-processing, and output writes.
- `cusparselt`: chunked power-matrix construction, cuSPARSELt matmul, post-processing, and output writes.

`init ms` is reported separately for one-time setup work such as descriptors, static coefficient matrices, CSR construction, and cuSPARSELt compression. It is not included in the repeated per-iteration `avg ms`, because in the intended TACT use case those coefficient structures are reused across many input batches.

Section timing is enabled by default. Use `--no-profile-sections` for lower-overhead raw timing.

Validation is still on by default and is reported both globally and per function. A large error in `sin,cos,tan` at the old wide default range was mainly the truncated Taylor series for `tan(x)` near `pi/2`, not a sine/cosine range issue. The default input ranges are now validation-safe for `--terms 5`; use explicit `--min` and `--max` for stress tests.
