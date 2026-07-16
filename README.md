# STACT — Sparse Tensor Accelerated Computation of Transcendentals

STACT is a CUDA research project for comparing several GPU implementations of transcendental functions:

* **SP**: standard single-precision CUDA math functions
* **SFU**: fast intrinsic/SFU-style CUDA functions
* **Tensor Core**: TACT implementation based on WMMA
* **cuSPARSE**: sparse matrix implementation of the same TACT polynomial formulation
* **cuSPARSELt**: 2:4 structured-sparse Tensor Core implementation

The Tensor Core implementation follows the kernel and stress-test style of the reference programs in the `tact/` directory.

---

## Supported benchmark cases

The main benchmark cases are:

| Case   | Functions          |
| ------ | ------------------ |
| Single | `cos`              |
| Dual   | `sin,cos`          |
| Triple | `sin,cos,tan`      |
| Quad   | `sin,cos,tan,sinh` |

The old TACT-compatible default polynomial configuration is:

```text
terms = 3
```

---

## Stress-test semantics

This project intentionally follows the stress-test style of the reference TACT kernels.

The stress loop is applied only to the core computation.

### SP and SFU

Only the transcendental function evaluations are repeated.

### Tensor Core

Tile construction and fragment loading are performed once. Only the following operation is repeated:

```cpp
wmma::mma_sync(...)
```

### cuSPARSE

The power matrix is constructed once and postprocessing is performed once. Only the sparse matrix multiplication is repeated:

```cpp
cusparseSpMM(...)
```

### cuSPARSELt

Matrix descriptors, compression, workspace allocation, power-matrix construction, and postprocessing are outside the stress loop. Only the structured sparse multiplication is repeated:

```cpp
cusparseLtMatmul(...)
```

The default TACT-style stress count is:

```text
stress-repeat = 2000
```

There is no separate clean benchmark mode in the intended TACT-compatible experiment.

---

## Requirements

* Linux
* CMake 3.18 or newer
* CUDA Toolkit
* GPU supporting the requested CUDA architecture
* cuSPARSE
* cuSPARSELt for the cuSPARSELt engine

The example build commands target:

```text
sm_86
```

---

## Build the regular project

The regular build contains:

* SP
* SFU
* Tensor Core / WMMA
* cuSPARSE

Run:

```bash
rm -rf build

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86

cmake --build build -j

ctest --test-dir build --output-on-failure
```

The main executable is:

```text
build/tact_bench
```

---

## Build with cuSPARSELt

The project expects a cuSPARSELt installation containing:

```text
include/cusparseLt.h
lib/libcusparseLt.so
```

When using the cuSPARSELt copy included in this project:

```bash
rm -rf build-slt

cmake -S . -B build-slt \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DTACT_ENABLE_CUSPARSELT=ON \
  -DCUSPARSELT_ROOT="$PWD/.deps/cusparselt-0.4"

cmake --build build-slt -j

ctest --test-dir build-slt --output-on-failure
```

The cuSPARSELt-enabled executable is:

```text
build-slt/tact_bench
```

If the runtime cannot find `libcusparseLt.so`, run:

```bash
export LD_LIBRARY_PATH="$PWD/.deps/cusparselt-0.4/lib:${LD_LIBRARY_PATH:-}"
```

---

## Reference TACT compilation

The standalone reference programs in `tact/` should be compiled using the following styles.

### Tensor Core reference programs

```bash
nvcc -O0 -arch=sm_86 tensor_core.cu -o tensor_core
```

### SFU reference programs

```bash
nvcc -O0 -arch=sm_86 --use_fast_math SFU.cu -o SFU
```

### SP reference programs

```bash
nvcc -O0 -arch=sm_86 SP.cu -o SP
```

Use the actual source filenames in the `tact/` directory.

---

# Triple-function benchmark

The triple-function workload computes:

```text
sin(x), cos(x), tan(x)
```

The following commands use:

```text
N              = 100000000
terms          = 3
stress-repeat  = 2000
iterations     = 1
input range    = [-π/4, π/4]
threshold      = 5e-3
```

Because `N=100000000` and `stress-repeat=2000` create a very large workload, start with one benchmark iteration.

---

## SP

```bash
./build/tact_bench \
  --engine baseline_sp \
  --functions sin,cos,tan \
  --terms 3 \
  --n 100000000 \
  --iterations 1 \
  --stress-repeat 2000 \
  --threshold 5e-3 \
  --min -0.78539816339 \
  --max 0.78539816339
```

---

## SFU

```bash
./build/tact_bench \
  --engine baseline_fast \
  --functions sin,cos,tan \
  --terms 3 \
  --n 100000000 \
  --iterations 1 \
  --stress-repeat 2000 \
  --threshold 5e-3 \
  --min -0.78539816339 \
  --max 0.78539816339
```

---

## Tensor Core / WMMA

```bash
./build/tact_bench \
  --engine wmma \
  --layout dense \
  --functions sin,cos,tan \
  --terms 3 \
  --n 100000000 \
  --iterations 1 \
  --stress-repeat 2000 \
  --threshold 5e-3 \
  --min -0.78539816339 \
  --max 0.78539816339
```

---

## cuSPARSE

```bash
./build/tact_bench \
  --engine cusparse \
  --layout sparse24 \
  --functions sin,cos,tan \
  --terms 3 \
  --n 100000000 \
  --iterations 1 \
  --stress-repeat 2000 \
  --threshold 5e-3 \
  --min -0.78539816339 \
  --max 0.78539816339
```

---

## cuSPARSELt

```bash
./build-slt/tact_bench \
  --engine cusparselt \
  --layout sparse24 \
  --functions sin,cos,tan \
  --terms 3 \
  --n 100000000 \
  --iterations 1 \
  --stress-repeat 2000 \
  --threshold 5e-3 \
  --min -0.78539816339 \
  --max 0.78539816339
```

---

# Copy-paste all triple-function benchmarks

```bash
echo "================ SP ================"

./build/tact_bench \
  --engine baseline_sp \
  --functions sin,cos,tan \
  --terms 3 \
  --n 100000000 \
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
  --n 100000000 \
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
  --n 100000000 \
  --iterations 1 \
  --stress-repeat 2000 \
  --threshold 5e-3 \
  --min -0.78539816339 \
  --max 0.78539816339

echo "============= cuSPARSE ============="

./build/tact_bench \
  --engine cusparse \
  --layout sparse24 \
  --functions sin,cos,tan \
  --terms 3 \
  --n 100000000 \
  --iterations 1 \
  --stress-repeat 2000 \
  --threshold 5e-3 \
  --min -0.78539816339 \
  --max 0.78539816339

echo "============ cuSPARSELt ============"

./build-slt/tact_bench \
  --engine cusparselt \
  --layout sparse24 \
  --functions sin,cos,tan \
  --terms 3 \
  --n 100000000 \
  --iterations 1 \
  --stress-repeat 2000 \
  --threshold 5e-3 \
  --min -0.78539816339 \
  --max 0.78539816339
```

---

## Increase the number of measured iterations

After confirming that all five implementations run successfully, increase:

```bash
--iterations 1
```

to:

```bash
--iterations 3
```

or:

```bash
--iterations 5
```

Using multiple iterations provides a more stable average but increases the total execution time significantly.

---

## Validation

Validation reports:

* maximum absolute error
* maximum relative error
* total number of failures
* per-function errors where available

The Tensor Core and sparse methods use FP16 operands and may require a looser threshold than SP or SFU.

For the triple-function experiment, the recommended initial threshold is:

```text
5e-3
```

The input interval is restricted to:

```text
[-π/4, π/4]
```

because low-order Taylor approximations, especially for tangent, become less accurate over wider intervals.

---

## Meaning of `N`

`N` is the number of input values processed in one benchmark iteration.

For the triple-function workload:

```text
functions = 3
N         = 100000000
```

the benchmark produces:

```text
3 × 100000000 = 300000000
```

function output values per iteration.

---

## Memory warning for `N=100000000`

A triple-function output array alone requires approximately:

```text
3 × 100000000 × 4 bytes ≈ 1.2 GB
```

The input array requires approximately:

```text
100000000 × 4 bytes ≈ 400 MB
```

Tensor Core and sparse engines also allocate intermediate power and result matrices. Therefore, `N=100000000` may exceed the available VRAM on GPUs with 4 GB or 6 GB of memory, especially for cuSPARSE.

If an execution reports an out-of-memory error, first test with:

```bash
--n 10000000
```

and then increase `N` gradually.

---

## Important notes

* Use `terms=3` for the old TACT-compatible experiment.
* The stress loop repeats only the core operation.
* Do not use a separate `--clean` mode for this experiment.
* SFU should only be used for functions explicitly supported by the SFU engine.
* cuSPARSE and cuSPARSELt must compute the same logical polynomial as the Tensor Core implementation.
* cuSPARSELt requires a valid 2:4 structured sparse coefficient layout.
* cuSPARSE or cuSPARSELt may be slower than WMMA for very small coefficient matrices because library-call and sparse-format overhead can dominate.
