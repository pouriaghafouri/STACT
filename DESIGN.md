# TACT design notes

## Components

- `src/sfu_kernels.cu`: SFU clean and TACT scalar kernels; this is the only
  translation unit compiled with `--use_fast_math`.
- `src/sp_kernels.cu`: standard single-precision clean and TACT scalar kernels.
- `src/scalar_engine.cu`: SFU/SP engine dispatch and profiling.
- `src/wmma_engine.cu`: dense Tensor Core engine and coefficient ownership.
- `src/cusparse_engine.cu`: generic sparse CSR matrix engine.
- `src/cusparselt_engine.cu`: 2:4 structured-sparse Tensor Core engine.
- `src/matrix_builders.cu`: canonical coefficient matrix and CSR conversion.
- `src/device_kernels.cu`: WMMA, power construction, and post-processing.
- `src/runner.cu`: transfers, timing, validation, and debug output.

## One logical matrix product

`buildCoefficientLayout` is the sole coefficient-construction path. It returns
canonical row-major `B[K,16]`, physical `B24[K,16]`, and a complete K-row
permutation. All engines compute the same product using different physical
representations:

| Engine | Physical coefficient representation | Product |
|---|---|---|
| WMMA | physical dense `B[K,16]` | `A * B` |
| cuSPARSE | `CSR(Bphysical^T)[16,K]` | `Bphysical^T * Aphysical^T` |
| cuSPARSELt | structured row-major `B24[K,16]`, `opA=T` | `B24^T * A24^T` |

`ProblemSpec::positions` maps each `(input slot, term)` to K. Dense mode uses
sequential positions. The 2:4 mode permutes them so each coefficient column has
at most two nonzeros per four K elements. Power construction uses the same map,
making the permutation algebraically neutral.

Debug mode constructs logical and physical A/B side by side, evaluates both
dot products, prints the full permutation, and reports their maximum difference.

Triple fusion uses five inputs per A row and three powers per input. Coefficient
column groups are cosine, sine, and tangent. Sine and tangent multiply their
factored result by the original input.

## TACT workload

The canonical scalar kernels preserve their chained 64-thread stress loops and
in-loop global stores. SFU performs `sin`, `D=cos`, `tan(updated D)`; SP performs
`sin`, `tan`, `D=cos`.

WMMA initializes C to one and performs `steps` repeated MMA operations. Sparse
libraries express the identical aggregate with matmul `alpha=steps`, then the
common post-processing adds the leading one and applies odd-function multiply.
This avoids measuring thousands of API/kernel-launch boundaries instead of the
sparse matrix operation itself.

## Timing boundary

Each measured iteration synchronously measures HtoD, complete engine execution,
and DtoH. Matrix execution contains power construction, matrix multiplication,
post-processing, and output writes. Static coefficient conversion, descriptors,
compression, and workspace allocation are reported as initialization.

## Scalar capability and validation

SFU supports only `__expf`, `__logf`, `__sinf`, `__cosf`, and `__tanf` paths.
Hyperbolic requests are rejected before launch. Clean validation uses SFU when
the entire fused request is supported and otherwise selects SP, so unsupported
SFU work is never mislabeled as an SFU reference or timing.
