# STACT implementation contract

This project intentionally has **one benchmark workload**: the stress pattern in
`./tact/cosine_*` and `./tact/TripleTranscend*`. There is no clean/alternate
execution mode and no whole-engine inner-repeat option.

## Reference behavior

The reference CUDA programs establish the rules used by the modular benchmark:

- SP and SFU use 64 threads per block and perform a stateful loop inside one
  kernel. In the triple case each iteration computes sine, overwrites the input
  state with cosine, then computes tangent from that updated state.
- WMMA uses `m16n16k16`, eight warps per block, three polynomial terms by
  default, and five inputs per Tile-A row for the single/triple reference
  layouts. A and B are prepared once, C is filled with `1.0f`, and only
  `wmma::mma_sync` is repeated (2000 times by default).
- Output storage is function-major: `out[f*N + i]`.

Dual and quad modes are direct extensions of these rules. SFU quad is rejected
because the requested `sinh` has no approved SFU implementation in this
project. SP quad appends `sinhf` after the old triple state sequence.

## One logical matrix product

`buildCoefficientLayout()` is the single source of coefficient matrices. The
logical operation is

```text
C = A * B
```

where A contains FP16-rounded powers, B contains FP16 Taylor coefficients, and
C starts at one. Only the product is stress-repeated.

| Backend | Physical operation | Repeated work |
|---|---|---|
| WMMA | `A * B` inside one CUDA kernel | only `wmma::mma_sync` |
| cuSPARSE | `CSR(B^T) * A^T` | only `cusparseSpMM` |
| cuSPARSELt | structured `B24^T * A24^T` | only `cusparseLtMatmul` |

Power generation, C initialization, descriptors/compression, and final
post-processing are not inside the 2000-iteration stress loop.

## Dense and cuSPARSE layout

Dense layout uses sequential K positions:

```text
slot s, term t -> k = s * terms + t
```

The coefficient matrix is stored as row-major `B[K,16]`. cuSPARSE constructs
`CSR(B^T)` with shape `16 x K`, builds `A^T` as a row-major dense matrix with
shape `K x rows`, initializes the FP32 C matrix to ones once, and performs

```text
C^T <- B^T * A^T + C^T
```

for every stress repetition. The postprocess runs once after the last SpMM.

## cuSPARSELt 2:4 layout

cuSPARSELt uses a paired K permutation rather than dropping coefficients:

1. Build the canonical logical A and B.
2. Assign every `(slot, term)` to a physical K position so each coefficient
   column has at most two nonzeros in each four-element K group.
3. Apply the same K permutation to A and B.
4. Verify `A_logical * B_logical == A_physical * B_physical`.
5. Store physical B as the structured row-major `K x 16` operand and use
   `opA=T`, while the chunked physical power matrix is row-major `rows x K`
   with `opB=T`.

The C/D buffer is initialized to FP16 ones once per chunk and aliases C and D,
matching the NVIDIA sample call pattern. Compression and workspace creation are
one-time engine initialization costs.

## Timing boundary

Every measured sample contains:

1. one input reset (H2D),
2. one engine execution,
3. one copy of every requested output plane (D2H).

A separate diagnostic execution collects section timing for:

- scalar kernel or fused WMMA kernel,
- sparse power build plus one C initialization,
- repeated sparse matmul,
- one postprocess.

The main benchmark calls the uninstrumented engine path; phase-event
synchronizations therefore cannot bias the cross-method latency comparison.
Descriptor creation, CSR construction, cuSPARSELt validation/compression, and
workspace allocation are reported as `init_ms`.

## Validation and diagnostics

Matrix validation models the reference recurrence `C = 1 + R*(A*B)` with
FP16-rounded A/B values. Scalar validation reproduces the reference state
chain. Debug output shows logical/physical A and B, the K permutation, 2:4
counts, CSR entries, and intermediate cuSPARSE values for small runs.

## Expected sparse-library behavior

The coefficient matrix is only `16 x 16`. WMMA performs all 2000 MMA operations
inside one launched kernel, while cuSPARSE and cuSPARSELt require 2000 host
library calls/kernel launches to reproduce the same stress location. Therefore
a correct sparse implementation can still be slower; moving the loop onto
power construction or postprocess would change the experiment and is not
allowed.
