# STACT implementation contract

The SP, SFU, and WMMA implementations follow the reference programs under
`tact/`. The project has one old-TACT-compatible stress parameter:

```text
stress-repeat = 2000
```

## Scalar paths

SP and SFU use the stateful reference loop and function-major output storage.
SFU is limited to the function sets implemented by the reference SFU kernels.

## WMMA path

WMMA uses `m16n16k16`. Tile A and Tile B are prepared once, the accumulator is
initialized to one, and `wmma::mma_sync` is repeated inside the kernel.

## cuSPARSE path

WMMA logically computes:

```text
C = A * B
```

cuSPARSE uses the explicit `csr` layout and the transposed equivalent:

```text
C^T = B^T * A^T
```

`B^T` is stored in CSR. Only the coefficient rows and K rows containing useful
values are included in the cuSPARSE descriptors.

The old recurrence repeats an identical linear product:

```text
C_R = 1 + R * (B^T * A^T)
```

The cuSPARSE backend evaluates that exact final expression with one
`cusparseSpMM` call using `alpha=R` and `beta=1`. Reissuing SpMM R times would
force the complete result matrix through global memory for every repetition,
which is not analogous to WMMA retaining C in accumulator fragments.

## Timing

A measured iteration contains:

1. input H2D copy
2. one engine execution
3. D2H copy of every output plane

Validation and the optional section-profile pass are outside the measured
latency.

## Mandatory validation

Validation cannot be disabled.

- SP/SFU: full deterministic device replay
- WMMA: full independent host FP16-rounded polynomial model
- cuSPARSE: full host model plus full WMMA parity comparison

Any validation or parity failure returns exit code 2.
