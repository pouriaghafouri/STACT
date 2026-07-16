# TACT project design notes

The project is split by execution engine:

- `src/baseline_engine.cu`: dispatch for scalar baseline kernels.
- `src/historical_sfu_kernels.cu`: old 64-thread SFU stress kernels, compiled
  with `--use_fast_math`.
- `src/historical_sp_kernels.cu`: old 64-thread standard-libdevice SP stress
  kernels, compiled without fast math.
- `src/wmma_engine.cu`: dense WMMA Tensor Core path.
- `src/cusparse_engine.cu`: generic sparse CSR coefficient matrix multiplied by a dense power matrix.
- `src/cusparselt_engine.cu`: 2:4 structured-sparse Tensor Core path using cuSPARSELt.
- `src/device_kernels.cu`: shared CUDA kernels for power-matrix construction and post-processing.
- `src/matrix_builders.cu`: coefficient matrix construction and sparse layouts.
- `src/runner.cu`: benchmark orchestration, timing, and validation.

For every timed iteration, the benchmark measures input HtoD, the full engine
kernel phase, and output DtoH separately. For matrix engines the kernel phase
includes power-matrix construction and post-processing, not just the multiply.
Static setup and compression are reported separately as `init ms`.
Historical triple DtoH uses three synchronous copies in the same order as the
old programs: cosine/D, sine, then tangent.

The `sparse24` K layout permutes the power positions so each logical coefficient column has at most two nonzeros in every group of four K entries. The same K permutation is applied to the power matrix and the coefficient matrix, so the mathematical result is unchanged.

## Reference layout and compatibility behavior

The files in `tact/` are treated as historical source material. Their two core layouts are:

- Single cosine: one warp owns a 16x16 A tile; each row packs five inputs, with three powers per input at K positions `(3s,3s+1,3s+2)`. B places the three cosine coefficients in the corresponding K rows and slot column. One warp therefore covers 80 inputs.
- Triple sin/cos/tan: A uses the same five-input, three-power packing. B uses physical column groups `cos`, `sin`, then `tan`. Sine and tangent multiply the MMA result by the original input during writeback.

The old WMMA accumulator starts at `1.0f` and executes the same MMA 2000 times.
The old scalar kernels also execute 2000 chained operations and retain their
global stores inside every stress iteration. Their triple ordering is not the
same: SFU does `sin`, `D=cos`, `tan(updated D)`, while SP does `sin`, `tan`,
`D=cos`. Historical kernels use 64-thread blocks. The build retains the old
SFU-only fast-math split, CUDA `-O0`, SM86 default, and non-RDC compilation;
the resulting triple resource profiles match the old binaries (SFU: 18
registers, SP: 31 registers, neither with a stack frame). These are stress and
timing semantics rather than a one-pass function approximation. The modular
benchmark selects these semantics automatically for historical cosine and
triple SFU/SP/WMMA comparisons; `--tact-reference` selects them explicitly,
and `--clean` selects one-pass mathematical behavior. Padding is explicitly
zeroed instead of reproducing undefined reads from old uninitialized shared memory.
