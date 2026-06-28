# TACT project design notes

The project is split by execution engine:

- `src/baseline_engine.cu`: scalar baseline kernels. `baseline_fast` uses CUDA fast intrinsics/SFU-style functions where available; `baseline_sp` uses regular libdevice functions.
- `src/wmma_engine.cu`: dense WMMA Tensor Core path.
- `src/cusparse_engine.cu`: generic sparse CSR coefficient matrix multiplied by a dense power matrix.
- `src/cusparselt_engine.cu`: 2:4 structured-sparse Tensor Core path using cuSPARSELt.
- `src/device_kernels.cu`: shared CUDA kernels for power-matrix construction and post-processing.
- `src/matrix_builders.cu`: coefficient matrix construction and sparse layouts.
- `src/runner.cu`: benchmark orchestration, timing, and validation.

For every timed iteration, the benchmark executes all work needed to produce the outputs for that iteration. For matrix engines this includes power-matrix construction and post-processing, not just the multiply. Static setup and compression are reported separately as `init ms`.

The `sparse24` K layout permutes the power positions so each logical coefficient column has at most two nonzeros in every group of four K entries. The same K permutation is applied to the power matrix and the coefficient matrix, so the mathematical result is unchanged.
