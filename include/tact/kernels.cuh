#pragma once
#include "tact/common.cuh"

namespace tact {

// Scalar kernels preserve the old TACT stateful stress order. d_in is also the
// cosine state/output, exactly like cosine_SFU and TripleTranscendSFU.
void launchSfu(float* d_in, float* d_out, int num_funcs, int n,
               int stress_repeat, cudaStream_t stream = nullptr);
void launchSp(float* d_in, float* d_out, int num_funcs, int n,
              int stress_repeat, cudaStream_t stream = nullptr);

// Dense WMMA kernels build/load A and B once, initialize C to one, and repeat
// only mma_sync. The final store/postprocess happens once after the loop.
void launchWmma(const float* d_in, float* d_out, const int* d_funcs,
                int num_funcs, int n, int terms, int ipr, BasisKind basis,
                const int* d_positions, const int* d_output_groups,
                const __half* d_coefficients, int stress_repeat,
                cudaStream_t stream = nullptr);

void launchCanonicalWmma(const float* d_in, float* d_out, int num_funcs,
                         int n, int terms, const __half* d_coefficients,
                         int stress_repeat,
                         cudaStream_t stream = nullptr);

void launchBuildPowerT(const float* d_in, __half* d_pT, int n, int rows_pad,
                       int terms, int ipr, BasisKind basis,
                       const int* d_positions, cudaStream_t stream = nullptr);

void launchBuildPowerRowsChunk(const float* d_in, __half* d_pRows, int n,
                               int row_start, int chunk_rows_pad, int terms,
                               int ipr, BasisKind basis,
                               const int* d_positions,
                               cudaStream_t stream = nullptr);

// Sparse paths initialize C once to the same all-ones accumulator state used
// by the WMMA fragment, then repeat only the library matmul with beta=1.
void launchFillFloatOnes(float* d_values, size_t count,
                         cudaStream_t stream = nullptr);
void launchFillHalfOnes(__half* d_values, size_t count,
                        cudaStream_t stream = nullptr);

void launchPostFloat(const float* d_ct, const float* d_in, float* d_out,
                     const int* d_funcs, int num_funcs, int n, int rows_pad,
                     int ipr, const int* d_output_groups,
                     cudaStream_t stream = nullptr);

void launchPostHalfChunk(const __half* d_ct, const float* d_in, float* d_out,
                         const int* d_funcs, int num_funcs, int n,
                         int row_start, int chunk_rows_pad, int ipr,
                         const int* d_output_groups,
                         cudaStream_t stream = nullptr);

}  // namespace tact
