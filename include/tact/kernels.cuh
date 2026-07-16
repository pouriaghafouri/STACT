#pragma once
#include "tact/common.cuh"

namespace tact {

void launchSfu(float* d_in, float* d_out, const int* d_funcs,
               int num_funcs, int n, bool tact_mode, int tact_steps,
               cudaStream_t stream = nullptr);
void launchSp(float* d_in, float* d_out, const int* d_funcs,
              int num_funcs, int n, bool tact_mode, int tact_steps,
              cudaStream_t stream = nullptr);

void launchWmma(const float* d_in, float* d_out, const int* d_funcs,
                int num_funcs, int n, int terms, int ipr, BasisKind basis,
                const int* d_positions, const int* d_output_groups,
                const __half* d_coefficients, bool tact_mode, int tact_steps,
                cudaStream_t stream = nullptr);

void launchBuildPowerT(const float* d_in, __half* d_pT, int n, int rows_pad,
                       int terms, int ipr, BasisKind basis,
                       const int* d_positions, cudaStream_t stream = nullptr);

void launchPostFloat(const float* d_ct, const float* d_in, float* d_out,
                     const int* d_funcs, int num_funcs, int n, int rows_pad,
                     int ipr, const int* d_output_groups, bool tact_mode,
                     cudaStream_t stream = nullptr);

}  // namespace tact
namespace tact {
void launchBuildPowerRowsChunk(const float* d_in, __half* d_pRows, int n,
                               int row_start, int chunk_rows_pad, int terms,
                               int ipr, BasisKind basis, const int* d_positions,
                               cudaStream_t stream = nullptr);
void launchPostHalfChunk(const __half* d_ct, const float* d_in, float* d_out,
                         const int* d_funcs, int num_funcs, int n,
                         int row_start, int chunk_rows_pad, int ipr,
                         const int* d_output_groups, bool tact_mode,
                         cudaStream_t stream = nullptr);
}  // namespace tact
