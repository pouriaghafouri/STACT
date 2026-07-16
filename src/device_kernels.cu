#include "tact/kernels.cuh"
#include <mma.h>

namespace tact {
using namespace nvcuda;

__global__ void tactWmmaKernel(const float* __restrict__ in,
                               float* __restrict__ out,
                               const int* __restrict__ funcs,
                               int num_funcs,
                               int n,
                               int terms,
                               int ipr,
                               BasisKind basis,
                               const int* __restrict__ positions,
                               const int* __restrict__ output_groups,
                               const __half* __restrict__ coefficients,
                               int tact_mode,
                               int tact_steps) {
  __shared__ __half sharedB[WMMA_K * WMMA_N];
  __shared__ __half sharedA[WARPS_PER_BLOCK * WMMA_M * WMMA_K];
  __shared__ float sharedC[WARPS_PER_BLOCK * WMMA_M * WMMA_N];

  for (int i = threadIdx.x; i < WARPS_PER_BLOCK * WMMA_M * WMMA_K; i += blockDim.x) sharedA[i] = __float2half(0.0f);
  for (int i = threadIdx.x; i < WARPS_PER_BLOCK * WMMA_M * WMMA_N; i += blockDim.x) sharedC[i] = 0.0f;
  __syncthreads();

  for (int i = threadIdx.x; i < WMMA_K * WMMA_N; i += blockDim.x) {
    sharedB[i] = coefficients[i];
  }
  __syncthreads();

  int warp_id = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x & (WARP_SIZE - 1);
  int global_warp = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  int base_row = global_warp * WMMA_M;
  __half* myA = sharedA + warp_id * WMMA_M * WMMA_K;
  float* myC = sharedC + warp_id * WMMA_M * WMMA_N;

  if (lane < WMMA_M) {
    int row = lane;
    int logical_row = base_row + row;
    for (int s = 0; s < ipr; ++s) {
      int idx = logical_row * ipr + s;
      float x = (idx < n) ? in[idx] : 0.0f;
      for (int t = 0; t < terms; ++t) {
        int k = positions[s * terms + t];
        float p = (idx < n) ? powerFor(x, t, basis) : 0.0f;
        myA[row * WMMA_K + k] = __float2half(p);
      }
    }
  }
  __syncthreads();

  if (base_row * ipr < n) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    // TACT mode seeds every accumulator element with one and repeats the same
    // MMA. Clean mode performs one product and adds leading terms afterward.
    wmma::fill_fragment(c_frag, tact_mode ? 1.0f : 0.0f);
    wmma::load_matrix_sync(a_frag, myA, WMMA_K);
    wmma::load_matrix_sync(b_frag, sharedB, WMMA_N);
    int mma_repeats = tact_mode ? tact_steps : 1;
    for (int repeat = 0; repeat < mma_repeats; ++repeat) {
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    wmma::store_matrix_sync(myC, c_frag, WMMA_N, wmma::mem_row_major);
  }
  __syncthreads();

  int total_local = WMMA_M * ipr;
  for (int local = lane; local < total_local; local += WARP_SIZE) {
    int row = local / ipr;
    int s = local % ipr;
    int idx = (base_row + row) * ipr + s;
    if (idx >= n) continue;
    float x = in[idx];
    for (int f = 0; f < num_funcs; ++f) {
      FunctionId fid = static_cast<FunctionId>(funcs[f]);
      int col = output_groups[f] * ipr + s;
      float residual = myC[row * WMMA_N + col];
      out[size_t(f) * n + idx] = tact_mode
          ? (needsMultiplyByX(fid) ? residual * x : residual)
          : postprocess(fid, residual, x);
    }
  }
}

__global__ void buildPowerMatrixTransposeKernel(const float* __restrict__ in,
                                                __half* __restrict__ pT,
                                                int n,
                                                int rows_pad,
                                                int terms,
                                                int ipr,
                                                BasisKind basis,
                                                const int* __restrict__ positions) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  int row = idx / ipr;
  int slot = idx % ipr;
  float x = in[idx];
  for (int t = 0; t < terms; ++t) {
    int k = positions[slot * terms + t];
    pT[k * rows_pad + row] = __float2half(powerFor(x, t, basis));
  }
}

__global__ void postprocessTransposeFloatKernel(const float* __restrict__ cT,
                                                const float* __restrict__ in,
                                                float* __restrict__ out,
                                                const int* __restrict__ funcs,
                                                int num_funcs,
                                                int n,
                                                int rows_pad,
                                                int ipr,
                                                const int* __restrict__ output_groups,
                                                int tact_mode) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  int row = idx / ipr;
  int slot = idx % ipr;
  float x = in[idx];
  for (int f = 0; f < num_funcs; ++f) {
    FunctionId fid = static_cast<FunctionId>(funcs[f]);
    int out_col = output_groups[f] * ipr + slot;
    float residual = cT[out_col * rows_pad + row];
    out[size_t(f) * n + idx] = tact_mode
        ? (needsMultiplyByX(fid) ? (residual + 1.0f) * x : residual + 1.0f)
        : postprocess(fid, residual, x);
  }
}

void launchWmma(const float* d_in, float* d_out, const int* d_funcs,
                int num_funcs, int n, int terms, int ipr, BasisKind basis,
                const int* d_positions, const int* d_output_groups,
                const __half* d_coefficients, bool tact_mode, int tact_steps,
                cudaStream_t stream) {
  int rows = (n + ipr - 1) / ipr;
  int warp_tiles = (rows + WMMA_M - 1) / WMMA_M;
  int grid = (warp_tiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
  tactWmmaKernel<<<grid, THREADS_PER_BLOCK, 0, stream>>>(
      d_in, d_out, d_funcs, num_funcs, n, terms, ipr, basis, d_positions,
      d_output_groups, d_coefficients, tact_mode ? 1 : 0, tact_steps);
  TACT_CUDA_CHECK(cudaGetLastError());
}

void launchBuildPowerT(const float* d_in, __half* d_pT, int n, int rows_pad,
                       int terms, int ipr, BasisKind basis,
                       const int* d_positions, cudaStream_t stream) {
  TACT_CUDA_CHECK(cudaMemsetAsync(d_pT, 0, size_t(WMMA_K) * rows_pad * sizeof(__half), stream));
  int block = 256;
  int grid = (n + block - 1) / block;
  buildPowerMatrixTransposeKernel<<<grid, block, 0, stream>>>(d_in, d_pT, n, rows_pad, terms, ipr, basis, d_positions);
  TACT_CUDA_CHECK(cudaGetLastError());
}

void launchPostFloat(const float* d_ct, const float* d_in, float* d_out,
                     const int* d_funcs, int num_funcs, int n, int rows_pad,
                     int ipr, const int* d_output_groups, bool tact_mode,
                     cudaStream_t stream) {
  int block = 256;
  int grid = (n + block - 1) / block;
  postprocessTransposeFloatKernel<<<grid, block, 0, stream>>>(
      d_ct, d_in, d_out, d_funcs, num_funcs, n, rows_pad, ipr,
      d_output_groups, tact_mode ? 1 : 0);
  TACT_CUDA_CHECK(cudaGetLastError());
}

}  // namespace tact

namespace tact {

__global__ void buildPowerRowsChunkKernel(const float* __restrict__ in,
                                          __half* __restrict__ pRows,
                                          int n,
                                          int row_start,
                                          int chunk_rows_pad,
                                          int terms,
                                          int ipr,
                                          BasisKind basis,
                                          const int* __restrict__ positions) {
  int local_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int max_values = chunk_rows_pad * ipr;
  if (local_idx >= max_values) return;
  int local_row = local_idx / ipr;
  int slot = local_idx % ipr;
  int global_idx = (row_start + local_row) * ipr + slot;
  if (global_idx >= n) return;
  float x = in[global_idx];
  for (int t = 0; t < terms; ++t) {
    int k = positions[slot * terms + t];
    pRows[local_row * WMMA_K + k] = __float2half(powerFor(x, t, basis));
  }
}

__global__ void postprocessHalfChunkKernel(const __half* __restrict__ cT,
                                           const float* __restrict__ in,
                                           float* __restrict__ out,
                                           const int* __restrict__ funcs,
                                           int num_funcs,
                                           int n,
                                           int row_start,
                                           int chunk_rows_pad,
                                           int ipr,
                                           const int* __restrict__ output_groups,
                                           int tact_mode) {
  int local_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int max_values = chunk_rows_pad * ipr;
  if (local_idx >= max_values) return;
  int local_row = local_idx / ipr;
  int slot = local_idx % ipr;
  int global_idx = (row_start + local_row) * ipr + slot;
  if (global_idx >= n) return;
  float x = in[global_idx];
  for (int f = 0; f < num_funcs; ++f) {
    FunctionId fid = static_cast<FunctionId>(funcs[f]);
    int out_col = output_groups[f] * ipr + slot;
    float residual = __half2float(cT[out_col * chunk_rows_pad + local_row]);
    out[size_t(f) * n + global_idx] = tact_mode
        ? (needsMultiplyByX(fid) ? (residual + 1.0f) * x : residual + 1.0f)
        : postprocess(fid, residual, x);
  }
}

void launchBuildPowerRowsChunk(const float* d_in, __half* d_pRows, int n,
                               int row_start, int chunk_rows_pad, int terms,
                               int ipr, BasisKind basis, const int* d_positions,
                               cudaStream_t stream) {
  TACT_CUDA_CHECK(cudaMemsetAsync(d_pRows, 0, size_t(chunk_rows_pad) * WMMA_K * sizeof(__half), stream));
  int block = 256;
  int grid = (chunk_rows_pad * ipr + block - 1) / block;
  buildPowerRowsChunkKernel<<<grid, block, 0, stream>>>(d_in, d_pRows, n, row_start, chunk_rows_pad, terms, ipr, basis, d_positions);
  TACT_CUDA_CHECK(cudaGetLastError());
}

void launchPostHalfChunk(const __half* d_ct, const float* d_in, float* d_out,
                         const int* d_funcs, int num_funcs, int n,
                         int row_start, int chunk_rows_pad, int ipr,
                         const int* d_output_groups, bool tact_mode,
                         cudaStream_t stream) {
  int block = 256;
  int grid = (chunk_rows_pad * ipr + block - 1) / block;
  postprocessHalfChunkKernel<<<grid, block, 0, stream>>>(
      d_ct, d_in, d_out, d_funcs, num_funcs, n, row_start,
      chunk_rows_pad, ipr, d_output_groups, tact_mode ? 1 : 0);
  TACT_CUDA_CHECK(cudaGetLastError());
}

}  // namespace tact
