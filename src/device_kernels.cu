#include "tact/kernels.cuh"
#include <mma.h>

namespace tact {
using namespace nvcuda;

__device__ inline float fastBaseline(FunctionId f, float x) {
  switch (f) {
    case F_EXP: return __expf(x);
    case F_LOG1P: return __logf(1.0f + x);
    case F_SIN: return __sinf(x);
    case F_COS: return __cosf(x);
    case F_TAN: return __sinf(x) / __cosf(x);
    case F_SINH: return sinhf(x);
    case F_COSH: return coshf(x);
    case F_TANH: return tanhf(x);
  }
  return 0.0f;
}

__device__ inline float spBaseline(FunctionId f, float x) {
  switch (f) {
    case F_EXP: return expf(x);
    case F_LOG1P: return log1pf(x);
    case F_SIN: return sinf(x);
    case F_COS: return cosf(x);
    case F_TAN: return tanf(x);
    case F_SINH: return sinhf(x);
    case F_COSH: return coshf(x);
    case F_TANH: return tanhf(x);
  }
  return 0.0f;
}

__global__ void baselineKernel(const float* __restrict__ in,
                               float* __restrict__ out,
                               const int* __restrict__ funcs,
                               int num_funcs,
                               int n,
                               int use_fast) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  float x = in[idx];
  for (int f = 0; f < num_funcs; ++f) {
    FunctionId fid = static_cast<FunctionId>(funcs[f]);
    out[size_t(f) * n + idx] = use_fast ? fastBaseline(fid, x) : spBaseline(fid, x);
  }
}

__global__ void tactWmmaKernel(const float* __restrict__ in,
                               float* __restrict__ out,
                               const int* __restrict__ funcs,
                               int num_funcs,
                               int n,
                               int terms,
                               int ipr,
                               BasisKind basis,
                               const int* __restrict__ positions) {
  __shared__ __half sharedB[WMMA_K * WMMA_N];
  __shared__ __half sharedA[WARPS_PER_BLOCK * WMMA_M * WMMA_K];
  __shared__ float sharedC[WARPS_PER_BLOCK * WMMA_M * WMMA_N];

  for (int i = threadIdx.x; i < WMMA_K * WMMA_N; i += blockDim.x) sharedB[i] = __float2half(0.0f);
  for (int i = threadIdx.x; i < WARPS_PER_BLOCK * WMMA_M * WMMA_K; i += blockDim.x) sharedA[i] = __float2half(0.0f);
  for (int i = threadIdx.x; i < WARPS_PER_BLOCK * WMMA_M * WMMA_N; i += blockDim.x) sharedC[i] = 0.0f;
  __syncthreads();

  for (int linear = threadIdx.x; linear < num_funcs * ipr * terms; linear += blockDim.x) {
    int t = linear % terms;
    int tmp = linear / terms;
    int s = tmp % ipr;
    int f = tmp / ipr;
    int k = positions[s * terms + t];
    int col = f * ipr + s;
    FunctionId fid = static_cast<FunctionId>(funcs[f]);
    sharedB[k * WMMA_N + col] = __float2half(coeffFor(fid, t));
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
    wmma::fill_fragment(c_frag, 0.0f);
    wmma::load_matrix_sync(a_frag, myA, WMMA_K);
    wmma::load_matrix_sync(b_frag, sharedB, WMMA_N);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
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
      int col = f * ipr + s;
      float residual = myC[row * WMMA_N + col];
      out[size_t(f) * n + idx] = postprocess(fid, residual, x);
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

__global__ void buildPowerRowsKernel(const float* __restrict__ in,
                                     __half* __restrict__ pRows,
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
    pRows[row * WMMA_K + k] = __float2half(powerFor(x, t, basis));
  }
}

__global__ void postprocessTransposeFloatKernel(const float* __restrict__ cT,
                                                const float* __restrict__ in,
                                                float* __restrict__ out,
                                                const int* __restrict__ funcs,
                                                int num_funcs,
                                                int n,
                                                int rows_pad,
                                                int ipr) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  int row = idx / ipr;
  int slot = idx % ipr;
  float x = in[idx];
  for (int f = 0; f < num_funcs; ++f) {
    FunctionId fid = static_cast<FunctionId>(funcs[f]);
    int out_col = f * ipr + slot;
    float residual = cT[out_col * rows_pad + row];
    out[size_t(f) * n + idx] = postprocess(fid, residual, x);
  }
}

__global__ void postprocessTransposeHalfKernel(const __half* __restrict__ cT,
                                               const float* __restrict__ in,
                                               float* __restrict__ out,
                                               const int* __restrict__ funcs,
                                               int num_funcs,
                                               int n,
                                               int rows_pad,
                                               int ipr) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  int row = idx / ipr;
  int slot = idx % ipr;
  float x = in[idx];
  for (int f = 0; f < num_funcs; ++f) {
    FunctionId fid = static_cast<FunctionId>(funcs[f]);
    int out_col = f * ipr + slot;
    float residual = __half2float(cT[out_col * rows_pad + row]);
    out[size_t(f) * n + idx] = postprocess(fid, residual, x);
  }
}

void launchBaseline(const float* d_in, float* d_out, const int* d_funcs,
                    int num_funcs, int n, bool fast, cudaStream_t stream) {
  int block = 256;
  int grid = (n + block - 1) / block;
  baselineKernel<<<grid, block, 0, stream>>>(d_in, d_out, d_funcs, num_funcs, n, fast ? 1 : 0);
  TACT_CUDA_CHECK(cudaGetLastError());
}

void launchWmma(const float* d_in, float* d_out, const int* d_funcs,
                int num_funcs, int n, int terms, int ipr, BasisKind basis,
                const int* d_positions, cudaStream_t stream) {
  int rows = (n + ipr - 1) / ipr;
  int warp_tiles = (rows + WMMA_M - 1) / WMMA_M;
  int grid = (warp_tiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
  tactWmmaKernel<<<grid, THREADS_PER_BLOCK, 0, stream>>>(d_in, d_out, d_funcs, num_funcs, n, terms, ipr, basis, d_positions);
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

void launchBuildPowerRows(const float* d_in, __half* d_pRows, int n, int rows_pad,
                          int terms, int ipr, BasisKind basis,
                          const int* d_positions, cudaStream_t stream) {
  TACT_CUDA_CHECK(cudaMemsetAsync(d_pRows, 0, size_t(rows_pad) * WMMA_K * sizeof(__half), stream));
  int block = 256;
  int grid = (n + block - 1) / block;
  buildPowerRowsKernel<<<grid, block, 0, stream>>>(d_in, d_pRows, n, rows_pad, terms, ipr, basis, d_positions);
  TACT_CUDA_CHECK(cudaGetLastError());
}

void launchPostFloat(const float* d_ct, const float* d_in, float* d_out,
                     const int* d_funcs, int num_funcs, int n, int rows_pad,
                     int ipr, cudaStream_t stream) {
  int block = 256;
  int grid = (n + block - 1) / block;
  postprocessTransposeFloatKernel<<<grid, block, 0, stream>>>(d_ct, d_in, d_out, d_funcs, num_funcs, n, rows_pad, ipr);
  TACT_CUDA_CHECK(cudaGetLastError());
}

void launchPostHalf(const __half* d_ct, const float* d_in, float* d_out,
                    const int* d_funcs, int num_funcs, int n, int rows_pad,
                    int ipr, cudaStream_t stream) {
  int block = 256;
  int grid = (n + block - 1) / block;
  postprocessTransposeHalfKernel<<<grid, block, 0, stream>>>(d_ct, d_in, d_out, d_funcs, num_funcs, n, rows_pad, ipr);
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
                                           int ipr) {
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
    int out_col = f * ipr + slot;
    float residual = __half2float(cT[out_col * chunk_rows_pad + local_row]);
    out[size_t(f) * n + global_idx] = postprocess(fid, residual, x);
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
                         cudaStream_t stream) {
  int block = 256;
  int grid = (chunk_rows_pad * ipr + block - 1) / block;
  postprocessHalfChunkKernel<<<grid, block, 0, stream>>>(d_ct, d_in, d_out, d_funcs, num_funcs, n, row_start, chunk_rows_pad, ipr);
  TACT_CUDA_CHECK(cudaGetLastError());
}

}  // namespace tact
