#include "tact/kernels.cuh"
#include <mma.h>

namespace tact {
using namespace nvcuda;

template <int NumFuncs, int Terms>
__global__ void canonicalWmmaKernel(const float* __restrict__ in,
                                    float* __restrict__ out,
                                    int n,
                                    const __half* __restrict__ coefficients,
                                    int stress_repeat) {
  constexpr int IprByK = WMMA_K / Terms;
  constexpr int IprByN = WMMA_N / NumFuncs;
  constexpr int Ipr = IprByK < IprByN ? IprByK : IprByN;

  __shared__ __half sharedB[WMMA_K * WMMA_N];
  __shared__ __half sharedA[WARPS_PER_BLOCK * WMMA_M * WMMA_K];
  __shared__ float sharedC[WARPS_PER_BLOCK * WMMA_M * WMMA_N];

  for (int i = threadIdx.x;
       i < WARPS_PER_BLOCK * WMMA_M * WMMA_K; i += blockDim.x) {
    sharedA[i] = __float2half(0.0f);
  }
  for (int i = threadIdx.x; i < WMMA_K * WMMA_N; i += blockDim.x) {
    sharedB[i] = coefficients[i];
  }
  __syncthreads();

  int warp_id = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x & (WARP_SIZE - 1);
  int global_warp = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  int base_input = global_warp * WMMA_M * Ipr;
  __half* myA = sharedA + warp_id * WMMA_M * WMMA_K;
  float* myC = sharedC + warp_id * WMMA_M * WMMA_N;

  if (lane < WMMA_M) {
    int row = lane;
#pragma unroll
    for (int slot = 0; slot < Ipr; ++slot) {
      int idx = base_input + row * Ipr + slot;
      if (idx < n) {
        float x = in[idx];
#pragma unroll
        for (int term = 0; term < Terms; ++term) {
          myA[row * WMMA_K + slot * Terms + term] =
              __float2half(powerFor(x, term, BASIS_EVEN));
        }
      }
    }
  }
  __syncthreads();

  if (base_input < n) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                   float> c_frag;
    wmma::fill_fragment(c_frag, 1.0f);
    wmma::load_matrix_sync(a_frag, myA, WMMA_K);
    wmma::load_matrix_sync(b_frag, sharedB, WMMA_N);
    for (int repeat = 0; repeat < stress_repeat; ++repeat) {
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    wmma::store_matrix_sync(
        myC, c_frag, WMMA_N, wmma::mem_row_major);
  }
  __syncthreads();

  int local_values = WMMA_M * Ipr;
  for (int local = lane; local < local_values; local += WARP_SIZE) {
    int row = local / Ipr;
    int slot = local % Ipr;
    int idx = base_input + local;
    if (idx >= n) continue;
    float x = in[idx];
    if constexpr (NumFuncs == 1) {
      float cosine = myC[row * WMMA_N + slot];
      out[idx] = cosine;
    } else {
      // TripleTranscendCore stores cosine, sine, tangent column groups while
      // the public output order remains sine, cosine, tangent.
      float cosine = myC[row * WMMA_N + slot];
      float sine = myC[row * WMMA_N + Ipr + slot];
      float tangent = myC[row * WMMA_N + 2 * Ipr + slot];
      out[idx] = sine * x;
      out[size_t(n) + idx] = cosine;
      out[size_t(2) * n + idx] = tangent * x;
    }
  }
}

template <int NumFuncs, int Terms>
void launchCanonicalWmmaTerms(const float* d_in, float* d_out, int n,
                              const __half* d_coefficients,
                              int stress_repeat, cudaStream_t stream) {
  constexpr int IprByK = WMMA_K / Terms;
  constexpr int IprByN = WMMA_N / NumFuncs;
  constexpr int Ipr = IprByK < IprByN ? IprByK : IprByN;
  int rows = (n + Ipr - 1) / Ipr;
  int warp_tiles = (rows + WMMA_M - 1) / WMMA_M;
  int grid = (warp_tiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
  canonicalWmmaKernel<NumFuncs, Terms>
      <<<grid, THREADS_PER_BLOCK, 0, stream>>>(
          d_in, d_out, n, d_coefficients, stress_repeat);
}

template <int NumFuncs>
void dispatchCanonicalWmmaTerms(const float* d_in, float* d_out, int n,
                                int terms, const __half* d_coefficients,
                                int stress_repeat,
                                cudaStream_t stream) {
  switch (terms) {
    case 1: launchCanonicalWmmaTerms<NumFuncs, 1>(d_in, d_out, n, d_coefficients, stress_repeat, stream); break;
    case 2: launchCanonicalWmmaTerms<NumFuncs, 2>(d_in, d_out, n, d_coefficients, stress_repeat, stream); break;
    case 3: launchCanonicalWmmaTerms<NumFuncs, 3>(d_in, d_out, n, d_coefficients, stress_repeat, stream); break;
    case 4: launchCanonicalWmmaTerms<NumFuncs, 4>(d_in, d_out, n, d_coefficients, stress_repeat, stream); break;
    case 5: launchCanonicalWmmaTerms<NumFuncs, 5>(d_in, d_out, n, d_coefficients, stress_repeat, stream); break;
    case 6: launchCanonicalWmmaTerms<NumFuncs, 6>(d_in, d_out, n, d_coefficients, stress_repeat, stream); break;
    case 7: launchCanonicalWmmaTerms<NumFuncs, 7>(d_in, d_out, n, d_coefficients, stress_repeat, stream); break;
    case 8: launchCanonicalWmmaTerms<NumFuncs, 8>(d_in, d_out, n, d_coefficients, stress_repeat, stream); break;
    default: throw std::runtime_error("canonical WMMA terms must be in [1,8]");
  }
  TACT_CUDA_CHECK(cudaGetLastError());
}

void launchCanonicalWmma(const float* d_in, float* d_out, int num_funcs,
                         int n, int terms, const __half* d_coefficients,
                         int stress_repeat,
                         cudaStream_t stream) {
  if (num_funcs == 1) {
    dispatchCanonicalWmmaTerms<1>(d_in, d_out, n, terms, d_coefficients,
                                  stress_repeat, stream);
  } else if (num_funcs == 3) {
    dispatchCanonicalWmmaTerms<3>(d_in, d_out, n, terms, d_coefficients,
                                  stress_repeat, stream);
  } else {
    throw std::runtime_error(
        "canonical WMMA path supports only cosine or sin,cos,tan");
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
                               const int* __restrict__ positions,
                               const int* __restrict__ output_groups,
                               const __half* __restrict__ coefficients,
                               int stress_repeat) {
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
    // Match the reference TACT kernel: seed C with one, then repeat
    // only the same MMA operation. A/B construction and postprocess occur once.
    wmma::fill_fragment(c_frag, 1.0f);
    wmma::load_matrix_sync(a_frag, myA, WMMA_K);
    wmma::load_matrix_sync(b_frag, sharedB, WMMA_N);
    for (int repeat = 0; repeat < stress_repeat; ++repeat) {
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
      out[size_t(f) * n + idx] =
          needsMultiplyByX(fid) ? residual * x : residual;
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
                                                const int* __restrict__ output_groups) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  int row = idx / ipr;
  int slot = idx % ipr;
  float x = in[idx];
  for (int f = 0; f < num_funcs; ++f) {
    FunctionId fid = static_cast<FunctionId>(funcs[f]);
    int out_col = output_groups[f] * ipr + slot;
    float residual = cT[out_col * rows_pad + row];
    out[size_t(f) * n + idx] =
        needsMultiplyByX(fid) ? residual * x : residual;
  }
}

__global__ void fillFloatOnesKernel(float* values, size_t count) {
  const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < count) values[idx] = 1.0f;
}

__global__ void fillHalfOnesKernel(__half* values, size_t count) {
  const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < count) values[idx] = __float2half(1.0f);
}

void launchFillFloatOnes(float* d_values, size_t count, cudaStream_t stream) {
  constexpr int block = 256;
  const int grid = int((count + block - 1) / block);
  fillFloatOnesKernel<<<grid, block, 0, stream>>>(d_values, count);
  TACT_CUDA_CHECK(cudaGetLastError());
}

void launchFillHalfOnes(__half* d_values, size_t count, cudaStream_t stream) {
  constexpr int block = 256;
  const int grid = int((count + block - 1) / block);
  fillHalfOnesKernel<<<grid, block, 0, stream>>>(d_values, count);
  TACT_CUDA_CHECK(cudaGetLastError());
}

void launchWmma(const float* d_in, float* d_out, const int* d_funcs,
                int num_funcs, int n, int terms, int ipr, BasisKind basis,
                const int* d_positions, const int* d_output_groups,
                const __half* d_coefficients, int stress_repeat,
                cudaStream_t stream) {
  int rows = (n + ipr - 1) / ipr;
  int warp_tiles = (rows + WMMA_M - 1) / WMMA_M;
  int grid = (warp_tiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
  tactWmmaKernel<<<grid, THREADS_PER_BLOCK, 0, stream>>>(
      d_in, d_out, d_funcs, num_funcs, n, terms, ipr, basis, d_positions,
      d_output_groups, d_coefficients, stress_repeat);
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
                     int ipr, const int* d_output_groups,
                     cudaStream_t stream) {
  int block = 256;
  int grid = (n + block - 1) / block;
  postprocessTransposeFloatKernel<<<grid, block, 0, stream>>>(
      d_ct, d_in, d_out, d_funcs, num_funcs, n, rows_pad, ipr,
      d_output_groups);
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
                                           const int* __restrict__ output_groups) {
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
    out[size_t(f) * n + global_idx] =
        needsMultiplyByX(fid) ? residual * x : residual;
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
                         const int* d_output_groups,
                         cudaStream_t stream) {
  int block = 256;
  int grid = (chunk_rows_pad * ipr + block - 1) / block;
  postprocessHalfChunkKernel<<<grid, block, 0, stream>>>(
      d_ct, d_in, d_out, d_funcs, num_funcs, n, row_start,
      chunk_rows_pad, ipr, d_output_groups);
  TACT_CUDA_CHECK(cudaGetLastError());
}

}  // namespace tact
