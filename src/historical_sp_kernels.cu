#include "tact/kernels.cuh"

namespace tact {
namespace {

// Keep these kernels structurally identical to cosine_SP.cu and
// TripleTranscendSP.cu. The triple SP ordering intentionally differs from
// SFU: sin and tan use the old D, then cosine updates D.
__global__ void historicalSpCos2000(float* d_D, int n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size_t(n)) {
    for (int j = 0; j < 2000; ++j) {
      d_D[idx] = cosf(d_D[idx] + j);
    }
  }
}

__global__ void historicalSpTriple2000(float* d_D, float* d_sin_list,
                                       float* d_tan_list, int n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size_t(n)) {
    for (int j = 0; j < 2000; ++j) {
      d_sin_list[idx] = sinf(d_D[idx] + j);
      d_tan_list[idx] = tanf(d_D[idx] + j);
      d_D[idx] = cosf(d_D[idx] + j);
    }
  }
}

__global__ void historicalSpCosDynamic(float* d_D, int n, int repeats) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size_t(n)) {
    for (int j = 0; j < repeats; ++j) {
      d_D[idx] = cosf(d_D[idx] + j);
    }
  }
}

__global__ void historicalSpTripleDynamic(float* d_D, float* d_sin_list,
                                          float* d_tan_list, int n,
                                          int repeats) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size_t(n)) {
    for (int j = 0; j < repeats; ++j) {
      d_sin_list[idx] = sinf(d_D[idx] + j);
      d_tan_list[idx] = tanf(d_D[idx] + j);
      d_D[idx] = cosf(d_D[idx] + j);
    }
  }
}

}  // namespace

void launchHistoricalSp(float* d_D, float* d_out, int num_funcs, int n,
                        int reference_repeat, cudaStream_t stream) {
  constexpr int block = 64;
  int grid = (n + block - 1) / block;
  if (num_funcs == 1) {
    if (reference_repeat == 2000) {
      historicalSpCos2000<<<grid, block, 0, stream>>>(d_D, n);
    } else {
      historicalSpCosDynamic<<<grid, block, 0, stream>>>(
          d_D, n, reference_repeat);
    }
  } else {
    float* d_sin_list = d_out;
    float* d_tan_list = d_out + size_t(2) * n;
    if (reference_repeat == 2000) {
      historicalSpTriple2000<<<grid, block, 0, stream>>>(
          d_D, d_sin_list, d_tan_list, n);
    } else {
      historicalSpTripleDynamic<<<grid, block, 0, stream>>>(
          d_D, d_sin_list, d_tan_list, n, reference_repeat);
    }
  }
  TACT_CUDA_CHECK(cudaGetLastError());
}

}  // namespace tact
