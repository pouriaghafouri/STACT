#include "tact/kernels.cuh"

namespace tact {
namespace {

__device__ float evaluateSfu(FunctionId f, float x) {
  switch (f) {
    case F_EXP: return __expf(x);
    case F_LOG1P: return __logf(1.0f + x);
    case F_SIN: return __sinf(x);
    case F_COS: return __cosf(x);
    case F_TAN: return __tanf(x);
    case F_SINH:
    case F_COSH:
    case F_TANH:
      break;
  }
  return 0.0f;
}

__global__ void sfuCleanKernel(const float* __restrict__ in,
                               float* __restrict__ out,
                               const int* __restrict__ funcs,
                               int num_funcs, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  float x = in[idx];
  for (int f = 0; f < num_funcs; ++f) {
    out[size_t(f) * n + idx] =
        evaluateSfu(static_cast<FunctionId>(funcs[f]), x);
  }
}

__global__ void sfuCosTactKernel(float* d_D, int n, int steps) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size_t(n)) {
    for (int j = 0; j < steps; ++j) {
      d_D[idx] = __cosf(d_D[idx] + j);
    }
  }
}

__global__ void sfuTripleTactKernel(float* d_D, float* d_sin_list,
                                    float* d_tan_list, int n, int steps) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size_t(n)) {
    for (int j = 0; j < steps; ++j) {
      d_sin_list[idx] = __sinf(d_D[idx] + j);
      d_D[idx] = __cosf(d_D[idx] + j);
      d_tan_list[idx] = __tanf(d_D[idx] + j);
    }
  }
}

}  // namespace

void launchSfu(float* d_in, float* d_out, const int* d_funcs,
               int num_funcs, int n, bool tact_mode, int tact_steps,
               cudaStream_t stream) {
  if (!tact_mode) {
    constexpr int block = 256;
    int grid = (n + block - 1) / block;
    sfuCleanKernel<<<grid, block, 0, stream>>>(
        d_in, d_out, d_funcs, num_funcs, n);
  } else {
    constexpr int block = 64;
    int grid = (n + block - 1) / block;
    if (num_funcs == 1) {
      sfuCosTactKernel<<<grid, block, 0, stream>>>(d_in, n, tact_steps);
    } else {
      sfuTripleTactKernel<<<grid, block, 0, stream>>>(
          d_in, d_out, d_out + size_t(2) * n, n, tact_steps);
    }
  }
  TACT_CUDA_CHECK(cudaGetLastError());
}

}  // namespace tact
