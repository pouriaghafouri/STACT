#include "tact/kernels.cuh"

namespace tact {
namespace {

__global__ void spCosTactKernel(float* d_D, int n, int steps) {
  const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= size_t(n)) return;
  for (int j = 0; j < steps; ++j) {
    d_D[idx] = cosf(d_D[idx] + float(j));
  }
}

__global__ void spDualTactKernel(float* d_D, float* d_sin_list,
                                 int n, int steps) {
  const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= size_t(n)) return;
  for (int j = 0; j < steps; ++j) {
    d_sin_list[idx] = sinf(d_D[idx] + float(j));
    d_D[idx] = cosf(d_D[idx] + float(j));
  }
}

__global__ void spTripleTactKernel(float* d_D, float* d_sin_list,
                                   float* d_tan_list, int n, int steps) {
  const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= size_t(n)) return;
  for (int j = 0; j < steps; ++j) {
    d_sin_list[idx] = sinf(d_D[idx] + float(j));
    d_D[idx] = cosf(d_D[idx] + float(j));
    d_tan_list[idx] = tanf(d_D[idx] + float(j));
  }
}

__global__ void spQuadTactKernel(float* d_D, float* d_sin_list,
                                 float* d_tan_list, float* d_sinh_list,
                                 int n, int steps) {
  const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= size_t(n)) return;
  for (int j = 0; j < steps; ++j) {
    d_sin_list[idx] = sinf(d_D[idx] + float(j));
    d_D[idx] = cosf(d_D[idx] + float(j));
    d_tan_list[idx] = tanf(d_D[idx] + float(j));
    // Quad is the direct extension of the triple reference: every function
    // after cosine sees the same overwritten state for this iteration.
    d_sinh_list[idx] = sinhf(d_D[idx] + float(j));
  }
}

}  // namespace

void launchSp(float* d_in, float* d_out, int num_funcs, int n,
              int stress_repeat, cudaStream_t stream) {
  constexpr int block = 64;  // Exact reference launch width.
  const int grid = (n + block - 1) / block;
  switch (num_funcs) {
    case 1:
      spCosTactKernel<<<grid, block, 0, stream>>>(
          d_in, n, stress_repeat);
      break;
    case 2:
      spDualTactKernel<<<grid, block, 0, stream>>>(
          d_in, d_out, n, stress_repeat);
      break;
    case 3:
      spTripleTactKernel<<<grid, block, 0, stream>>>(
          d_in, d_out, d_out + size_t(2) * n, n, stress_repeat);
      break;
    case 4:
      spQuadTactKernel<<<grid, block, 0, stream>>>(
          d_in, d_out, d_out + size_t(2) * n,
          d_out + size_t(3) * n, n, stress_repeat);
      break;
    default:
      throw UnsupportedError("SP old-TACT stress kernels support 1..4 canonical functions.");
  }
  TACT_CUDA_CHECK(cudaGetLastError());
}

}  // namespace tact
