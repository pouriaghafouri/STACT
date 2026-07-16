#include "tact/kernels.cuh"

namespace tact {
namespace {

__global__ void sfuCosTactKernel(float* d_D, int n, int steps) {
  const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= size_t(n)) return;
  for (int j = 0; j < steps; ++j) {
    d_D[idx] = __cosf(d_D[idx] + float(j));
  }
}

__global__ void sfuDualTactKernel(float* d_D, float* d_sin_list,
                                  int n, int steps) {
  const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= size_t(n)) return;
  for (int j = 0; j < steps; ++j) {
    d_sin_list[idx] = __sinf(d_D[idx] + float(j));
    d_D[idx] = __cosf(d_D[idx] + float(j));
  }
}

__global__ void sfuTripleTactKernel(float* d_D, float* d_sin_list,
                                    float* d_tan_list, int n, int steps) {
  const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= size_t(n)) return;
  for (int j = 0; j < steps; ++j) {
    d_sin_list[idx] = __sinf(d_D[idx] + float(j));
    d_D[idx] = __cosf(d_D[idx] + float(j));
    // This intentionally uses the newly overwritten cosine state, exactly as
    // TripleTranscendSFU.cu does.
    d_tan_list[idx] = __tanf(d_D[idx] + float(j));
  }
}

}  // namespace

void launchSfu(float* d_in, float* d_out, int num_funcs, int n,
               int stress_repeat, cudaStream_t stream) {
  constexpr int block = 64;  // Exact reference launch width.
  const int grid = (n + block - 1) / block;
  switch (num_funcs) {
    case 1:
      sfuCosTactKernel<<<grid, block, 0, stream>>>(
          d_in, n, stress_repeat);
      break;
    case 2:
      sfuDualTactKernel<<<grid, block, 0, stream>>>(
          d_in, d_out, n, stress_repeat);
      break;
    case 3:
      sfuTripleTactKernel<<<grid, block, 0, stream>>>(
          d_in, d_out, d_out + size_t(2) * n, n, stress_repeat);
      break;
    default:
      throw UnsupportedError(
          "SFU old-TACT stress kernels support only cos; sin,cos; or "
          "sin,cos,tan. The quad case contains sinh, which has no approved "
          "SFU implementation in this project.");
  }
  TACT_CUDA_CHECK(cudaGetLastError());
}

}  // namespace tact
