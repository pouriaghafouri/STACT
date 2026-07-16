#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cusparse.h>

#ifdef TACT_ENABLE_CUSPARSELT
#include <cusparseLt.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define TACT_CUDA_CHECK(call) do {                                      \
  cudaError_t _e = (call);                                              \
  if (_e != cudaSuccess) {                                              \
    std::ostringstream _oss;                                            \
    _oss << "CUDA error " << cudaGetErrorString(_e)                    \
         << " at " << __FILE__ << ":" << __LINE__;                    \
    throw std::runtime_error(_oss.str());                               \
  }                                                                     \
} while(0)

#define TACT_CUSPARSE_CHECK(call) do {                                  \
  cusparseStatus_t _s = (call);                                         \
  if (_s != CUSPARSE_STATUS_SUCCESS) {                                  \
    std::ostringstream _oss;                                            \
    _oss << "cuSPARSE error code " << int(_s)                          \
         << " at " << __FILE__ << ":" << __LINE__;                    \
    throw std::runtime_error(_oss.str());                               \
  }                                                                     \
} while(0)

#ifdef TACT_ENABLE_CUSPARSELT
#define TACT_CUSPARSELT_CHECK(call) do {                                \
  cusparseStatus_t _s = (call);                                         \
  if (_s != CUSPARSE_STATUS_SUCCESS) {                                  \
    std::ostringstream _oss;                                            \
    _oss << "cuSPARSELt error " << cusparseGetErrorName(_s)            \
         << " (" << cusparseGetErrorString(_s) << ", code "           \
         << int(_s) << ") at " << __FILE__ << ":" << __LINE__;        \
    throw std::runtime_error(_oss.str());                               \
  }                                                                     \
} while(0)
#else
#define TACT_CUSPARSELT_CHECK(call) do { (void)sizeof(call); } while(0)
#endif

namespace tact {

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int WARP_SIZE = 32;
// The reference TACT kernels launch eight warps per block.  Keeping the same
// shape preserves their 80-input-per-warp packing and occupancy experiment.
constexpr int WARPS_PER_BLOCK = 8;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
constexpr int MAX_FUNCS = 4;
constexpr int OUT_PAD = 16;

enum FunctionId : int {
  F_EXP = 0,
  F_LOG1P = 1,
  F_SIN = 2,
  F_COS = 3,
  F_TAN = 4,
  F_SINH = 5,
  F_COSH = 6,
  F_TANH = 7
};

enum BasisKind : int {
  BASIS_EVEN = 0,
  BASIS_MIXED = 1
};

enum LayoutKind : int {
  LAYOUT_DENSE = 0,
  LAYOUT_SPARSE24 = 1
};

enum EngineKind : int {
  ENGINE_SFU = 0,
  ENGINE_SP = 1,
  ENGINE_WMMA = 2,
  ENGINE_CUSPARSE = 3,
  ENGINE_CUSPARSELT = 4
};

struct Options {
  EngineKind engine = ENGINE_WMMA;
  LayoutKind layout = LAYOUT_DENSE;
  std::vector<FunctionId> funcs{F_SIN, F_COS, F_TAN};
  int terms = 5;
  int n = 1 << 20;
  int warmup = 5;
  int iterations = 20;
  int inner_repeat = 1;
  float threshold = 1e-5f;
  uint64_t seed = 1234;
  bool min_set = false;
  bool max_set = false;
  float xmin = 0.0f;
  float xmax = 0.0f;
  std::string csv_path;
  bool validate = true;
  bool print_layout = false;
  bool profile_sections = true;
  bool tact_mode = false;
  int tact_steps = 2000;
  bool debug = false;
  int debug_values = 8;
  int debug_rows = 2;
  std::string debug_file;
};

struct ProblemSpec {
  BasisKind basis = BASIS_EVEN;
  int num_funcs = 0;
  int ipr = 0;
  std::vector<int> positions;
  // Requested-output index -> physical coefficient-column group.  The old
  // triple kernel stores cos, sin, tan groups in that order even though the
  // public CLI convention is commonly sin,cos,tan.
  std::vector<int> output_groups;
};

struct SectionTimes {
  float total_ms = 0.0f;
  float scalar_ms = 0.0f;
  float fused_kernel_ms = 0.0f;
  float build_power_ms = 0.0f;
  float matmul_ms = 0.0f;
  float postprocess_ms = 0.0f;
};

struct RunResult {
  float init_ms = 0.0f;
  float avg_h2d_ms = 0.0f;
  float avg_kernel_ms = 0.0f;
  float avg_d2h_ms = 0.0f;
  float avg_ms = 0.0f;
  float min_ms = 0.0f;
  float max_ms = 0.0f;
  float avg_scalar_ms = 0.0f;
  float avg_fused_kernel_ms = 0.0f;
  float avg_build_power_ms = 0.0f;
  float avg_matmul_ms = 0.0f;
  float avg_postprocess_ms = 0.0f;
  double throughput_mvals_s = 0.0;
  double timed_values = 0.0;
  double timed_outputs = 0.0;
  double max_abs = 0.0;
  double max_rel = 0.0;
  int64_t failures = 0;
  std::vector<double> func_max_abs;
  std::vector<double> func_max_rel;
  std::vector<int64_t> func_failures;
};

class DeviceBuffer {
 public:
  void* ptr = nullptr;
  size_t bytes = 0;
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t nbytes) { allocate(nbytes); }
  ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept : ptr(other.ptr), bytes(other.bytes) {
    other.ptr = nullptr; other.bytes = 0;
  }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (ptr) cudaFree(ptr);
      ptr = other.ptr; bytes = other.bytes;
      other.ptr = nullptr; other.bytes = 0;
    }
    return *this;
  }
  void allocate(size_t nbytes) {
    if (ptr) cudaFree(ptr);
    ptr = nullptr; bytes = nbytes;
    if (nbytes > 0) TACT_CUDA_CHECK(cudaMalloc(&ptr, nbytes));
  }
  template <typename T> T* as() { return reinterpret_cast<T*>(ptr); }
  template <typename T> const T* as() const { return reinterpret_cast<const T*>(ptr); }
};

class GpuTimer {
 public:
  cudaEvent_t start{}, stop{};
  GpuTimer() {
    TACT_CUDA_CHECK(cudaEventCreate(&start));
    TACT_CUDA_CHECK(cudaEventCreate(&stop));
  }
  ~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  void tic(cudaStream_t stream = nullptr) { TACT_CUDA_CHECK(cudaEventRecord(start, stream)); }
  float toc(cudaStream_t stream = nullptr) {
    TACT_CUDA_CHECK(cudaEventRecord(stop, stream));
    TACT_CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    TACT_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
  }
};

const char* functionName(FunctionId f);
std::string functionListName(const std::vector<FunctionId>& fs);
std::string engineName(EngineKind e);
std::string layoutName(LayoutKind l);
FunctionId parseFunction(const std::string& raw);
EngineKind parseEngine(const std::string& raw);
LayoutKind parseLayout(const std::string& raw);
Options parseArgs(int argc, char** argv);

bool isEvenGroup(FunctionId f);
bool isMixedGroup(FunctionId f);
bool isSfuSupported(FunctionId f);
bool allSfuSupported(const std::vector<FunctionId>& funcs);
BasisKind determineBasis(const std::vector<FunctionId>& funcs);
int inputsPerRow(int terms, int num_funcs);
int roundUp(int x, int a);
std::vector<int> makePositions(int terms, int ipr, LayoutKind layout);
void checkSparse24(const std::vector<int>& pos, int terms, int ipr);
void printProblemSpec(const ProblemSpec& spec, const Options& opt);
ProblemSpec makeProblemSpec(const Options& opt);
std::vector<float> makeInputs(const Options& opt, BasisKind basis);
RunResult computeValidation(const std::vector<float>& out,
                            const std::vector<float>& ref,
                            int n,
                            int num_funcs,
                            float threshold);
void appendCsv(const std::string& path, const Options& opt, const RunResult& r);
RunResult runBenchmark(const Options& opt);

__host__ __device__ inline double factorialInt(int n) {
  double r = 1.0;
  for (int i = 2; i <= n; ++i) r *= static_cast<double>(i);
  return r;
}

__host__ __device__ inline float coeffFor(FunctionId f, int t) {
  switch (f) {
    case F_COS:
    case F_COSH: {
      int p = 2 * (t + 1);
      float v = 1.0f / static_cast<float>(factorialInt(p));
      return (f == F_COS && (t % 2 == 0)) ? -v : v;
    }
    case F_SIN:
    case F_SINH: {
      int p = 2 * t + 3;
      float v = 1.0f / static_cast<float>(factorialInt(p));
      return (f == F_SIN && (t % 2 == 0)) ? -v : v;
    }
    case F_TAN: {
      constexpr float coeffs[8] = {
        1.0f / 3.0f,
        2.0f / 15.0f,
        17.0f / 315.0f,
        62.0f / 2835.0f,
        1382.0f / 155925.0f,
        21844.0f / 6081075.0f,
        929569.0f / 638512875.0f,
        6404582.0f / 10854718875.0f
      };
      return t < 8 ? coeffs[t] : 0.0f;
    }
    case F_TANH: {
      constexpr float coeffs[8] = {
        -1.0f / 3.0f,
        2.0f / 15.0f,
        -17.0f / 315.0f,
        62.0f / 2835.0f,
        -1382.0f / 155925.0f,
        21844.0f / 6081075.0f,
        -929569.0f / 638512875.0f,
        6404582.0f / 10854718875.0f
      };
      return t < 8 ? coeffs[t] : 0.0f;
    }
    case F_EXP: {
      int p = t + 1;
      return 1.0f / static_cast<float>(factorialInt(p));
    }
    case F_LOG1P: {
      int p = t + 1;
      float v = 1.0f / float(p);
      return (t % 2 == 0) ? v : -v;
    }
  }
  return 0.0f;
}

__host__ __device__ inline bool hasLeadingOne(FunctionId f) {
  return f == F_EXP || f == F_SIN || f == F_COS || f == F_TAN ||
         f == F_SINH || f == F_COSH || f == F_TANH;
}

__host__ __device__ inline bool needsMultiplyByX(FunctionId f) {
  return f == F_SIN || f == F_TAN || f == F_SINH || f == F_TANH;
}

__host__ __device__ inline float postprocess(FunctionId f, float residual, float x) {
  float y = residual;
  if (hasLeadingOne(f)) y += 1.0f;
  if (needsMultiplyByX(f)) y *= x;
  return y;
}

__host__ __device__ inline float powerFor(float x, int t, BasisKind basis) {
  float p = 1.0f;
  if (basis == BASIS_EVEN) {
    float x2 = x * x;
    for (int i = 0; i <= t; ++i) p *= x2;
  } else {
    for (int i = 0; i <= t; ++i) p *= x;
  }
  return p;
}

}  // namespace tact
