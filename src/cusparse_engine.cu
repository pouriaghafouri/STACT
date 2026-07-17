#include "tact/engines.cuh"
#include "tact/kernels.cuh"
#include "tact/matrix_builders.cuh"

namespace tact {
namespace {

// The WMMA backend evaluates independent 16x16 tiles.  For cuSPARSE we keep
// exactly the same logical tiles, but concatenate their transposes along the
// dense-column dimension:
//
//   A_wide^T = [A_0^T | A_1^T | ...]
//   C_wide^T = [C_0^T | C_1^T | ...]
//   C_wide^T = B^T * A_wide^T + C_wide^T
//
// Every group of 16 dense columns is still one independent WMMA-equivalent
// tile.  Concatenation only removes tiny-matrix batching/dispatch overhead; it
// does not mix tiles or add arithmetic.
constexpr int kTileColumns = WMMA_M;
constexpr int kSelfTestRepeats = 3;

__global__ void buildWidePowerTransposeKernel(
    const float* __restrict__ input,
    float* __restrict__ power_t,
    int n,
    int rows_pad,
    int terms,
    int ipr,
    BasisKind basis,
    const int* __restrict__ positions) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) return;

  const int packed_row = index / ipr;
  const int slot = index % ipr;
  const float x = input[index];

  for (int term = 0; term < terms; ++term) {
    const int k = positions[slot * terms + term];
    const __half rounded = __float2half(powerFor(x, term, basis));
    // A^T is row-major: A^T[k, packed_row].  Keeping the long dense
    // dimension contiguous is the layout recommended for CSR_ALG2.
    power_t[size_t(k) * rows_pad + packed_row] = __half2float(rounded);
  }
}

__global__ void postprocessWideTransposeKernel(
    const float* __restrict__ result_t,
    const float* __restrict__ input,
    float* __restrict__ output,
    const int* __restrict__ funcs,
    int num_funcs,
    int n,
    int rows_pad,
    int ipr,
    const int* __restrict__ output_groups) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) return;

  const int packed_row = index / ipr;
  const int slot = index % ipr;
  const float x = input[index];

  for (int f = 0; f < num_funcs; ++f) {
    const FunctionId function = static_cast<FunctionId>(funcs[f]);
    const int output_column = output_groups[f] * ipr + slot;
    const float residual =
        result_t[size_t(output_column) * rows_pad + packed_row];
    output[size_t(f) * n + index] =
        needsMultiplyByX(function) ? residual * x : residual;
  }
}

void launchBuildWidePowerTranspose(
    const EngineContext& ctx,
    float* power_t,
    int rows_pad,
    cudaStream_t stream) {
  TACT_CUDA_CHECK(cudaMemsetAsync(
      power_t, 0,
      size_t(WMMA_K) * rows_pad * sizeof(float), stream));
  constexpr int block = 256;
  const int grid = (ctx.opt->n + block - 1) / block;
  buildWidePowerTransposeKernel<<<grid, block, 0, stream>>>(
      ctx.d_in, power_t, ctx.opt->n, rows_pad, ctx.opt->terms,
      ctx.spec.ipr, ctx.spec.basis, ctx.d_positions);
  TACT_CUDA_CHECK(cudaGetLastError());
}

void launchPostprocessWideTranspose(
    const EngineContext& ctx,
    const float* result_t,
    int rows_pad,
    cudaStream_t stream) {
  constexpr int block = 256;
  const int grid = (ctx.opt->n + block - 1) / block;
  postprocessWideTransposeKernel<<<grid, block, 0, stream>>>(
      result_t, ctx.d_in, ctx.d_out, ctx.d_funcs, ctx.spec.num_funcs,
      ctx.opt->n, rows_pad, ctx.spec.ipr, ctx.d_output_groups);
  TACT_CUDA_CHECK(cudaGetLastError());
}

class CuSparseEngine final : public Engine {
 public:
  explicit CuSparseEngine(EngineContext ctx) : ctx_(ctx) {
    const Options& opt = *ctx_.opt;
    rows_ = (opt.n + ctx_.spec.ipr - 1) / ctx_.spec.ipr;
    // Preserve exact WMMA tile boundaries.  Sixteen consecutive columns of
    // A^T/C^T correspond to one 16x16 WMMA tile.
    rows_pad_ = roundUp(rows_, kTileColumns);
    tile_count_ = rows_pad_ / kTileColumns;

    const auto coefficients = buildCoefficientLayout(
        opt.funcs, opt.terms, ctx_.spec.ipr,
        ctx_.spec.positions, ctx_.spec.output_groups);

    std::vector<int> row_offsets;
    std::vector<int> columns;
    std::vector<__half> half_values;
    buildCoefficientCsrTranspose(
        coefficients.physical, WMMA_N, WMMA_K,
        row_offsets, columns, half_values);

    host_row_offsets_ = row_offsets;
    host_columns_ = columns;
    host_values_.resize(half_values.size());
    for (size_t i = 0; i < half_values.size(); ++i) {
      // Match WMMA operands numerically: coefficients are rounded to FP16
      // once, then widened to FP32 because uniform FP32 is the robust CSR
      // SpMM datatype combination on CUDA 12.0.
      host_values_[i] = __half2float(half_values[i]);
    }
    nnz_ = static_cast<int64_t>(host_values_.size());
    validateCsrAgainstDense(coefficients.physical);

    d_row_offsets_.allocate(host_row_offsets_.size() * sizeof(int));
    d_columns_.allocate(host_columns_.size() * sizeof(int));
    d_values_.allocate(host_values_.size() * sizeof(float));
    d_power_t_.allocate(size_t(WMMA_K) * rows_pad_ * sizeof(float));
    d_result_t_.allocate(size_t(WMMA_N) * rows_pad_ * sizeof(float));

    TACT_CUDA_CHECK(cudaMemcpy(
        d_row_offsets_.as<int>(), host_row_offsets_.data(),
        host_row_offsets_.size() * sizeof(int), cudaMemcpyHostToDevice));
    TACT_CUDA_CHECK(cudaMemcpy(
        d_columns_.as<int>(), host_columns_.data(),
        host_columns_.size() * sizeof(int), cudaMemcpyHostToDevice));
    TACT_CUDA_CHECK(cudaMemcpy(
        d_values_.as<float>(), host_values_.data(),
        host_values_.size() * sizeof(float), cudaMemcpyHostToDevice));

    TACT_CUSPARSE_CHECK(cusparseCreate(&handle_));
    TACT_CUSPARSE_CHECK(cusparseCreateCsr(
        &mat_coeff_t_, WMMA_N, WMMA_K, nnz_,
        d_row_offsets_.as<int>(), d_columns_.as<int>(),
        d_values_.as<float>(), CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F));

    // One wide dense matrix, but still exact independent 16-column tile
    // groups.  Row-major gives contiguous access across the long dimension.
    TACT_CUSPARSE_CHECK(cusparseCreateDnMat(
        &mat_power_t_, WMMA_K, rows_pad_, rows_pad_,
        d_power_t_.as<float>(), CUDA_R_32F, CUSPARSE_ORDER_ROW));
    TACT_CUSPARSE_CHECK(cusparseCreateDnMat(
        &mat_result_t_, WMMA_N, rows_pad_, rows_pad_,
        d_result_t_.as<float>(), CUDA_R_32F, CUSPARSE_ORDER_ROW));

    // Input is already on the device before engine construction.  Build it
    // once here for algorithm selection and the mandatory validation gate.
    launchBuildWidePowerTranspose(ctx_, d_power_t_.as<float>(), rows_pad_, 0);
    TACT_CUDA_CHECK(cudaDeviceSynchronize());

    chooseFastestSupportedAlgorithm();
    runMandatoryWideSelfTest();
    graph_ready_ = tryCaptureStressGraph();
  }

  ~CuSparseEngine() override {
    if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
    if (graph_) cudaGraphDestroy(graph_);
    if (mat_coeff_t_) cusparseDestroySpMat(mat_coeff_t_);
    if (mat_power_t_) cusparseDestroyDnMat(mat_power_t_);
    if (mat_result_t_) cusparseDestroyDnMat(mat_result_t_);
    if (handle_) cusparseDestroy(handle_);
  }

  void run(cudaStream_t stream = nullptr) override {
    TACT_CUSPARSE_CHECK(cusparseSetStream(handle_, stream));
    launchBuildWidePowerTranspose(ctx_, d_power_t_.as<float>(), rows_pad_, stream);
    launchFillFloatOnes(
        d_result_t_.as<float>(), size_t(WMMA_N) * rows_pad_, stream);
    multiplyRepeated(stream);
    launchPostprocessWideTranspose(
        ctx_, d_result_t_.as<float>(), rows_pad_, stream);
  }

  SectionTimes runProfile(cudaStream_t stream = nullptr) override {
    TACT_CUSPARSE_CHECK(cusparseSetStream(handle_, stream));
    SectionTimes times;
    GpuTimer total;
    GpuTimer phase;
    total.tic(stream);

    phase.tic(stream);
    launchBuildWidePowerTranspose(ctx_, d_power_t_.as<float>(), rows_pad_, stream);
    launchFillFloatOnes(
        d_result_t_.as<float>(), size_t(WMMA_N) * rows_pad_, stream);
    times.build_power_ms = phase.toc(stream);

    phase.tic(stream);
    multiplyRepeated(stream);
    times.matmul_ms = phase.toc(stream);

    phase.tic(stream);
    launchPostprocessWideTranspose(
        ctx_, d_result_t_.as<float>(), rows_pad_, stream);
    times.postprocess_ms = phase.toc(stream);

    times.total_ms = total.toc(stream);
    return times;
  }

  bool copyIntermediate(std::vector<float>& values,
                        int& leading_dimension) override {
    values.resize(size_t(WMMA_N) * rows_pad_);
    TACT_CUDA_CHECK(cudaMemcpy(
        values.data(), d_result_t_.as<float>(),
        values.size() * sizeof(float), cudaMemcpyDeviceToHost));
    leading_dimension = rows_pad_;
    return true;
  }

 private:
  void validateCsrAgainstDense(const std::vector<__half>& dense_b) const {
    for (int output_column = 0; output_column < WMMA_N; ++output_column) {
      for (int k = 0; k < WMMA_K; ++k) {
        float reconstructed = 0.0f;
        for (int p = host_row_offsets_[output_column];
             p < host_row_offsets_[output_column + 1]; ++p) {
          if (host_columns_[p] == k) {
            reconstructed = host_values_[p];
            break;
          }
        }
        const float expected =
            __half2float(dense_b[k * WMMA_N + output_column]);
        if (reconstructed != expected) {
          std::ostringstream oss;
          oss << "cuSPARSE CSR validation failed at B^T["
              << output_column << ',' << k << "]: got "
              << reconstructed << ", expected " << expected;
          throw std::runtime_error(oss.str());
        }
      }
    }
  }

  void createCoefficientDescriptor(cusparseSpMatDescr_t* descriptor) {
    // cusparseCreateCsr follows the official NVIDIA sample and accepts
    // mutable device pointers (void*), even though SpMM only reads them.
    // Use DeviceBuffer::ptr directly here; calling the const as<T>() overload
    // produced const T* and failed to compile with CUDA 12.0.
    TACT_CUSPARSE_CHECK(cusparseCreateCsr(
        descriptor, WMMA_N, WMMA_K, nnz_,
        d_row_offsets_.ptr, d_columns_.ptr, d_values_.ptr,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  }

  bool queryBufferSize(cusparseSpMatDescr_t coefficient_descriptor,
                       cusparseSpMMAlg_t algorithm,
                       size_t& size_out) const {
    size_out = 0;
    const cusparseStatus_t status = cusparseSpMM_bufferSize(
        handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha_, coefficient_descriptor, mat_power_t_, &beta_, mat_result_t_,
        CUDA_R_32F, algorithm, &size_out);
    return status == CUSPARSE_STATUS_SUCCESS;
  }

  bool preprocessConfiguration(cusparseSpMatDescr_t coefficient_descriptor,
                               cusparseSpMMAlg_t algorithm,
                               DeviceBuffer& buffer) {
    const cusparseStatus_t status = cusparseSpMM_preprocess(
        handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha_, coefficient_descriptor, mat_power_t_, &beta_, mat_result_t_,
        CUDA_R_32F, algorithm, buffer.ptr);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      cudaGetLastError();
      return false;
    }
    TACT_CUDA_CHECK(cudaDeviceSynchronize());
    return true;
  }

  std::vector<float> buildFirstTileExpected(int repeats) const {
    const int inputs_per_tile = WMMA_M * ctx_.spec.ipr;
    const int input_count = std::min(ctx_.opt->n, inputs_per_tile);

    std::vector<float> host_input(
        static_cast<size_t>(input_count), 0.0f);
    TACT_CUDA_CHECK(cudaMemcpy(
        host_input.data(), ctx_.d_in,
        host_input.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // A_tile^T is represented in row-major order: [k, packed_row].
    std::vector<float> host_power(
        size_t(WMMA_K) * WMMA_M, 0.0f);
    for (int index = 0; index < input_count; ++index) {
      const int packed_row = index / ctx_.spec.ipr;
      const int slot = index % ctx_.spec.ipr;
      const float x = host_input[index];
      for (int term = 0; term < ctx_.opt->terms; ++term) {
        const int k =
            ctx_.spec.positions[slot * ctx_.opt->terms + term];
        host_power[size_t(k) * WMMA_M + packed_row] = __half2float(
            __float2half(powerFor(x, term, ctx_.spec.basis)));
      }
    }

    std::vector<float> expected(
        size_t(WMMA_N) * WMMA_M, 1.0f);
    for (int output_column = 0; output_column < WMMA_N;
         ++output_column) {
      for (int packed_row = 0; packed_row < WMMA_M; ++packed_row) {
        float dot = 0.0f;
        for (int p = host_row_offsets_[output_column];
             p < host_row_offsets_[output_column + 1]; ++p) {
          const int k = host_columns_[p];
          dot += host_values_[p] *
              host_power[size_t(k) * WMMA_M + packed_row];
        }
        expected[size_t(output_column) * WMMA_M + packed_row] +=
            float(repeats) * dot;
      }
    }
    return expected;
  }

  std::vector<float> copyFirstTileResult() const {
    std::vector<float> actual(
        size_t(WMMA_N) * WMMA_M, 0.0f);
    // The production C_wide^T descriptor is row-major with rows_pad_ columns.
    // Copy the first 16 columns of each of its 16 rows.  This is exactly the
    // first independent WMMA-equivalent 16x16 tile, but it is read from the
    // real production allocation and descriptor rather than a special test
    // descriptor.
    TACT_CUDA_CHECK(cudaMemcpy2D(
        actual.data(), size_t(WMMA_M) * sizeof(float),
        d_result_t_.as<float>(), size_t(rows_pad_) * sizeof(float),
        size_t(WMMA_M) * sizeof(float), WMMA_N,
        cudaMemcpyDeviceToHost));
    return actual;
  }

  bool validateFirstTile(const std::vector<float>& actual,
                         const std::vector<float>& expected,
                         std::string& error_out) const {
    for (size_t i = 0; i < actual.size(); ++i) {
      const float abs_error = std::abs(actual[i] - expected[i]);
      const float tolerance =
          2.0e-4f + 2.0e-5f * std::abs(expected[i]);
      if (abs_error > tolerance) {
        const int output_column = int(i / WMMA_M);
        const int packed_row = int(i % WMMA_M);
        std::ostringstream oss;
        oss << "C_wide^T first-tile mismatch at ["
            << output_column << ',' << packed_row << "]: got "
            << actual[i] << ", expected " << expected[i]
            << ", abs_error " << abs_error;
        error_out = oss.str();
        return false;
      }
    }
    return true;
  }

  bool runWideSelfTest(cusparseSpMatDescr_t coefficient_descriptor,
                       cusparseSpMMAlg_t algorithm,
                       DeviceBuffer& buffer,
                       int repeats,
                       std::string& error_out) {
    launchFillFloatOnes(
        d_result_t_.as<float>(), size_t(WMMA_N) * rows_pad_, 0);
    for (int repeat = 0; repeat < repeats; ++repeat) {
      const cusparseStatus_t status = cusparseSpMM(
          handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
          CUSPARSE_OPERATION_NON_TRANSPOSE,
          &alpha_, coefficient_descriptor, mat_power_t_, &beta_, mat_result_t_,
          CUDA_R_32F, algorithm, buffer.ptr);
      if (status != CUSPARSE_STATUS_SUCCESS) {
        std::ostringstream oss;
        oss << "cusparseSpMM returned " << cusparseGetErrorString(status)
            << " during mandatory wide-layout validation";
        error_out = oss.str();
        cudaGetLastError();
        return false;
      }
    }
    TACT_CUDA_CHECK(cudaDeviceSynchronize());

    const std::vector<float> expected = buildFirstTileExpected(repeats);
    const std::vector<float> actual = copyFirstTileResult();
    return validateFirstTile(actual, expected, error_out);
  }

  float benchmarkAlgorithm(cusparseSpMatDescr_t coefficient_descriptor,
                           cusparseSpMMAlg_t algorithm,
                           DeviceBuffer& candidate_buffer) {
    launchFillFloatOnes(
        d_result_t_.as<float>(), size_t(WMMA_N) * rows_pad_, 0);
    TACT_CUSPARSE_CHECK(cusparseSpMM(
        handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha_, coefficient_descriptor, mat_power_t_, &beta_, mat_result_t_,
        CUDA_R_32F, algorithm, candidate_buffer.ptr));
    TACT_CUDA_CHECK(cudaDeviceSynchronize());

    launchFillFloatOnes(
        d_result_t_.as<float>(), size_t(WMMA_N) * rows_pad_, 0);
    GpuTimer timer;
    timer.tic();
    constexpr int trials = 4;
    for (int i = 0; i < trials; ++i) {
      TACT_CUSPARSE_CHECK(cusparseSpMM(
          handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
          CUSPARSE_OPERATION_NON_TRANSPOSE,
          &alpha_, coefficient_descriptor, mat_power_t_, &beta_, mat_result_t_,
          CUDA_R_32F, algorithm, candidate_buffer.ptr));
    }
    return timer.toc() / float(trials);
  }

  void chooseFastestSupportedAlgorithm() {
    const cusparseSpMMAlg_t candidates[] = {
        CUSPARSE_SPMM_CSR_ALG2,
        CUSPARSE_SPMM_CSR_ALG1,
        CUSPARSE_SPMM_ALG_DEFAULT};

    float best_ms = std::numeric_limits<float>::infinity();
    bool found = false;
    std::string last_failure;

    for (cusparseSpMMAlg_t candidate : candidates) {
      cusparseSpMatDescr_t candidate_coeff = nullptr;
      createCoefficientDescriptor(&candidate_coeff);

      size_t candidate_size = 0;
      if (!queryBufferSize(candidate_coeff, candidate, candidate_size)) {
        cusparseDestroySpMat(candidate_coeff);
        continue;
      }

      DeviceBuffer candidate_buffer(candidate_size);
      // Preprocess only the candidate descriptor.  cuSPARSE may attach
      // algorithm-specific analysis state to the sparse descriptor; keeping a
      // private descriptor here prevents one candidate or matrix shape from
      // contaminating another one.
      preprocessConfiguration(candidate_coeff, candidate, candidate_buffer);

      std::string validation_error;
      const bool valid = runWideSelfTest(
          candidate_coeff, candidate, candidate_buffer,
          std::max(1, std::min(kSelfTestRepeats,
                               ctx_.opt->stress_repeat)),
          validation_error);
      if (!valid) {
        last_failure = validation_error;
        cusparseDestroySpMat(candidate_coeff);
        continue;
      }

      float elapsed = std::numeric_limits<float>::infinity();
      try {
        elapsed = benchmarkAlgorithm(
            candidate_coeff, candidate, candidate_buffer);
      } catch (...) {
        cudaGetLastError();
        cusparseDestroySpMat(candidate_coeff);
        continue;
      }

      if (elapsed < best_ms) {
        best_ms = elapsed;
        algorithm_ = candidate;
        buffer_size_ = candidate_size;
        found = true;
      }
      cusparseDestroySpMat(candidate_coeff);
    }

    if (!found) {
      std::ostringstream oss;
      oss << "no correct and supported cuSPARSE CSR SpMM algorithm for the "
             "wide WMMA-equivalent layout";
      if (!last_failure.empty()) {
        oss << "; last validation error: " << last_failure;
      }
      throw std::runtime_error(oss.str());
    }

    d_buffer_.allocate(buffer_size_);
    // Configure only the final production descriptor after algorithm
    // selection.  This avoids the v11 bug where preprocessing the
    // production-wide descriptor was then reused with an unrelated 16x16
    // test descriptor.
    preprocessConfiguration(mat_coeff_t_, algorithm_, d_buffer_);
  }

  void runMandatoryWideSelfTest() {
    const int repeats = std::max(
        1, std::min(kSelfTestRepeats, ctx_.opt->stress_repeat));
    std::string error;
    if (!runWideSelfTest(
            mat_coeff_t_, algorithm_, d_buffer_, repeats, error)) {
      throw std::runtime_error(
          "mandatory cuSPARSE production-layout validation failed: " + error);
    }
  }

  bool tryCaptureStressGraph() {
    if (ctx_.opt->stress_repeat < 2) return false;

    cudaStream_t capture_stream = nullptr;
    if (cudaStreamCreateWithFlags(
            &capture_stream, cudaStreamNonBlocking) != cudaSuccess) {
      cudaGetLastError();
      return false;
    }

    bool success = false;
    do {
      if (cusparseSetStream(handle_, capture_stream) !=
          CUSPARSE_STATUS_SUCCESS) {
        break;
      }

      // Resolve any lazy library setup before capture.
      if (cudaMemsetAsync(
              d_result_t_.as<float>(), 0,
              size_t(WMMA_N) * rows_pad_ * sizeof(float),
              capture_stream) != cudaSuccess) {
        break;
      }
      if (cusparseSpMM(
              handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
              CUSPARSE_OPERATION_NON_TRANSPOSE,
              &alpha_, mat_coeff_t_, mat_power_t_, &beta_, mat_result_t_,
              CUDA_R_32F, algorithm_, d_buffer_.ptr) !=
          CUSPARSE_STATUS_SUCCESS) {
        break;
      }
      if (cudaStreamSynchronize(capture_stream) != cudaSuccess) break;

      if (cudaStreamBeginCapture(
              capture_stream, cudaStreamCaptureModeThreadLocal) !=
          cudaSuccess) {
        break;
      }

      bool calls_ok = true;
      for (int repeat = 0; repeat < ctx_.opt->stress_repeat; ++repeat) {
        if (cusparseSpMM(
                handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha_, mat_coeff_t_, mat_power_t_, &beta_, mat_result_t_,
                CUDA_R_32F, algorithm_, d_buffer_.ptr) !=
            CUSPARSE_STATUS_SUCCESS) {
          calls_ok = false;
          break;
        }
      }

      cudaGraph_t captured = nullptr;
      const cudaError_t end_status =
          cudaStreamEndCapture(capture_stream, &captured);
      if (!calls_ok || end_status != cudaSuccess || !captured) {
        if (captured) cudaGraphDestroy(captured);
        break;
      }
      graph_ = captured;

      if (cudaGraphInstantiate(
              &graph_exec_, graph_, nullptr, nullptr, 0) != cudaSuccess) {
        cudaGraphDestroy(graph_);
        graph_ = nullptr;
        graph_exec_ = nullptr;
        break;
      }
      cudaGraphUpload(graph_exec_, capture_stream);
      cudaStreamSynchronize(capture_stream);
      success = true;
    } while (false);

    cusparseSetStream(handle_, nullptr);
    cudaStreamDestroy(capture_stream);
    if (!success) cudaGetLastError();
    return success;
  }

  void multiplyRepeated(cudaStream_t stream) {
    if (graph_ready_) {
      TACT_CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream));
      return;
    }

    for (int repeat = 0; repeat < ctx_.opt->stress_repeat; ++repeat) {
      TACT_CUSPARSE_CHECK(cusparseSpMM(
          handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
          CUSPARSE_OPERATION_NON_TRANSPOSE,
          &alpha_, mat_coeff_t_, mat_power_t_, &beta_, mat_result_t_,
          CUDA_R_32F, algorithm_, d_buffer_.ptr));
    }
  }

  EngineContext ctx_;
  int rows_ = 0;
  int rows_pad_ = 0;
  int tile_count_ = 0;
  int64_t nnz_ = 0;

  float alpha_ = 1.0f;
  float beta_ = 1.0f;
  cusparseSpMMAlg_t algorithm_ = CUSPARSE_SPMM_CSR_ALG2;
  size_t buffer_size_ = 0;

  std::vector<int> host_row_offsets_;
  std::vector<int> host_columns_;
  std::vector<float> host_values_;

  DeviceBuffer d_row_offsets_;
  DeviceBuffer d_columns_;
  DeviceBuffer d_values_;
  DeviceBuffer d_power_t_;
  DeviceBuffer d_result_t_;
  DeviceBuffer d_buffer_;

  cusparseHandle_t handle_ = nullptr;
  cusparseSpMatDescr_t mat_coeff_t_ = nullptr;
  cusparseDnMatDescr_t mat_power_t_ = nullptr;
  cusparseDnMatDescr_t mat_result_t_ = nullptr;

  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t graph_exec_ = nullptr;
  bool graph_ready_ = false;
};

}  // namespace

std::unique_ptr<Engine> makeCuSparseEngine(const EngineContext& ctx) {
  return std::make_unique<CuSparseEngine>(ctx);
}

}  // namespace tact
