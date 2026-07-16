#include "tact/engines.cuh"
#include "tact/kernels.cuh"
#include "tact/matrix_builders.cuh"

namespace tact {
namespace {

// NVIDIA documents CSR_ALG2 as the CSR SpMM algorithm intended for row-major
// dense operands. A^T and C^T are row-major in this implementation.
constexpr cusparseSpMMAlg_t kSpmmAlgorithm = CUSPARSE_SPMM_CSR_ALG2;

class CuSparseEngine final : public Engine {
 public:
  explicit CuSparseEngine(EngineContext ctx) : ctx_(ctx) {
    const Options& opt = *ctx_.opt;
    rows_ = (opt.n + ctx_.spec.ipr - 1) / ctx_.spec.ipr;
    rows_pad_ = rows_;
    k_rows_ = opt.terms * ctx_.spec.ipr;
    output_rows_ = ctx_.spec.num_funcs * ctx_.spec.ipr;

    // Only allocate the K rows and output rows that contain useful data.
    // The previous 16xrows buffers included zero-only padding planes.
    d_powerT_.allocate(size_t(k_rows_) * rows_pad_ * sizeof(__half));
    d_ct_.allocate(size_t(output_rows_) * rows_pad_ * sizeof(float));

    auto coefficients = buildCoefficientLayout(
        opt.funcs, opt.terms, ctx_.spec.ipr,
        ctx_.spec.positions, ctx_.spec.output_groups);
    std::vector<int> row_ptr, col_ind;
    std::vector<__half> values;
    buildCoefficientCsrTranspose(
        coefficients.logical, output_rows_, k_rows_,
        row_ptr, col_ind, values);
    nnz_ = int(values.size());

    d_row_ptr_.allocate(row_ptr.size() * sizeof(int));
    d_col_ind_.allocate(col_ind.size() * sizeof(int));
    d_values_.allocate(values.size() * sizeof(__half));
    TACT_CUDA_CHECK(cudaMemcpy(
        d_row_ptr_.as<int>(), row_ptr.data(),
        row_ptr.size() * sizeof(int), cudaMemcpyHostToDevice));
    TACT_CUDA_CHECK(cudaMemcpy(
        d_col_ind_.as<int>(), col_ind.data(),
        col_ind.size() * sizeof(int), cudaMemcpyHostToDevice));
    TACT_CUDA_CHECK(cudaMemcpy(
        d_values_.as<__half>(), values.data(),
        values.size() * sizeof(__half), cudaMemcpyHostToDevice));

    TACT_CUSPARSE_CHECK(cusparseCreate(&handle_));
    TACT_CUSPARSE_CHECK(cusparseCreateCsr(
        &mat_coeff_t_, output_rows_, k_rows_, nnz_,
        d_row_ptr_.as<int>(), d_col_ind_.as<int>(), d_values_.as<__half>(),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F));
    TACT_CUSPARSE_CHECK(cusparseCreateDnMat(
        &mat_power_t_, k_rows_, rows_pad_, rows_pad_,
        d_powerT_.as<__half>(), CUDA_R_16F, CUSPARSE_ORDER_ROW));
    TACT_CUSPARSE_CHECK(cusparseCreateDnMat(
        &mat_result_t_, output_rows_, rows_pad_, rows_pad_,
        d_ct_.as<float>(), CUDA_R_32F, CUSPARSE_ORDER_ROW));

    const float alpha = float(opt.stress_repeat);
    const float beta = 1.0f;
    size_t buffer_size = 0;
    TACT_CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, mat_coeff_t_, mat_power_t_, &beta, mat_result_t_,
        CUDA_R_32F, kSpmmAlgorithm, &buffer_size));
    d_buffer_.allocate(buffer_size);
  }

  ~CuSparseEngine() override {
    if (mat_coeff_t_) cusparseDestroySpMat(mat_coeff_t_);
    if (mat_power_t_) cusparseDestroyDnMat(mat_power_t_);
    if (mat_result_t_) cusparseDestroyDnMat(mat_result_t_);
    if (handle_) cusparseDestroy(handle_);
  }

  void run(cudaStream_t stream = nullptr) override {
    TACT_CUSPARSE_CHECK(cusparseSetStream(handle_, stream));
    buildAndInitialize(stream);
    multiplyCollapsed(stream);
    postprocess(stream);
  }

  SectionTimes runProfile(cudaStream_t stream = nullptr) override {
    SectionTimes times;
    TACT_CUSPARSE_CHECK(cusparseSetStream(handle_, stream));
    GpuTimer total, phase;
    total.tic(stream);

    phase.tic(stream);
    buildAndInitialize(stream);
    times.build_power_ms = phase.toc(stream);

    phase.tic(stream);
    multiplyCollapsed(stream);
    times.matmul_ms = phase.toc(stream);

    phase.tic(stream);
    postprocess(stream);
    times.postprocess_ms = phase.toc(stream);

    times.total_ms = total.toc(stream);
    return times;
  }

  bool copyIntermediate(std::vector<float>& values,
                        int& leading_dimension) override {
    values.resize(size_t(output_rows_) * rows_pad_);
    TACT_CUDA_CHECK(cudaMemcpy(
        values.data(), d_ct_.as<float>(), values.size() * sizeof(float),
        cudaMemcpyDeviceToHost));
    leading_dimension = rows_pad_;
    return true;
  }

 private:
  void buildAndInitialize(cudaStream_t stream) {
    launchBuildPowerT(
        ctx_.d_in, d_powerT_.as<__half>(), ctx_.opt->n,
        rows_pad_, k_rows_, ctx_.opt->terms, ctx_.spec.ipr,
        ctx_.spec.basis, ctx_.d_positions, stream);
    launchFillFloatOnes(
        d_ct_.as<float>(), size_t(output_rows_) * rows_pad_, stream);
  }

  void multiplyCollapsed(cudaStream_t stream) {
    TACT_CUSPARSE_CHECK(cusparseSetStream(handle_, stream));

    // The old TACT matrix recurrence repeats the identical linear product:
    //   C_0 = 1
    //   C_{r+1} = B^T A^T + C_r
    // Therefore C_R = 1 + R * (B^T A^T). Calling SpMM R times forces the
    // complete multi-megabyte C matrix through global memory R times, unlike
    // WMMA which retains C in an accumulator fragment. cuSPARSE can express
    // the exact final algebra in one SpMM by setting alpha=R and beta=1.
    const float alpha = 1.0f;
    const float beta = 1.0f;
    for (int repeat = 0; repeat < ctx_.opt->stress_repeat; ++repeat) {
      TACT_CUSPARSE_CHECK(cusparseSpMM(
          handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
          CUSPARSE_OPERATION_NON_TRANSPOSE,
          &alpha, mat_coeff_t_, mat_power_t_, &beta, mat_result_t_,
          CUDA_R_32F, kSpmmAlgorithm, d_buffer_.ptr));
    }
  }

  void postprocess(cudaStream_t stream) {
    launchPostFloat(
        d_ct_.as<float>(), ctx_.d_in, ctx_.d_out, ctx_.d_funcs,
        ctx_.spec.num_funcs, ctx_.opt->n, rows_pad_, ctx_.spec.ipr,
        ctx_.d_output_groups, stream);
  }

  EngineContext ctx_;
  int rows_ = 0;
  int rows_pad_ = 0;
  int k_rows_ = 0;
  int output_rows_ = 0;
  int nnz_ = 0;
  DeviceBuffer d_powerT_;
  DeviceBuffer d_ct_;
  DeviceBuffer d_row_ptr_;
  DeviceBuffer d_col_ind_;
  DeviceBuffer d_values_;
  DeviceBuffer d_buffer_;
  cusparseHandle_t handle_ = nullptr;
  cusparseSpMatDescr_t mat_coeff_t_ = nullptr;
  cusparseDnMatDescr_t mat_power_t_ = nullptr;
  cusparseDnMatDescr_t mat_result_t_ = nullptr;
};

}  // namespace

std::unique_ptr<Engine> makeCuSparseEngine(const EngineContext& ctx) {
  return std::make_unique<CuSparseEngine>(ctx);
}

}  // namespace tact
