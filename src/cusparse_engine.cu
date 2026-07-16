#include "tact/engines.cuh"
#include "tact/kernels.cuh"
#include "tact/matrix_builders.cuh"

namespace tact {
namespace {
constexpr cusparseSpMMAlg_t kSpmmAlgorithm = CUSPARSE_SPMM_CSR_ALG2;

class CuSparseEngine final : public Engine {
 public:
  explicit CuSparseEngine(EngineContext ctx) : ctx_(ctx) {
    const Options& opt = *ctx_.opt;
    rows_ = (opt.n + ctx_.spec.ipr - 1) / ctx_.spec.ipr;
    rows_pad_ = roundUp(rows_, 16);
    d_powerT_.allocate(size_t(WMMA_K) * rows_pad_ * sizeof(__half));
    d_ct_.allocate(size_t(OUT_PAD) * rows_pad_ * sizeof(float));

    auto coefficients = buildCoefficientLayout(
        opt.funcs, opt.terms, ctx_.spec.ipr,
        ctx_.spec.positions, ctx_.spec.output_groups);
    std::vector<int> rowPtr, colInd;
    std::vector<__half> vals;
    buildCoefficientCsrTranspose(coefficients.physical, rowPtr, colInd, vals);
    d_rowPtr_.allocate(rowPtr.size() * sizeof(int));
    d_colInd_.allocate(colInd.size() * sizeof(int));
    d_vals_.allocate(vals.size() * sizeof(__half));
    TACT_CUDA_CHECK(cudaMemcpy(d_rowPtr_.as<int>(), rowPtr.data(), rowPtr.size() * sizeof(int), cudaMemcpyHostToDevice));
    TACT_CUDA_CHECK(cudaMemcpy(d_colInd_.as<int>(), colInd.data(), colInd.size() * sizeof(int), cudaMemcpyHostToDevice));
    TACT_CUDA_CHECK(cudaMemcpy(d_vals_.as<__half>(), vals.data(), vals.size() * sizeof(__half), cudaMemcpyHostToDevice));

    TACT_CUSPARSE_CHECK(cusparseCreate(&handle_));
    TACT_CUSPARSE_CHECK(cusparseCreateCsr(&matA_, OUT_PAD, WMMA_K, vals.size(),
                                          d_rowPtr_.as<int>(), d_colInd_.as<int>(), d_vals_.as<__half>(),
                                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F));
    TACT_CUSPARSE_CHECK(cusparseCreateDnMat(&matB_, WMMA_K, rows_pad_, rows_pad_,
                                            d_powerT_.as<__half>(), CUDA_R_16F, CUSPARSE_ORDER_ROW));
    TACT_CUSPARSE_CHECK(cusparseCreateDnMat(&matC_, OUT_PAD, rows_pad_, rows_pad_,
                                            d_ct_.as<float>(), CUDA_R_32F, CUSPARSE_ORDER_ROW));
    float alpha = 1.0f, beta = 1.0f;
    size_t buffer_size = 0;
    TACT_CUSPARSE_CHECK(cusparseSpMM_bufferSize(handle_,
                                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                &alpha, matA_, matB_, &beta, matC_,
                                                CUDA_R_32F, kSpmmAlgorithm,
                                                &buffer_size));
    d_buffer_.allocate(buffer_size);
  }

  ~CuSparseEngine() override {
    if (matA_) cusparseDestroySpMat(matA_);
    if (matB_) cusparseDestroyDnMat(matB_);
    if (matC_) cusparseDestroyDnMat(matC_);
    if (handle_) cusparseDestroy(handle_);
  }

  void run(cudaStream_t stream = nullptr) override {
    TACT_CUSPARSE_CHECK(cusparseSetStream(handle_, stream));
    launchBuildPowerT(ctx_.d_in, d_powerT_.as<__half>(), ctx_.opt->n, rows_pad_,
                      ctx_.opt->terms, ctx_.spec.ipr, ctx_.spec.basis,
                      ctx_.d_positions, stream);
    launchFillFloatOnes(d_ct_.as<float>(), size_t(OUT_PAD) * rows_pad_, stream);
    multiplyRepeated(stream);
    launchPostFloat(d_ct_.as<float>(), ctx_.d_in, ctx_.d_out, ctx_.d_funcs,
                    ctx_.spec.num_funcs, ctx_.opt->n, rows_pad_, ctx_.spec.ipr,
                    ctx_.d_output_groups, stream);
  }

  SectionTimes runProfile(cudaStream_t stream = nullptr) override {
    SectionTimes st;
    TACT_CUSPARSE_CHECK(cusparseSetStream(handle_, stream));
    GpuTimer total, phase;
    total.tic(stream);

    phase.tic(stream);
    launchBuildPowerT(ctx_.d_in, d_powerT_.as<__half>(), ctx_.opt->n, rows_pad_,
                      ctx_.opt->terms, ctx_.spec.ipr, ctx_.spec.basis,
                      ctx_.d_positions, stream);
    launchFillFloatOnes(d_ct_.as<float>(), size_t(OUT_PAD) * rows_pad_, stream);
    st.build_power_ms += phase.toc(stream);

    phase.tic(stream);
    multiplyRepeated(stream);
    st.matmul_ms += phase.toc(stream);

    phase.tic(stream);
    launchPostFloat(d_ct_.as<float>(), ctx_.d_in, ctx_.d_out, ctx_.d_funcs,
                    ctx_.spec.num_funcs, ctx_.opt->n, rows_pad_, ctx_.spec.ipr,
                    ctx_.d_output_groups, stream);
    st.postprocess_ms += phase.toc(stream);

    st.total_ms = total.toc(stream);
    return st;
  }

  bool copyIntermediate(std::vector<float>& values,
                        int& leading_dimension) override {
    values.resize(size_t(OUT_PAD) * rows_pad_);
    TACT_CUDA_CHECK(cudaMemcpy(values.data(), d_ct_.as<float>(),
                               values.size() * sizeof(float),
                               cudaMemcpyDeviceToHost));
    leading_dimension = rows_pad_;
    return true;
  }

 private:
  void multiplyRepeated(cudaStream_t stream) {
    TACT_CUSPARSE_CHECK(cusparseSetStream(handle_, stream));
    const float alpha = 1.0f;
    const float beta = 1.0f;
    // Exact TACT stress semantics: C starts at one and only SpMM is repeated.
    // Power construction and postprocessing remain outside this loop.
    for (int repeat = 0; repeat < ctx_.opt->stress_repeat; ++repeat) {
      TACT_CUSPARSE_CHECK(cusparseSpMM(handle_,
                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       &alpha, matA_, matB_, &beta, matC_,
                                       CUDA_R_32F,
                                       kSpmmAlgorithm,
                                       d_buffer_.ptr));
    }
  }

  EngineContext ctx_;
  int rows_ = 0;
  int rows_pad_ = 0;
  DeviceBuffer d_powerT_, d_ct_, d_rowPtr_, d_colInd_, d_vals_, d_buffer_;
  cusparseHandle_t handle_ = nullptr;
  cusparseSpMatDescr_t matA_ = nullptr;
  cusparseDnMatDescr_t matB_ = nullptr;
  cusparseDnMatDescr_t matC_ = nullptr;
};
}  // namespace

std::unique_ptr<Engine> makeCuSparseEngine(const EngineContext& ctx) {
  return std::make_unique<CuSparseEngine>(ctx);
}

}  // namespace tact
