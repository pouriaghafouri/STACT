#include "tact/engines.cuh"
#include "tact/kernels.cuh"
#include "tact/matrix_builders.cuh"

namespace tact {

#ifndef TACT_ENABLE_CUSPARSELT
std::unique_ptr<Engine> makeCuSparseLtEngine(const EngineContext&) {
  throw std::runtime_error("This binary was built without cuSPARSELt. Reconfigure with -DTACT_ENABLE_CUSPARSELT=ON and set CUSPARSELT_ROOT if needed.");
}
#else
namespace {
class CuSparseLtEngine final : public Engine {
 public:
  explicit CuSparseLtEngine(EngineContext ctx) : ctx_(ctx) {
    const Options& opt = *ctx_.opt;
    if (opt.layout != LAYOUT_SPARSE24) {
      throw std::runtime_error("cuSPARSELt backend requires --layout sparse24.");
    }

    int device = 0;
    TACT_CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    TACT_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    if (prop.major < 8) {
      throw std::runtime_error("cuSPARSELt sparse tensor cores require Ampere or newer (SM 8.0+). Current device is SM " + std::to_string(prop.major) + "." + std::to_string(prop.minor));
    }

    rows_total_ = (opt.n + ctx_.spec.ipr - 1) / ctx_.spec.ipr;
    // cuSPARSELt limits each C/D dimension to 2,097,120 elements. 2,097,120 is
    // divisible by 16, so it also satisfies the FP16 descriptor multiple.
    chunk_rows_pad_ = std::min(roundUp(rows_total_, 16), 2097120);
    chunk_rows_pad_ = (chunk_rows_pad_ / 16) * 16;
    if (chunk_rows_pad_ <= 0) chunk_rows_pad_ = 16;

    d_powerRows_.allocate(size_t(chunk_rows_pad_) * WMMA_K * sizeof(__half));
    d_cth_.allocate(size_t(OUT_PAD) * chunk_rows_pad_ * sizeof(__half));

    // Important: to follow NVIDIA's cuSPARSELt sample, use row-major storage
    // with opA=T and opB=T. The physical sparse A is K x N, so opA(A) is the
    // logical coefficient matrix B^T: N x K. This avoids the NOT_SUPPORTED /
    // operation-not-permitted descriptor combination hit by row-major NN.
    auto b = buildCoefficientB(opt.funcs, opt.terms, ctx_.spec.ipr,
                               ctx_.spec.positions, ctx_.spec.output_groups);
    d_sparseA_.allocate(size_t(WMMA_K) * OUT_PAD * sizeof(__half));
    TACT_CUDA_CHECK(cudaMemcpy(d_sparseA_.as<__half>(), b.data(), b.size() * sizeof(__half), cudaMemcpyHostToDevice));

    TACT_CUSPARSELT_CHECK(cusparseLtInit(&handle_));
    constexpr int alignment = 16;

    // opA=T: physical A dimensions are K x M = 16 x 16.
    TACT_CUSPARSELT_CHECK(cusparseLtStructuredDescriptorInit(&handle_, &matA_,
                                                             WMMA_K, OUT_PAD, OUT_PAD,
                                                             alignment, CUDA_R_16F, CUSPARSE_ORDER_ROW,
                                                             CUSPARSELT_SPARSITY_50_PERCENT));
    // opB=T: physical B dimensions are N x K = chunk_rows_pad x 16.
    TACT_CUSPARSELT_CHECK(cusparseLtDenseDescriptorInit(&handle_, &matB_,
                                                        chunk_rows_pad_, WMMA_K, WMMA_K,
                                                        alignment, CUDA_R_16F, CUSPARSE_ORDER_ROW));
    // D = C shape is M x N = 16 x chunk_rows_pad.
    TACT_CUSPARSELT_CHECK(cusparseLtDenseDescriptorInit(&handle_, &matC_,
                                                        OUT_PAD, chunk_rows_pad_, chunk_rows_pad_,
                                                        alignment, CUDA_R_16F, CUSPARSE_ORDER_ROW));

    TACT_CUSPARSELT_CHECK(cusparseLtMatmulDescriptorInit(&handle_, &matmul_,
                                                         CUSPARSE_OPERATION_TRANSPOSE,
                                                         CUSPARSE_OPERATION_TRANSPOSE,
                                                         &matA_, &matB_, &matC_, &matC_,
#if CUSPARSELT_VERSION < 500
                                                         CUSPARSE_COMPUTE_16F));
#else
                                                         CUSPARSE_COMPUTE_32F));
#endif
    TACT_CUSPARSELT_CHECK(cusparseLtMatmulAlgSelectionInit(&handle_, &alg_sel_, &matmul_,
                                                           CUSPARSELT_MATMUL_ALG_DEFAULT));
#if CUSPARSELT_VERSION < 500
    // cuSPARSELt 0.3 exposes a workspace-size limit in PlanInit. Zero asks for
    // the default plan; MatmulGetWorkspace below reports its actual allocation.
    TACT_CUSPARSELT_CHECK(cusparseLtMatmulPlanInit(&handle_, &plan_, &matmul_, &alg_sel_, 0));
#else
    TACT_CUSPARSELT_CHECK(cusparseLtMatmulPlanInit(&handle_, &plan_, &matmul_, &alg_sel_));
#endif

#if CUSPARSELT_VERSION >= 500
    void* sparse_ptr_for_desc = d_sparseA_.ptr;
    TACT_CUSPARSELT_CHECK(cusparseLtMatmulDescSetAttribute(&handle_, &matmul_,
                                                           CUSPARSELT_MATMUL_SPARSE_MAT_POINTER,
                                                           &sparse_ptr_for_desc,
                                                           sizeof(sparse_ptr_for_desc)));
#endif

    DeviceBuffer d_valid(sizeof(int));
    TACT_CUDA_CHECK(cudaMemset(d_valid.ptr, 0, sizeof(int)));
    TACT_CUSPARSELT_CHECK(cusparseLtSpMMAPruneCheck(&handle_, &matmul_,
                                                    d_sparseA_.as<__half>(),
                                                    d_valid.as<int>(), nullptr));
    TACT_CUDA_CHECK(cudaDeviceSynchronize());
    int is_valid = 1;
    TACT_CUDA_CHECK(cudaMemcpy(&is_valid, d_valid.as<int>(), sizeof(int), cudaMemcpyDeviceToHost));
    if (is_valid != 0) {
      throw std::runtime_error("Generated coefficient matrix is not valid for cuSPARSELt 2:4 structured sparsity.");
    }

    size_t compressed_size = 0;
#if CUSPARSELT_VERSION < 500
    TACT_CUSPARSELT_CHECK(cusparseLtSpMMACompressedSize(&handle_, &plan_,
                                                        &compressed_size));
    d_compressed_.allocate(compressed_size);
    TACT_CUSPARSELT_CHECK(cusparseLtSpMMACompress(&handle_, &plan_,
                                                  d_sparseA_.as<__half>(),
                                                  d_compressed_.ptr,
                                                  nullptr));
#else
    size_t compressed_buffer_size = 0;
    TACT_CUSPARSELT_CHECK(cusparseLtSpMMACompressedSize(&handle_, &plan_,
                                                        &compressed_size,
                                                        &compressed_buffer_size));
    d_compressed_.allocate(compressed_size);
    d_compressed_buffer_.allocate(compressed_buffer_size);
    TACT_CUSPARSELT_CHECK(cusparseLtSpMMACompress(&handle_, &plan_,
                                                  d_sparseA_.as<__half>(),
                                                  d_compressed_.ptr,
                                                  d_compressed_buffer_.ptr,
                                                  nullptr));
#endif
    TACT_CUDA_CHECK(cudaDeviceSynchronize());

    size_t workspace_size = 0;
    TACT_CUSPARSELT_CHECK(cusparseLtMatmulGetWorkspace(&handle_, &plan_, &workspace_size));
    d_workspace_.allocate(workspace_size);
  }

  ~CuSparseLtEngine() override {
    cusparseLtMatmulPlanDestroy(&plan_);
#if CUSPARSELT_VERSION >= 500
    cusparseLtMatmulAlgSelectionDestroy(&alg_sel_);
#endif
    cusparseLtMatDescriptorDestroy(&matA_);
    cusparseLtMatDescriptorDestroy(&matB_);
    cusparseLtMatDescriptorDestroy(&matC_);
    cusparseLtDestroy(&handle_);
  }

  void run(cudaStream_t stream = nullptr) override {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cudaStream_t streams[1] = {stream};
    int processed = 0;
    while (processed < rows_total_) {
      launchBuildPowerRowsChunk(ctx_.d_in, d_powerRows_.as<__half>(), ctx_.opt->n,
                                processed, chunk_rows_pad_, ctx_.opt->terms,
                                ctx_.spec.ipr, ctx_.spec.basis, ctx_.d_positions, stream);
      TACT_CUSPARSELT_CHECK(cusparseLtMatmul(&handle_, &plan_,
                                             &alpha,
                                             d_compressed_.ptr,
                                             d_powerRows_.as<__half>(),
                                             &beta,
                                             d_cth_.as<__half>(),
                                             d_cth_.as<__half>(),
                                             d_workspace_.ptr,
                                             streams, 1));
      launchPostHalfChunk(d_cth_.as<__half>(), ctx_.d_in, ctx_.d_out,
                          ctx_.d_funcs, ctx_.spec.num_funcs, ctx_.opt->n,
                          processed, chunk_rows_pad_, ctx_.spec.ipr,
                          ctx_.d_output_groups, stream);
      processed += chunk_rows_pad_;
    }
  }

  SectionTimes runProfile(cudaStream_t stream = nullptr) override {
    SectionTimes st;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cudaStream_t streams[1] = {stream};
    GpuTimer total, phase;
    total.tic(stream);
    int processed = 0;
    while (processed < rows_total_) {
      phase.tic(stream);
      launchBuildPowerRowsChunk(ctx_.d_in, d_powerRows_.as<__half>(), ctx_.opt->n,
                                processed, chunk_rows_pad_, ctx_.opt->terms,
                                ctx_.spec.ipr, ctx_.spec.basis, ctx_.d_positions, stream);
      st.build_power_ms += phase.toc(stream);

      phase.tic(stream);
      TACT_CUSPARSELT_CHECK(cusparseLtMatmul(&handle_, &plan_,
                                             &alpha,
                                             d_compressed_.ptr,
                                             d_powerRows_.as<__half>(),
                                             &beta,
                                             d_cth_.as<__half>(),
                                             d_cth_.as<__half>(),
                                             d_workspace_.ptr,
                                             streams, 1));
      st.matmul_ms += phase.toc(stream);

      phase.tic(stream);
      launchPostHalfChunk(d_cth_.as<__half>(), ctx_.d_in, ctx_.d_out,
                          ctx_.d_funcs, ctx_.spec.num_funcs, ctx_.opt->n,
                          processed, chunk_rows_pad_, ctx_.spec.ipr,
                          ctx_.d_output_groups, stream);
      st.postprocess_ms += phase.toc(stream);
      processed += chunk_rows_pad_;
    }
    st.total_ms = total.toc(stream);
    return st;
  }

 private:
  EngineContext ctx_;
  int rows_total_ = 0;
  int chunk_rows_pad_ = 0;
  DeviceBuffer d_powerRows_, d_cth_, d_sparseA_, d_compressed_, d_compressed_buffer_, d_workspace_;
  cusparseLtHandle_t handle_{};
  cusparseLtMatDescriptor_t matA_{}, matB_{}, matC_{};
  cusparseLtMatmulDescriptor_t matmul_{};
  cusparseLtMatmulAlgSelection_t alg_sel_{};
  cusparseLtMatmulPlan_t plan_{};
};
}  // namespace

std::unique_ptr<Engine> makeCuSparseLtEngine(const EngineContext& ctx) {
  return std::make_unique<CuSparseLtEngine>(ctx);
}
#endif

}  // namespace tact
