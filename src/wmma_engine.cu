#include "tact/engines.cuh"
#include "tact/kernels.cuh"
#include "tact/matrix_builders.cuh"

namespace tact {
namespace {
class WmmaEngine final : public Engine {
 public:
  explicit WmmaEngine(EngineContext ctx) : ctx_(ctx) {
    const auto& funcs = ctx_.opt->funcs;
    canonical_path_ = ctx_.opt->layout == LAYOUT_DENSE &&
        ((funcs.size() == 1 && funcs[0] == F_COS) ||
         (funcs.size() == 3 && funcs[0] == F_SIN &&
          funcs[1] == F_COS && funcs[2] == F_TAN));
    auto coefficients = buildCoefficientLayout(
        ctx_.opt->funcs, ctx_.opt->terms, ctx_.spec.ipr,
        ctx_.spec.positions, ctx_.spec.output_groups);
    d_coefficients_.allocate(coefficients.physical.size() * sizeof(__half));
    TACT_CUDA_CHECK(cudaMemcpy(d_coefficients_.as<__half>(),
                               coefficients.physical.data(),
                               coefficients.physical.size() * sizeof(__half),
                               cudaMemcpyHostToDevice));
  }
  void run(cudaStream_t stream = nullptr) override {
    if (canonical_path_) {
      launchCanonicalWmma(
          ctx_.d_in, ctx_.d_out, ctx_.spec.num_funcs, ctx_.opt->n,
          ctx_.opt->terms, d_coefficients_.as<__half>(),
          ctx_.opt->stress_repeat, stream);
      return;
    }
    launchWmma(ctx_.d_in, ctx_.d_out, ctx_.d_funcs,
               ctx_.spec.num_funcs, ctx_.opt->n, ctx_.opt->terms,
               ctx_.spec.ipr, ctx_.spec.basis, ctx_.d_positions,
               ctx_.d_output_groups, d_coefficients_.as<__half>(),
               ctx_.opt->stress_repeat, stream);
  }
  SectionTimes runProfile(cudaStream_t stream = nullptr) override {
    SectionTimes st;
    GpuTimer t;
    t.tic(stream);
    run(stream);
    st.fused_kernel_ms = t.toc(stream);
    st.total_ms = st.fused_kernel_ms;
    return st;
  }
 private:
  EngineContext ctx_;
  DeviceBuffer d_coefficients_;
  bool canonical_path_ = false;
};
}  // namespace

std::unique_ptr<Engine> makeWmmaEngine(const EngineContext& ctx) {
  return std::make_unique<WmmaEngine>(ctx);
}

}  // namespace tact
