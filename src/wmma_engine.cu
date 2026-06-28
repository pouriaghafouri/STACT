#include "tact/engines.cuh"
#include "tact/kernels.cuh"

namespace tact {
namespace {
class WmmaEngine final : public Engine {
 public:
  explicit WmmaEngine(EngineContext ctx) : ctx_(ctx) {}
  void run(cudaStream_t stream = nullptr) override {
    launchWmma(ctx_.d_in, ctx_.d_out, ctx_.d_funcs,
               ctx_.spec.num_funcs, ctx_.opt->n, ctx_.opt->terms,
               ctx_.spec.ipr, ctx_.spec.basis, ctx_.d_positions, stream);
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
};
}  // namespace

std::unique_ptr<Engine> makeWmmaEngine(const EngineContext& ctx) {
  return std::make_unique<WmmaEngine>(ctx);
}

}  // namespace tact
