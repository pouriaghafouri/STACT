#include "tact/engines.cuh"
#include "tact/kernels.cuh"

namespace tact {
namespace {
class BaselineEngine final : public Engine {
 public:
  BaselineEngine(EngineContext ctx, bool fast) : ctx_(ctx), fast_(fast) {}
  void run(cudaStream_t stream = nullptr) override {
    if (ctx_.opt->reference_compat) {
      if (fast_) {
        launchHistoricalSfu(ctx_.d_in, ctx_.d_out, ctx_.spec.num_funcs,
                            ctx_.opt->n, ctx_.opt->reference_repeat, stream);
      } else {
        launchHistoricalSp(ctx_.d_in, ctx_.d_out, ctx_.spec.num_funcs,
                           ctx_.opt->n, ctx_.opt->reference_repeat, stream);
      }
    } else {
      launchBaseline(ctx_.d_in, ctx_.d_out, ctx_.d_funcs,
                     ctx_.spec.num_funcs, ctx_.opt->n, fast_, stream);
    }
  }
  SectionTimes runProfile(cudaStream_t stream = nullptr) override {
    SectionTimes st;
    GpuTimer t;
    t.tic(stream);
    run(stream);
    st.baseline_ms = t.toc(stream);
    st.total_ms = st.baseline_ms;
    return st;
  }
 private:
  EngineContext ctx_;
  bool fast_;
};
}  // namespace

std::unique_ptr<Engine> makeBaselineEngine(const EngineContext& ctx, bool fast) {
  return std::make_unique<BaselineEngine>(ctx, fast);
}

}  // namespace tact
