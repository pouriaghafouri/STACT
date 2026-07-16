#include "tact/engines.cuh"
#include "tact/kernels.cuh"

namespace tact {
namespace {

class ScalarEngine final : public Engine {
 public:
  ScalarEngine(EngineContext ctx, bool use_sfu)
      : ctx_(ctx), use_sfu_(use_sfu) {}

  void run(cudaStream_t stream = nullptr) override {
    if (use_sfu_) {
      launchSfu(ctx_.d_in, ctx_.d_out, ctx_.d_funcs, ctx_.spec.num_funcs,
                ctx_.opt->n, ctx_.opt->tact_mode, ctx_.opt->tact_steps, stream);
    } else {
      launchSp(ctx_.d_in, ctx_.d_out, ctx_.d_funcs, ctx_.spec.num_funcs,
               ctx_.opt->n, ctx_.opt->tact_mode, ctx_.opt->tact_steps, stream);
    }
  }

  SectionTimes runProfile(cudaStream_t stream = nullptr) override {
    SectionTimes st;
    GpuTimer timer;
    timer.tic(stream);
    run(stream);
    st.scalar_ms = timer.toc(stream);
    st.total_ms = st.scalar_ms;
    return st;
  }

 private:
  EngineContext ctx_;
  bool use_sfu_;
};

}  // namespace

std::unique_ptr<Engine> makeSfuEngine(const EngineContext& ctx) {
  for (FunctionId function : ctx.opt->funcs) {
    if (!isSfuSupported(function)) {
      throw std::runtime_error(
          std::string("SFU engine does not support function ") +
          functionName(function));
    }
  }
  return std::make_unique<ScalarEngine>(ctx, true);
}

std::unique_ptr<Engine> makeSpEngine(const EngineContext& ctx) {
  return std::make_unique<ScalarEngine>(ctx, false);
}

}  // namespace tact
