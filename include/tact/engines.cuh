#pragma once
#include "tact/common.cuh"
#include <memory>

namespace tact {

struct EngineContext {
  const Options* opt = nullptr;
  ProblemSpec spec;
  float* d_in = nullptr;
  float* d_out = nullptr;
  const int* d_funcs = nullptr;
  const int* d_positions = nullptr;
  const int* d_output_groups = nullptr;
};

class Engine {
 public:
  virtual ~Engine() = default;
  virtual void run(cudaStream_t stream = nullptr) = 0;
  virtual SectionTimes runProfile(cudaStream_t stream = nullptr) {
    GpuTimer t;
    t.tic(stream);
    run(stream);
    SectionTimes st;
    st.total_ms = t.toc(stream);
    return st;
  }
  // Sparse engines may expose C^T for small debug runs. The leading dimension
  // is returned so runner-side logical C can be compared without knowing the
  // engine's private storage layout.
  virtual bool copyIntermediate(std::vector<float>&, int&) { return false; }
};

std::unique_ptr<Engine> makeSfuEngine(const EngineContext& ctx);
std::unique_ptr<Engine> makeSpEngine(const EngineContext& ctx);
std::unique_ptr<Engine> makeWmmaEngine(const EngineContext& ctx);
std::unique_ptr<Engine> makeCuSparseEngine(const EngineContext& ctx);
std::unique_ptr<Engine> makeSparseTensorCoreEngine(const EngineContext& ctx);

}  // namespace tact
