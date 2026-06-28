#include "tact/common.cuh"
#include "tact/engines.cuh"
#include "tact/kernels.cuh"

namespace tact {

static std::unique_ptr<Engine> makeSelectedEngine(const Options& opt, const EngineContext& ctx) {
  switch (opt.engine) {
    case ENGINE_BASELINE_FAST: return makeBaselineEngine(ctx, true);
    case ENGINE_BASELINE_SP: return makeBaselineEngine(ctx, false);
    case ENGINE_WMMA: return makeWmmaEngine(ctx);
    case ENGINE_CUSPARSE: return makeCuSparseEngine(ctx);
    case ENGINE_CUSPARSELT: return makeCuSparseLtEngine(ctx);
  }
  throw std::runtime_error("Unknown engine");
}

RunResult runBenchmark(const Options& opt) {
  ProblemSpec spec = makeProblemSpec(opt);
  if (opt.print_layout) printProblemSpec(spec, opt);

  auto h_in = makeInputs(opt, spec.basis);
  std::vector<int> h_funcs(spec.num_funcs);
  for (int i = 0; i < spec.num_funcs; ++i) h_funcs[i] = int(opt.funcs[i]);

  DeviceBuffer d_in(size_t(opt.n) * sizeof(float));
  DeviceBuffer d_out(size_t(spec.num_funcs) * opt.n * sizeof(float));
  DeviceBuffer d_ref(size_t(spec.num_funcs) * opt.n * sizeof(float));
  DeviceBuffer d_funcs(size_t(spec.num_funcs) * sizeof(int));
  DeviceBuffer d_positions(size_t(spec.positions.size()) * sizeof(int));

  TACT_CUDA_CHECK(cudaMemcpy(d_in.as<float>(), h_in.data(), size_t(opt.n) * sizeof(float), cudaMemcpyHostToDevice));
  TACT_CUDA_CHECK(cudaMemcpy(d_funcs.as<int>(), h_funcs.data(), size_t(spec.num_funcs) * sizeof(int), cudaMemcpyHostToDevice));
  TACT_CUDA_CHECK(cudaMemcpy(d_positions.as<int>(), spec.positions.data(), spec.positions.size() * sizeof(int), cudaMemcpyHostToDevice));
  TACT_CUDA_CHECK(cudaMemset(d_out.as<float>(), 0, size_t(spec.num_funcs) * opt.n * sizeof(float)));
  TACT_CUDA_CHECK(cudaMemset(d_ref.as<float>(), 0, size_t(spec.num_funcs) * opt.n * sizeof(float)));

  EngineContext ctx{&opt, spec, d_in.as<float>(), d_out.as<float>(), d_funcs.as<int>(), d_positions.as<int>()};
  EngineContext ref_ctx{&opt, spec, d_in.as<float>(), d_ref.as<float>(), d_funcs.as<int>(), d_positions.as<int>()};

  auto ref_engine = makeBaselineEngine(ref_ctx, true);
  ref_engine->run();
  TACT_CUDA_CHECK(cudaDeviceSynchronize());

  auto init_start = std::chrono::high_resolution_clock::now();
  auto engine = makeSelectedEngine(opt, ctx);
  TACT_CUDA_CHECK(cudaDeviceSynchronize());
  auto init_stop = std::chrono::high_resolution_clock::now();

  GpuTimer timer;
  std::vector<float> times;
  times.reserve(opt.iterations);
  SectionTimes section_sum;

  for (int i = 0; i < opt.warmup; ++i) {
    for (int r = 0; r < opt.inner_repeat; ++r) engine->run();
    TACT_CUDA_CHECK(cudaDeviceSynchronize());
  }

  for (int i = 0; i < opt.iterations; ++i) {
    timer.tic();
    SectionTimes iter_sections;
    for (int r = 0; r < opt.inner_repeat; ++r) {
      if (opt.profile_sections) {
        SectionTimes st = engine->runProfile();
        iter_sections.baseline_ms += st.baseline_ms;
        iter_sections.fused_kernel_ms += st.fused_kernel_ms;
        iter_sections.build_power_ms += st.build_power_ms;
        iter_sections.matmul_ms += st.matmul_ms;
        iter_sections.postprocess_ms += st.postprocess_ms;
      } else {
        engine->run();
      }
    }
    float total_ms = timer.toc();
    times.push_back(total_ms);
    section_sum.baseline_ms += iter_sections.baseline_ms;
    section_sum.fused_kernel_ms += iter_sections.fused_kernel_ms;
    section_sum.build_power_ms += iter_sections.build_power_ms;
    section_sum.matmul_ms += iter_sections.matmul_ms;
    section_sum.postprocess_ms += iter_sections.postprocess_ms;
  }

  TACT_CUDA_CHECK(cudaDeviceSynchronize());

  RunResult result;
  result.init_ms = std::chrono::duration<float, std::milli>(init_stop - init_start).count();
  result.avg_ms = std::accumulate(times.begin(), times.end(), 0.0f) / float(times.size());
  result.min_ms = *std::min_element(times.begin(), times.end());
  result.max_ms = *std::max_element(times.begin(), times.end());
  if (opt.profile_sections) {
    float denom = float(opt.iterations);
    result.avg_baseline_ms = section_sum.baseline_ms / denom;
    result.avg_fused_kernel_ms = section_sum.fused_kernel_ms / denom;
    result.avg_build_power_ms = section_sum.build_power_ms / denom;
    result.avg_matmul_ms = section_sum.matmul_ms / denom;
    result.avg_postprocess_ms = section_sum.postprocess_ms / denom;
  }
  result.timed_values = double(opt.n) * double(opt.inner_repeat);
  result.timed_outputs = double(opt.n) * double(spec.num_funcs) * double(opt.inner_repeat);
  result.throughput_mvals_s = result.timed_outputs / (double(result.avg_ms) * 1000.0);

  if (opt.validate) {
    std::vector<float> h_out(size_t(spec.num_funcs) * opt.n);
    std::vector<float> h_ref(size_t(spec.num_funcs) * opt.n);
    TACT_CUDA_CHECK(cudaMemcpy(h_out.data(), d_out.as<float>(), h_out.size() * sizeof(float), cudaMemcpyDeviceToHost));
    TACT_CUDA_CHECK(cudaMemcpy(h_ref.data(), d_ref.as<float>(), h_ref.size() * sizeof(float), cudaMemcpyDeviceToHost));
    RunResult v = computeValidation(h_out, h_ref, opt.n, spec.num_funcs, opt.threshold);
    result.max_abs = v.max_abs;
    result.max_rel = v.max_rel;
    result.failures = v.failures;
    result.func_max_abs = std::move(v.func_max_abs);
    result.func_max_rel = std::move(v.func_max_rel);
    result.func_failures = std::move(v.func_failures);
  }

  return result;
}

}  // namespace tact
