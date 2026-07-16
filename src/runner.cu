#include "tact/common.cuh"
#include "tact/engines.cuh"
#include "tact/kernels.cuh"
#include "tact/matrix_builders.cuh"

namespace tact {

static void dumpDebug(std::ostream& os,
                      const Options& opt,
                      const ProblemSpec& spec,
                      const std::vector<float>& in,
                      const std::vector<float>& out,
                      const std::vector<float>& ref) {
  const int rows = std::min(opt.debug_rows, (opt.n + spec.ipr - 1) / spec.ipr);
  const int values = std::min(opt.debug_values, opt.n);
  auto b = buildCoefficientB(opt.funcs, opt.terms, spec.ipr,
                             spec.positions, spec.output_groups);

  std::vector<__half> a(size_t(rows) * WMMA_K, __float2half(0.0f));
  for (int row = 0; row < rows; ++row) {
    for (int slot = 0; slot < spec.ipr; ++slot) {
      int idx = row * spec.ipr + slot;
      if (idx >= opt.n) continue;
      for (int t = 0; t < opt.terms; ++t) {
        int k = spec.positions[slot * opt.terms + t];
        a[row * WMMA_K + k] = __float2half(powerFor(in[idx], t, spec.basis));
      }
    }
  }

  os << "\nTACT debug dump\n";
  os << "inputs (first " << values << "):";
  for (int i = 0; i < values; ++i) os << ' ' << in[i];
  os << "\nslot -> K positions\n";
  for (int slot = 0; slot < spec.ipr; ++slot) {
    os << "  slot " << slot << ':';
    for (int t = 0; t < opt.terms; ++t) os << " k" << spec.positions[slot * opt.terms + t];
    os << '\n';
  }
  os << "function -> coefficient-column group\n";
  for (int f = 0; f < spec.num_funcs; ++f) {
    os << "  " << functionName(opt.funcs[f]) << " -> " << spec.output_groups[f] << '\n';
  }

  os << "logical Tile A / power matrix (" << rows << "x16 shown)\n";
  for (int row = 0; row < rows; ++row) {
    os << "  A[" << row << "]";
    for (int k = 0; k < WMMA_K; ++k) os << ' ' << __half2float(a[row * WMMA_K + k]);
    os << '\n';
  }
  os << "logical Tile B / coefficient matrix (16x16)\n";
  for (int k = 0; k < WMMA_K; ++k) {
    os << "  B[" << k << "]";
    for (int col = 0; col < WMMA_N; ++col) os << ' ' << __half2float(b[k * WMMA_N + col]);
    os << '\n';
  }

  os << "logical Tile C / intermediate matrix (used columns)\n";
  for (int row = 0; row < rows; ++row) {
    os << "  C[" << row << "]";
    for (int col = 0; col < spec.num_funcs * spec.ipr; ++col) {
      float dot = 0.0f;
      for (int k = 0; k < WMMA_K; ++k) {
        dot += __half2float(a[row * WMMA_K + k]) * __half2float(b[k * WMMA_N + col]);
      }
      float c = opt.reference_compat ? 1.0f + opt.reference_repeat * dot : dot;
      os << ' ' << c;
    }
    os << '\n';
  }

  os << "outputs / references / absolute errors\n";
  for (int f = 0; f < spec.num_funcs; ++f) {
    os << "  " << functionName(opt.funcs[f]) << '\n';
    for (int i = 0; i < values; ++i) {
      float actual = out[size_t(f) * opt.n + i];
      float expected = ref[size_t(f) * opt.n + i];
      os << "    i=" << i << " x=" << in[i] << " out=" << actual
         << " ref=" << expected << " abs=" << std::abs(actual - expected) << '\n';
    }
  }

  if (opt.engine == ENGINE_CUSPARSE) {
    auto bt = buildCoefficientBT(opt.funcs, opt.terms, spec.ipr,
                                 spec.positions, spec.output_groups);
    std::vector<int> row_ptr, col_ind;
    std::vector<__half> vals;
    buildCSRFromBT(bt, row_ptr, col_ind, vals);
    os << "cuSPARSE CSR row_ptr:";
    for (int v : row_ptr) os << ' ' << v;
    os << "\ncuSPARSE CSR nonzeros (column:value):";
    for (size_t i = 0; i < vals.size(); ++i) {
      os << ' ' << col_ind[i] << ':' << __half2float(vals[i]);
    }
    os << '\n';
  }
  if (opt.layout == LAYOUT_SPARSE24) {
    os << "cuSPARSELt physical structured matrix is logical B (Kx16 row-major); opA=T.\n";
    bool valid = true;
    for (int col = 0; col < spec.num_funcs * spec.ipr; ++col) {
      os << "  col " << col << " 2:4 counts:";
      for (int group = 0; group < 4; ++group) {
        int count = 0;
        for (int k = group * 4; k < group * 4 + 4; ++k) {
          if (__half2float(b[k * WMMA_N + col]) != 0.0f) ++count;
        }
        valid = valid && count <= 2;
        os << ' ' << count;
      }
      os << '\n';
    }
    os << "  2:4 validity: " << (valid ? "valid" : "INVALID") << '\n';
  }
}

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
  DeviceBuffer d_output_groups(size_t(spec.output_groups.size()) * sizeof(int));

  TACT_CUDA_CHECK(cudaMemcpy(d_in.as<float>(), h_in.data(), size_t(opt.n) * sizeof(float), cudaMemcpyHostToDevice));
  TACT_CUDA_CHECK(cudaMemcpy(d_funcs.as<int>(), h_funcs.data(), size_t(spec.num_funcs) * sizeof(int), cudaMemcpyHostToDevice));
  TACT_CUDA_CHECK(cudaMemcpy(d_positions.as<int>(), spec.positions.data(), spec.positions.size() * sizeof(int), cudaMemcpyHostToDevice));
  TACT_CUDA_CHECK(cudaMemcpy(d_output_groups.as<int>(), spec.output_groups.data(), spec.output_groups.size() * sizeof(int), cudaMemcpyHostToDevice));
  TACT_CUDA_CHECK(cudaMemset(d_out.as<float>(), 0, size_t(spec.num_funcs) * opt.n * sizeof(float)));
  TACT_CUDA_CHECK(cudaMemset(d_ref.as<float>(), 0, size_t(spec.num_funcs) * opt.n * sizeof(float)));

  EngineContext ctx{&opt, spec, d_in.as<float>(), d_out.as<float>(), d_funcs.as<int>(),
                    d_positions.as<int>(), d_output_groups.as<int>()};
  EngineContext ref_ctx{&opt, spec, d_in.as<float>(), d_ref.as<float>(), d_funcs.as<int>(),
                        d_positions.as<int>(), d_output_groups.as<int>()};

  if (opt.validate || opt.debug) {
    auto ref_engine = makeBaselineEngine(ref_ctx, true);
    ref_engine->run();
    TACT_CUDA_CHECK(cudaDeviceSynchronize());
    if (opt.reference_compat) {
      int cos_output = opt.funcs.size() == 1 ? 0 : 1;
      TACT_CUDA_CHECK(cudaMemcpy(d_ref.as<float>() + size_t(cos_output) * opt.n,
                                 d_in.as<float>(), size_t(opt.n) * sizeof(float),
                                 cudaMemcpyDeviceToDevice));
    }
    // Historical scalar reference kernels overwrite D exactly like the old
    // programs. Restore the original input before warmup/measurement.
    TACT_CUDA_CHECK(cudaMemcpy(d_in.as<float>(), h_in.data(),
                               size_t(opt.n) * sizeof(float),
                               cudaMemcpyHostToDevice));
  }

  auto engine = makeSelectedEngine(opt, ctx);

  std::vector<float> times;
  std::vector<float> h2d_times;
  std::vector<float> kernel_times;
  std::vector<float> d2h_times;
  times.reserve(opt.iterations);
  h2d_times.reserve(opt.iterations);
  kernel_times.reserve(opt.iterations);
  d2h_times.reserve(opt.iterations);
  std::vector<float> h_timed_out(size_t(spec.num_funcs) * opt.n);
  SectionTimes section_sum;

  for (int i = 0; i < opt.warmup; ++i) {
    for (int r = 0; r < opt.inner_repeat; ++r) engine->run();
    TACT_CUDA_CHECK(cudaDeviceSynchronize());
  }

  for (int i = 0; i < opt.iterations; ++i) {
    // Match the historical TACT report: time input HtoD, the complete GPU
    // computation, and output DtoH separately on every measured iteration.
    auto h2d_start = std::chrono::high_resolution_clock::now();
    TACT_CUDA_CHECK(cudaMemcpy(d_in.as<float>(), h_in.data(),
                               size_t(opt.n) * sizeof(float),
                               cudaMemcpyHostToDevice));
    auto h2d_stop = std::chrono::high_resolution_clock::now();
    float h2d_ms = std::chrono::duration<float, std::milli>(h2d_stop - h2d_start).count();

    auto kernel_start = std::chrono::high_resolution_clock::now();
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
    TACT_CUDA_CHECK(cudaDeviceSynchronize());
    auto kernel_stop = std::chrono::high_resolution_clock::now();
    float kernel_ms = std::chrono::duration<float, std::milli>(kernel_stop - kernel_start).count();

    auto d2h_start = std::chrono::high_resolution_clock::now();
    // TripleTranscend* performs one DtoH copy per output array. Copy planes in
    // the same physical function-group order (cos, sin, tan for triple mode).
    for (int group = 0; group < spec.num_funcs; ++group) {
      int output_index = 0;
      while (spec.output_groups[output_index] != group) ++output_index;
      const float* copy_source = d_out.as<float>() + size_t(output_index) * opt.n;
      bool historical_scalar = opt.reference_compat &&
          (opt.engine == ENGINE_BASELINE_FAST || opt.engine == ENGINE_BASELINE_SP);
      if (historical_scalar && opt.funcs[output_index] == F_COS) {
        copy_source = d_in.as<float>();
      }
      TACT_CUDA_CHECK(cudaMemcpy(h_timed_out.data() + size_t(output_index) * opt.n,
                                 copy_source,
                                 size_t(opt.n) * sizeof(float),
                                 cudaMemcpyDeviceToHost));
    }
    auto d2h_stop = std::chrono::high_resolution_clock::now();
    float d2h_ms = std::chrono::duration<float, std::milli>(d2h_stop - d2h_start).count();

    float total_ms = h2d_ms + kernel_ms + d2h_ms;
    times.push_back(total_ms);
    h2d_times.push_back(h2d_ms);
    kernel_times.push_back(kernel_ms);
    d2h_times.push_back(d2h_ms);
    section_sum.baseline_ms += iter_sections.baseline_ms;
    section_sum.fused_kernel_ms += iter_sections.fused_kernel_ms;
    section_sum.build_power_ms += iter_sections.build_power_ms;
    section_sum.matmul_ms += iter_sections.matmul_ms;
    section_sum.postprocess_ms += iter_sections.postprocess_ms;
  }

  TACT_CUDA_CHECK(cudaDeviceSynchronize());

  RunResult result;
<<<<<<< HEAD
=======
  result.init_ms = std::chrono::duration<float, std::milli>(init_stop - init_start).count();
  result.avg_h2d_ms = std::accumulate(h2d_times.begin(), h2d_times.end(), 0.0f) / float(h2d_times.size());
  result.avg_kernel_ms = std::accumulate(kernel_times.begin(), kernel_times.end(), 0.0f) / float(kernel_times.size());
  result.avg_d2h_ms = std::accumulate(d2h_times.begin(), d2h_times.end(), 0.0f) / float(d2h_times.size());
>>>>>>> e30f8f9 (Changing SFU/SP/Tensore_Core kernels according to the real TACT)
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

  std::vector<float> h_out;
  std::vector<float> h_ref;
  if (opt.validate || opt.debug) {
    h_out = h_timed_out;
    h_ref.resize(size_t(spec.num_funcs) * opt.n);
    TACT_CUDA_CHECK(cudaMemcpy(h_ref.data(), d_ref.as<float>(), h_ref.size() * sizeof(float), cudaMemcpyDeviceToHost));
  }
  if (opt.validate) {
    RunResult v = computeValidation(h_out, h_ref, opt.n, spec.num_funcs, opt.threshold);
    result.max_abs = v.max_abs;
    result.max_rel = v.max_rel;
    result.failures = v.failures;
    result.func_max_abs = std::move(v.func_max_abs);
    result.func_max_rel = std::move(v.func_max_rel);
    result.func_failures = std::move(v.func_failures);
  }
  if (opt.debug) {
    if (opt.debug_file.empty()) {
      dumpDebug(std::cerr, opt, spec, h_in, h_out, h_ref);
    } else {
      std::ofstream debug_out(opt.debug_file);
      if (!debug_out) throw std::runtime_error("Could not open debug file: " + opt.debug_file);
      dumpDebug(debug_out, opt, spec, h_in, h_out, h_ref);
    }
  }

  return result;
}

}  // namespace tact
