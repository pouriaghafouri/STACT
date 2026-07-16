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
                      const std::vector<float>& ref,
                      const std::vector<float>* sparse_intermediate,
                      int sparse_ld) {
  const int rows = std::min(opt.debug_rows, (opt.n + spec.ipr - 1) / spec.ipr);
  const int values = std::min(opt.debug_values, opt.n);
  auto coefficients = buildCoefficientLayout(
      opt.funcs, opt.terms, spec.ipr, spec.positions, spec.output_groups);

  std::vector<__half> logical_a(size_t(rows) * WMMA_K, __float2half(0.0f));
  std::vector<__half> physical_a(size_t(rows) * WMMA_K, __float2half(0.0f));
  for (int row = 0; row < rows; ++row) {
    for (int slot = 0; slot < spec.ipr; ++slot) {
      int idx = row * spec.ipr + slot;
      if (idx >= opt.n) continue;
      for (int t = 0; t < opt.terms; ++t) {
        int logical_k = slot * opt.terms + t;
        int physical_k = spec.positions[logical_k];
        __half power = __float2half(powerFor(in[idx], t, spec.basis));
        logical_a[row * WMMA_K + logical_k] = power;
        physical_a[row * WMMA_K + physical_k] = power;
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
  os << "logical K -> physical K:";
  for (int k = 0; k < WMMA_K; ++k) {
    os << ' ' << k << "->" << coefficients.logical_to_physical[k];
  }
  os << '\n';
  os << "function -> coefficient-column group\n";
  for (int f = 0; f < spec.num_funcs; ++f) {
    os << "  " << functionName(opt.funcs[f]) << " -> " << spec.output_groups[f] << '\n';
  }

  os << "logical Tile A / power matrix (" << rows << "x16 shown)\n";
  for (int row = 0; row < rows; ++row) {
    os << "  A[" << row << "]";
    for (int k = 0; k < WMMA_K; ++k) {
      os << ' ' << __half2float(logical_a[row * WMMA_K + k]);
    }
    os << '\n';
  }
  os << "logical Tile B / coefficient matrix (16x16)\n";
  for (int k = 0; k < WMMA_K; ++k) {
    os << "  B[" << k << "]";
    for (int col = 0; col < WMMA_N; ++col) {
      os << ' ' << __half2float(coefficients.logical[k * WMMA_N + col]);
    }
    os << '\n';
  }

  if (opt.layout == LAYOUT_SPARSE24) {
    os << "physical 2:4 Tile A / permuted power matrix\n";
    for (int row = 0; row < rows; ++row) {
      os << "  A24[" << row << "]";
      for (int k = 0; k < WMMA_K; ++k) {
        os << ' ' << __half2float(physical_a[row * WMMA_K + k]);
      }
      os << '\n';
    }
    os << "physical 2:4 Tile B / permuted coefficient matrix\n";
    for (int k = 0; k < WMMA_K; ++k) {
      os << "  B24[" << k << "]";
      for (int col = 0; col < WMMA_N; ++col) {
        os << ' ' << __half2float(coefficients.physical[k * WMMA_N + col]);
      }
      os << '\n';
    }
  }

  os << "logical Tile C / intermediate matrix (used columns)\n";
  float max_permutation_difference = 0.0f;
  for (int row = 0; row < rows; ++row) {
    os << "  C[" << row << "]";
    for (int col = 0; col < spec.num_funcs * spec.ipr; ++col) {
      float logical_dot = 0.0f;
      float physical_dot = 0.0f;
      for (int k = 0; k < WMMA_K; ++k) {
        logical_dot += __half2float(logical_a[row * WMMA_K + k]) *
                       __half2float(coefficients.logical[k * WMMA_N + col]);
        physical_dot += __half2float(physical_a[row * WMMA_K + k]) *
                        __half2float(coefficients.physical[k * WMMA_N + col]);
      }
      max_permutation_difference = std::max(
          max_permutation_difference, std::abs(logical_dot - physical_dot));
      const float c = 1.0f + opt.stress_repeat * physical_dot;
      os << ' ' << c;
    }
    os << '\n';
  }
  os << "logical/physical max dot difference: "
     << max_permutation_difference << '\n';

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
    std::vector<int> row_ptr, col_ind;
    std::vector<__half> vals;
    buildCoefficientCsrTranspose(coefficients.physical, row_ptr, col_ind, vals);
    os << "cuSPARSE CSR row_ptr:";
    for (int v : row_ptr) os << ' ' << v;
    os << "\ncuSPARSE CSR nonzeros (column:value):";
    for (size_t i = 0; i < vals.size(); ++i) {
      os << ' ' << col_ind[i] << ':' << __half2float(vals[i]);
    }
    os << '\n';
    if (sparse_intermediate && sparse_ld > 0) {
      float max_intermediate_difference = 0.0f;
      os << "cuSPARSE actual C^T / dense logical reference\n";
      for (int row = 0; row < rows; ++row) {
        os << "  row " << row << ':';
        for (int col = 0; col < spec.num_funcs * spec.ipr; ++col) {
          float expected = 0.0f;
          for (int k = 0; k < WMMA_K; ++k) {
            expected += __half2float(physical_a[row * WMMA_K + k]) *
                        __half2float(
                            coefficients.physical[k * WMMA_N + col]);
          }
          expected = 1.0f + opt.stress_repeat * expected;
          float actual = (*sparse_intermediate)[col * sparse_ld + row];
          max_intermediate_difference = std::max(
              max_intermediate_difference, std::abs(actual - expected));
          os << " c" << col << '=' << actual << "/" << expected;
        }
        os << '\n';
      }
      os << "cuSPARSE intermediate max difference: "
         << max_intermediate_difference << '\n';
    }
  }
  if (opt.layout == LAYOUT_SPARSE24) {
    os << "cuSPARSELt structured operand is physical B24 (Kx16 row-major); opA=T.\n";
    for (int col = 0; col < spec.num_funcs * spec.ipr; ++col) {
      os << "  col " << col << " 2:4 counts:";
      for (int group = 0; group < 4; ++group) {
        int count = 0;
        for (int k = group * 4; k < group * 4 + 4; ++k) {
          if (__half2float(coefficients.physical[k * WMMA_N + col]) != 0.0f) {
            ++count;
          }
        }
        os << ' ' << count;
      }
      os << '\n';
    }
    os << "  2:4 validity: "
       << (isCoefficientSparse24(coefficients.physical) ? "valid" : "INVALID")
       << '\n';
  }
}

static std::unique_ptr<Engine> makeSelectedEngine(const Options& opt, const EngineContext& ctx) {
  switch (opt.engine) {
    case ENGINE_SFU: return makeSfuEngine(ctx);
    case ENGINE_SP: return makeSpEngine(ctx);
    case ENGINE_WMMA: return makeWmmaEngine(ctx);
    case ENGINE_CUSPARSE: return makeCuSparseEngine(ctx);
    case ENGINE_CUSPARSELT: return makeCuSparseLtEngine(ctx);
  }
  throw std::runtime_error("Unknown engine");
}

static std::vector<float> makeStressReference(
    const Options& opt, const ProblemSpec& spec,
    const std::vector<float>& input) {
  std::vector<float> ref(size_t(spec.num_funcs) * opt.n, 0.0f);

  if (opt.engine == ENGINE_SP || opt.engine == ENGINE_SFU) {
    for (int i = 0; i < opt.n; ++i) {
      float state = input[i];
      float sine = 0.0f;
      float tangent = 0.0f;
      float hyperbolic_sine = 0.0f;
      for (int j = 0; j < opt.stress_repeat; ++j) {
        const float offset = float(j);
        if (spec.num_funcs == 1) {
          state = std::cos(state + offset);
        } else {
          sine = std::sin(state + offset);
          state = std::cos(state + offset);
          if (spec.num_funcs >= 3) tangent = std::tan(state + offset);
          if (spec.num_funcs >= 4) {
            hyperbolic_sine = std::sinh(state + offset);
          }
        }
      }
      if (spec.num_funcs == 1) {
        ref[i] = state;
      } else {
        ref[i] = sine;
        ref[size_t(opt.n) + i] = state;
        if (spec.num_funcs >= 3) {
          ref[size_t(2) * opt.n + i] = tangent;
        }
        if (spec.num_funcs >= 4) {
          ref[size_t(3) * opt.n + i] = hyperbolic_sine;
        }
      }
    }
    return ref;
  }

  // Matrix reference: exactly the old Tensor Core algebra. A and B values are
  // rounded to FP16 once, C starts at one, and only A*B is repeated.
  for (int i = 0; i < opt.n; ++i) {
    const float x = input[i];
    for (int f = 0; f < spec.num_funcs; ++f) {
      float dot = 0.0f;
      for (int t = 0; t < opt.terms; ++t) {
        const float power = __half2float(
            __float2half(powerFor(x, t, spec.basis)));
        const float coeff = __half2float(
            __float2half(coeffFor(opt.funcs[f], t)));
        dot += power * coeff;
      }
      const float residual = 1.0f + float(opt.stress_repeat) * dot;
      ref[size_t(f) * opt.n + i] =
          needsMultiplyByX(opt.funcs[f]) ? residual * x : residual;
    }
  }
  return ref;
}

RunResult runBenchmark(const Options& opt) {
  ProblemSpec spec = makeProblemSpec(opt);
  if (opt.print_layout) printProblemSpec(spec, opt);

  const auto h_in = makeInputs(opt, spec.basis);
  std::vector<int> h_funcs(spec.num_funcs);
  for (int i = 0; i < spec.num_funcs; ++i) h_funcs[i] = int(opt.funcs[i]);

  DeviceBuffer d_in(size_t(opt.n) * sizeof(float));
  DeviceBuffer d_out(size_t(spec.num_funcs) * opt.n * sizeof(float));
  DeviceBuffer d_funcs(size_t(spec.num_funcs) * sizeof(int));
  DeviceBuffer d_positions(size_t(spec.positions.size()) * sizeof(int));
  DeviceBuffer d_output_groups(size_t(spec.output_groups.size()) * sizeof(int));

  TACT_CUDA_CHECK(cudaMemcpy(d_in.as<float>(), h_in.data(),
                             size_t(opt.n) * sizeof(float),
                             cudaMemcpyHostToDevice));
  TACT_CUDA_CHECK(cudaMemcpy(d_funcs.as<int>(), h_funcs.data(),
                             size_t(spec.num_funcs) * sizeof(int),
                             cudaMemcpyHostToDevice));
  TACT_CUDA_CHECK(cudaMemcpy(d_positions.as<int>(), spec.positions.data(),
                             spec.positions.size() * sizeof(int),
                             cudaMemcpyHostToDevice));
  TACT_CUDA_CHECK(cudaMemcpy(d_output_groups.as<int>(),
                             spec.output_groups.data(),
                             spec.output_groups.size() * sizeof(int),
                             cudaMemcpyHostToDevice));
  TACT_CUDA_CHECK(cudaMemset(d_out.as<float>(), 0,
                             size_t(spec.num_funcs) * opt.n * sizeof(float)));

  EngineContext ctx{&opt, spec, d_in.as<float>(), d_out.as<float>(),
                    d_funcs.as<int>(), d_positions.as<int>(),
                    d_output_groups.as<int>()};

  const auto init_start = std::chrono::high_resolution_clock::now();
  auto engine = makeSelectedEngine(opt, ctx);
  TACT_CUDA_CHECK(cudaDeviceSynchronize());
  const auto init_stop = std::chrono::high_resolution_clock::now();

  std::vector<float> times;
  std::vector<float> h2d_times;
  std::vector<float> kernel_times;
  std::vector<float> d2h_times;
  times.reserve(opt.iterations);
  h2d_times.reserve(opt.iterations);
  kernel_times.reserve(opt.iterations);
  d2h_times.reserve(opt.iterations);
  std::vector<float> h_timed_out(size_t(spec.num_funcs) * opt.n);

  // Scalar kernels overwrite d_in, so every warmup starts from the same host
  // input. Matrix engines receive the same reset for a fair common path.
  for (int i = 0; i < opt.warmup; ++i) {
    TACT_CUDA_CHECK(cudaMemcpy(d_in.as<float>(), h_in.data(),
                               size_t(opt.n) * sizeof(float),
                               cudaMemcpyHostToDevice));
    engine->run();
    TACT_CUDA_CHECK(cudaDeviceSynchronize());
  }

  for (int i = 0; i < opt.iterations; ++i) {
    const auto h2d_start = std::chrono::high_resolution_clock::now();
    TACT_CUDA_CHECK(cudaMemcpy(d_in.as<float>(), h_in.data(),
                               size_t(opt.n) * sizeof(float),
                               cudaMemcpyHostToDevice));
    const auto h2d_stop = std::chrono::high_resolution_clock::now();
    const float h2d_ms =
        std::chrono::duration<float, std::milli>(h2d_stop - h2d_start).count();

    const auto kernel_start = std::chrono::high_resolution_clock::now();
    // Main benchmark timing always uses the uninstrumented engine path.
    // Section profiling is a separate diagnostic pass below so phase-event
    // synchronizations cannot distort the latency comparison.
    engine->run();
    TACT_CUDA_CHECK(cudaDeviceSynchronize());
    const auto kernel_stop = std::chrono::high_resolution_clock::now();
    const float kernel_ms =
        std::chrono::duration<float, std::milli>(kernel_stop - kernel_start).count();

    const auto d2h_start = std::chrono::high_resolution_clock::now();
    for (int output_index = 0; output_index < spec.num_funcs; ++output_index) {
      const float* source =
          d_out.as<float>() + size_t(output_index) * opt.n;
      if ((opt.engine == ENGINE_SFU || opt.engine == ENGINE_SP) &&
          opt.funcs[output_index] == F_COS) {
        source = d_in.as<float>();
      }
      TACT_CUDA_CHECK(cudaMemcpy(
          h_timed_out.data() + size_t(output_index) * opt.n,
          source, size_t(opt.n) * sizeof(float), cudaMemcpyDeviceToHost));
    }
    const auto d2h_stop = std::chrono::high_resolution_clock::now();
    const float d2h_ms =
        std::chrono::duration<float, std::milli>(d2h_stop - d2h_start).count();

    const float total_ms = h2d_ms + kernel_ms + d2h_ms;
    times.push_back(total_ms);
    h2d_times.push_back(h2d_ms);
    kernel_times.push_back(kernel_ms);
    d2h_times.push_back(d2h_ms);
  }

  RunResult result;
  result.init_ms =
      std::chrono::duration<float, std::milli>(init_stop - init_start).count();
  result.avg_h2d_ms =
      std::accumulate(h2d_times.begin(), h2d_times.end(), 0.0f) /
      float(h2d_times.size());
  result.avg_kernel_ms =
      std::accumulate(kernel_times.begin(), kernel_times.end(), 0.0f) /
      float(kernel_times.size());
  result.avg_d2h_ms =
      std::accumulate(d2h_times.begin(), d2h_times.end(), 0.0f) /
      float(d2h_times.size());
  result.avg_ms = std::accumulate(times.begin(), times.end(), 0.0f) /
                  float(times.size());
  result.min_ms = *std::min_element(times.begin(), times.end());
  result.max_ms = *std::max_element(times.begin(), times.end());
  if (opt.profile_sections) {
    // One separate diagnostic pass. It is intentionally excluded from avg_ms,
    // min/max, and avg_kernel_ms.
    TACT_CUDA_CHECK(cudaMemcpy(d_in.as<float>(), h_in.data(),
                               size_t(opt.n) * sizeof(float),
                               cudaMemcpyHostToDevice));
    SectionTimes profile = engine->runProfile();
    TACT_CUDA_CHECK(cudaDeviceSynchronize());
    result.avg_scalar_ms = profile.scalar_ms;
    result.avg_fused_kernel_ms = profile.fused_kernel_ms;
    result.avg_build_power_ms = profile.build_power_ms;
    result.avg_matmul_ms = profile.matmul_ms;
    result.avg_postprocess_ms = profile.postprocess_ms;
    result.avg_profiled_ms = profile.total_ms;
  }
  result.timed_values = double(opt.n);
  result.timed_outputs = double(opt.n) * double(spec.num_funcs);
  result.throughput_mvals_s =
      result.timed_outputs / (double(result.avg_ms) * 1000.0);

  std::vector<float> h_ref;
  if (opt.validate || opt.debug) {
    h_ref = makeStressReference(opt, spec, h_in);
  }
  if (opt.validate) {
    RunResult validation = computeValidation(
        h_timed_out, h_ref, opt.n, spec.num_funcs, opt.threshold);
    result.max_abs = validation.max_abs;
    result.max_rel = validation.max_rel;
    result.failures = validation.failures;
    result.func_max_abs = std::move(validation.func_max_abs);
    result.func_max_rel = std::move(validation.func_max_rel);
    result.func_failures = std::move(validation.func_failures);
  }

  if (opt.debug) {
    std::vector<float> sparse_intermediate;
    int sparse_ld = 0;
    const std::vector<float>* sparse_intermediate_ptr = nullptr;
    if (engine->copyIntermediate(sparse_intermediate, sparse_ld)) {
      sparse_intermediate_ptr = &sparse_intermediate;
    }
    if (opt.debug_file.empty()) {
      dumpDebug(std::cerr, opt, spec, h_in, h_timed_out, h_ref,
                sparse_intermediate_ptr, sparse_ld);
    } else {
      std::ofstream debug_out(opt.debug_file);
      if (!debug_out) {
        throw std::runtime_error(
            "Could not open debug file: " + opt.debug_file);
      }
      dumpDebug(debug_out, opt, spec, h_in, h_timed_out, h_ref,
                sparse_intermediate_ptr, sparse_ld);
    }
  }

  return result;
}

}  // namespace tact
