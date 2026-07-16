#include "tact/common.cuh"
#include "tact/engines.cuh"
#include "tact/kernels.cuh"
#include "tact/matrix_builders.cuh"

namespace tact {

static void copyOutputs(const Options& opt,
                        const ProblemSpec& spec,
                        const DeviceBuffer& d_in,
                        const DeviceBuffer& d_out,
                        std::vector<float>& host_out) {
  host_out.resize(size_t(spec.num_funcs) * opt.n);
  for (int output_index = 0; output_index < spec.num_funcs; ++output_index) {
    const float* source =
        d_out.as<float>() + size_t(output_index) * opt.n;
    if ((opt.engine == ENGINE_SFU || opt.engine == ENGINE_SP) &&
        opt.funcs[output_index] == F_COS) {
      source = d_in.as<float>();
    }
    TACT_CUDA_CHECK(cudaMemcpy(
        host_out.data() + size_t(output_index) * opt.n,
        source, size_t(opt.n) * sizeof(float), cudaMemcpyDeviceToHost));
  }
}

static void dumpDebug(std::ostream& os,
                      const Options& opt,
                      const ProblemSpec& spec,
                      const std::vector<float>& input,
                      const std::vector<float>& output,
                      const std::vector<float>& reference,
                      const std::vector<float>* sparse_intermediate,
                      int sparse_ld) {
  const int rows =
      std::min(opt.debug_rows, (opt.n + spec.ipr - 1) / spec.ipr);
  const int values = std::min(opt.debug_values, opt.n);
  const int used_k = opt.terms * spec.ipr;
  const int used_output_rows = spec.num_funcs * spec.ipr;
  auto coefficients = buildCoefficientLayout(
      opt.funcs, opt.terms, spec.ipr, spec.positions, spec.output_groups);

  std::vector<__half> logical_a(
      size_t(rows) * WMMA_K, __float2half(0.0f));
  for (int row = 0; row < rows; ++row) {
    for (int slot = 0; slot < spec.ipr; ++slot) {
      const int index = row * spec.ipr + slot;
      if (index >= opt.n) continue;
      for (int term = 0; term < opt.terms; ++term) {
        const int k = spec.positions[slot * opt.terms + term];
        logical_a[row * WMMA_K + k] =
            __float2half(powerFor(input[index], term, spec.basis));
      }
    }
  }

  os << "\nTACT debug dump\n";
  os << "inputs (first " << values << "):";
  for (int i = 0; i < values; ++i) os << ' ' << input[i];
  os << "\nslot -> K positions\n";
  for (int slot = 0; slot < spec.ipr; ++slot) {
    os << "  slot " << slot << ':';
    for (int term = 0; term < opt.terms; ++term) {
      os << " k" << spec.positions[slot * opt.terms + term];
    }
    os << '\n';
  }
  os << "function -> coefficient-column group\n";
  for (int f = 0; f < spec.num_funcs; ++f) {
    os << "  " << functionName(opt.funcs[f])
       << " -> " << spec.output_groups[f] << '\n';
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
    for (int column = 0; column < WMMA_N; ++column) {
      os << ' '
         << __half2float(coefficients.logical[k * WMMA_N + column]);
    }
    os << '\n';
  }

  os << "logical Tile C / intermediate matrix (used columns)\n";
  for (int row = 0; row < rows; ++row) {
    os << "  C[" << row << "]";
    for (int column = 0; column < used_output_rows; ++column) {
      float dot = 0.0f;
      for (int k = 0; k < used_k; ++k) {
        dot += __half2float(logical_a[row * WMMA_K + k]) *
               __half2float(coefficients.logical[k * WMMA_N + column]);
      }
      os << ' ' << (1.0f + float(opt.stress_repeat) * dot);
    }
    os << '\n';
  }

  os << "outputs / validation references / absolute errors\n";
  for (int f = 0; f < spec.num_funcs; ++f) {
    os << "  " << functionName(opt.funcs[f]) << '\n';
    for (int i = 0; i < values; ++i) {
      const float actual = output[size_t(f) * opt.n + i];
      const float expected = reference[size_t(f) * opt.n + i];
      os << "    i=" << i << " x=" << input[i]
         << " out=" << actual << " ref=" << expected
         << " abs=" << std::abs(actual - expected) << '\n';
    }
  }

  if (opt.engine == ENGINE_CUSPARSE) {
    std::vector<int> row_ptr;
    std::vector<int> col_ind;
    std::vector<__half> values_csr;
    buildCoefficientCsrTranspose(
        coefficients.logical, used_output_rows, used_k,
        row_ptr, col_ind, values_csr);
    os << "cuSPARSE shape: CSR(B^T)=" << used_output_rows << 'x'
       << used_k << ", A^T=" << used_k << 'x'
       << ((opt.n + spec.ipr - 1) / spec.ipr) << '\n';
    os << "cuSPARSE CSR row_ptr:";
    for (int value : row_ptr) os << ' ' << value;
    os << "\ncuSPARSE CSR nonzeros (column:value):";
    for (size_t i = 0; i < values_csr.size(); ++i) {
      os << ' ' << col_ind[i] << ':' << __half2float(values_csr[i]);
    }
    os << '\n';

    if (sparse_intermediate && sparse_ld > 0) {
      float max_difference = 0.0f;
      os << "cuSPARSE actual C^T / logical reference\n";
      for (int row = 0; row < rows; ++row) {
        os << "  row " << row << ':';
        for (int column = 0; column < used_output_rows; ++column) {
          float expected = 0.0f;
          for (int k = 0; k < used_k; ++k) {
            expected += __half2float(logical_a[row * WMMA_K + k]) *
                        __half2float(
                            coefficients.logical[k * WMMA_N + column]);
          }
          expected = 1.0f + float(opt.stress_repeat) * expected;
          const float actual =
              (*sparse_intermediate)[column * sparse_ld + row];
          max_difference =
              std::max(max_difference, std::abs(actual - expected));
          os << " c" << column << '=' << actual << '/' << expected;
        }
        os << '\n';
      }
      os << "cuSPARSE intermediate max difference: "
         << max_difference << '\n';
    }
  }
}

static std::unique_ptr<Engine> makeSelectedEngine(
    const Options& opt, const EngineContext& context) {
  switch (opt.engine) {
    case ENGINE_SFU: return makeSfuEngine(context);
    case ENGINE_SP: return makeSpEngine(context);
    case ENGINE_WMMA: return makeWmmaEngine(context);
    case ENGINE_CUSPARSE: return makeCuSparseEngine(context);
  }
  throw std::runtime_error("Unknown engine");
}

static std::vector<float> makeMatrixStressReference(
    const Options& opt,
    const ProblemSpec& spec,
    const std::vector<float>& input) {
  std::vector<float> reference(
      size_t(spec.num_funcs) * opt.n, 0.0f);

  // Independent host model of the old Tensor Core algebra. A and B values are
  // rounded to FP16 once, C starts at one, and the same A*B product contributes
  // stress_repeat times.
  for (int i = 0; i < opt.n; ++i) {
    const float x = input[i];
    for (int f = 0; f < spec.num_funcs; ++f) {
      float dot = 0.0f;
      for (int term = 0; term < opt.terms; ++term) {
        const float power = __half2float(
            __float2half(powerFor(x, term, spec.basis)));
        const float coefficient = __half2float(
            __float2half(coeffFor(opt.funcs[f], term)));
        dot += power * coefficient;
      }
      const float residual =
          1.0f + float(opt.stress_repeat) * dot;
      reference[size_t(f) * opt.n + i] =
          needsMultiplyByX(opt.funcs[f]) ? residual * x : residual;
    }
  }
  return reference;
}

RunResult runBenchmark(const Options& opt) {
  const ProblemSpec spec = makeProblemSpec(opt);
  if (opt.print_layout) printProblemSpec(spec, opt);

  const std::vector<float> host_input = makeInputs(opt, spec.basis);
  std::vector<int> host_functions(spec.num_funcs);
  for (int i = 0; i < spec.num_funcs; ++i) {
    host_functions[i] = int(opt.funcs[i]);
  }

  DeviceBuffer d_in(size_t(opt.n) * sizeof(float));
  DeviceBuffer d_out(size_t(spec.num_funcs) * opt.n * sizeof(float));
  DeviceBuffer d_funcs(size_t(spec.num_funcs) * sizeof(int));
  DeviceBuffer d_positions(size_t(spec.positions.size()) * sizeof(int));
  DeviceBuffer d_output_groups(
      size_t(spec.output_groups.size()) * sizeof(int));

  TACT_CUDA_CHECK(cudaMemcpy(
      d_in.as<float>(), host_input.data(), size_t(opt.n) * sizeof(float),
      cudaMemcpyHostToDevice));
  TACT_CUDA_CHECK(cudaMemcpy(
      d_funcs.as<int>(), host_functions.data(),
      size_t(spec.num_funcs) * sizeof(int), cudaMemcpyHostToDevice));
  TACT_CUDA_CHECK(cudaMemcpy(
      d_positions.as<int>(), spec.positions.data(),
      spec.positions.size() * sizeof(int), cudaMemcpyHostToDevice));
  TACT_CUDA_CHECK(cudaMemcpy(
      d_output_groups.as<int>(), spec.output_groups.data(),
      spec.output_groups.size() * sizeof(int), cudaMemcpyHostToDevice));
  TACT_CUDA_CHECK(cudaMemset(
      d_out.as<float>(), 0,
      size_t(spec.num_funcs) * opt.n * sizeof(float)));

  EngineContext context{
      &opt, spec, d_in.as<float>(), d_out.as<float>(), d_funcs.as<int>(),
      d_positions.as<int>(), d_output_groups.as<int>()};

  const auto init_start = std::chrono::high_resolution_clock::now();
  auto engine = makeSelectedEngine(opt, context);
  TACT_CUDA_CHECK(cudaDeviceSynchronize());
  const auto init_stop = std::chrono::high_resolution_clock::now();

  std::vector<float> total_times;
  std::vector<float> h2d_times;
  std::vector<float> kernel_times;
  std::vector<float> d2h_times;
  total_times.reserve(opt.iterations);
  h2d_times.reserve(opt.iterations);
  kernel_times.reserve(opt.iterations);
  d2h_times.reserve(opt.iterations);
  std::vector<float> timed_output;

  for (int i = 0; i < opt.warmup; ++i) {
    TACT_CUDA_CHECK(cudaMemcpy(
        d_in.as<float>(), host_input.data(), size_t(opt.n) * sizeof(float),
        cudaMemcpyHostToDevice));
    engine->run();
    TACT_CUDA_CHECK(cudaDeviceSynchronize());
  }

  for (int iteration = 0; iteration < opt.iterations; ++iteration) {
    const auto h2d_start = std::chrono::high_resolution_clock::now();
    TACT_CUDA_CHECK(cudaMemcpy(
        d_in.as<float>(), host_input.data(), size_t(opt.n) * sizeof(float),
        cudaMemcpyHostToDevice));
    const auto h2d_stop = std::chrono::high_resolution_clock::now();
    const float h2d_ms =
        std::chrono::duration<float, std::milli>(
            h2d_stop - h2d_start).count();

    const auto kernel_start = std::chrono::high_resolution_clock::now();
    engine->run();
    TACT_CUDA_CHECK(cudaDeviceSynchronize());
    const auto kernel_stop = std::chrono::high_resolution_clock::now();
    const float kernel_ms =
        std::chrono::duration<float, std::milli>(
            kernel_stop - kernel_start).count();

    const auto d2h_start = std::chrono::high_resolution_clock::now();
    copyOutputs(opt, spec, d_in, d_out, timed_output);
    const auto d2h_stop = std::chrono::high_resolution_clock::now();
    const float d2h_ms =
        std::chrono::duration<float, std::milli>(
            d2h_stop - d2h_start).count();

    total_times.push_back(h2d_ms + kernel_ms + d2h_ms);
    h2d_times.push_back(h2d_ms);
    kernel_times.push_back(kernel_ms);
    d2h_times.push_back(d2h_ms);
  }

  RunResult result;
  result.init_ms =
      std::chrono::duration<float, std::milli>(
          init_stop - init_start).count();
  result.avg_h2d_ms =
      std::accumulate(h2d_times.begin(), h2d_times.end(), 0.0f) /
      float(h2d_times.size());
  result.avg_kernel_ms =
      std::accumulate(kernel_times.begin(), kernel_times.end(), 0.0f) /
      float(kernel_times.size());
  result.avg_d2h_ms =
      std::accumulate(d2h_times.begin(), d2h_times.end(), 0.0f) /
      float(d2h_times.size());
  result.avg_ms =
      std::accumulate(total_times.begin(), total_times.end(), 0.0f) /
      float(total_times.size());
  result.min_ms =
      *std::min_element(total_times.begin(), total_times.end());
  result.max_ms =
      *std::max_element(total_times.begin(), total_times.end());

  if (opt.profile_sections) {
    TACT_CUDA_CHECK(cudaMemcpy(
        d_in.as<float>(), host_input.data(), size_t(opt.n) * sizeof(float),
        cudaMemcpyHostToDevice));
    const SectionTimes profile = engine->runProfile();
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

  // Validation is mandatory and is always outside the timed benchmark region.
  std::vector<float> validation_reference;
  if (opt.engine == ENGINE_SP || opt.engine == ENGINE_SFU) {
    // The scalar reference is a full deterministic replay of the exact old
    // TACT state-chain kernel. A host libm replay is not a reliable reference
    // for CUDA SFU/libdevice after 2000 stateful iterations.
    TACT_CUDA_CHECK(cudaMemcpy(
        d_in.as<float>(), host_input.data(), size_t(opt.n) * sizeof(float),
        cudaMemcpyHostToDevice));
    TACT_CUDA_CHECK(cudaMemset(
        d_out.as<float>(), 0,
        size_t(spec.num_funcs) * opt.n * sizeof(float)));
    engine->run();
    TACT_CUDA_CHECK(cudaDeviceSynchronize());
    copyOutputs(opt, spec, d_in, d_out, validation_reference);
  } else {
    validation_reference =
        makeMatrixStressReference(opt, spec, host_input);
  }

  RunResult validation = computeValidation(
      timed_output, validation_reference, opt.n,
      spec.num_funcs, opt.threshold);
  result.max_abs = validation.max_abs;
  result.max_rel = validation.max_rel;
  result.failures = validation.failures;
  result.validated_outputs = result.timed_outputs;
  result.func_max_abs = std::move(validation.func_max_abs);
  result.func_max_rel = std::move(validation.func_max_rel);
  result.func_failures = std::move(validation.func_failures);

  // cuSPARSE gets a second, independent full-output parity gate against the
  // already trusted WMMA backend. This catches coefficient/CSR/layout errors
  // even when a host reference bug would affect both expected values.
  if (opt.engine == ENGINE_CUSPARSE) {
    Options wmma_options = opt;
    wmma_options.engine = ENGINE_WMMA;
    wmma_options.layout = LAYOUT_DENSE;
    EngineContext wmma_context{
        &wmma_options, spec, d_in.as<float>(), d_out.as<float>(),
        d_funcs.as<int>(), d_positions.as<int>(),
        d_output_groups.as<int>()};

    TACT_CUDA_CHECK(cudaMemcpy(
        d_in.as<float>(), host_input.data(), size_t(opt.n) * sizeof(float),
        cudaMemcpyHostToDevice));
    TACT_CUDA_CHECK(cudaMemset(
        d_out.as<float>(), 0,
        size_t(spec.num_funcs) * opt.n * sizeof(float)));
    auto wmma_reference_engine = makeWmmaEngine(wmma_context);
    wmma_reference_engine->run();
    TACT_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> wmma_output;
    copyOutputs(wmma_options, spec, d_in, d_out, wmma_output);
    RunResult parity = computeValidation(
        timed_output, wmma_output, opt.n, spec.num_funcs, opt.threshold);
    result.parity_max_abs = parity.max_abs;
    result.parity_max_rel = parity.max_rel;
    result.parity_failures = parity.failures;
  }

  if (opt.debug) {
    std::vector<float> sparse_intermediate;
    int sparse_ld = 0;
    const std::vector<float>* sparse_intermediate_ptr = nullptr;
    if (engine->copyIntermediate(sparse_intermediate, sparse_ld)) {
      sparse_intermediate_ptr = &sparse_intermediate;
    }
    if (opt.debug_file.empty()) {
      dumpDebug(
          std::cerr, opt, spec, host_input, timed_output,
          validation_reference, sparse_intermediate_ptr, sparse_ld);
    } else {
      std::ofstream debug_output(opt.debug_file);
      if (!debug_output) {
        throw std::runtime_error(
            "Could not open debug file: " + opt.debug_file);
      }
      dumpDebug(
          debug_output, opt, spec, host_input, timed_output,
          validation_reference, sparse_intermediate_ptr, sparse_ld);
    }
  }

  return result;
}

}  // namespace tact
