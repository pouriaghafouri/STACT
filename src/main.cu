#include "tact/common.cuh"

int main(int argc, char** argv) {
  try {
    tact::Options opt = tact::parseArgs(argc, argv);
    tact::RunResult r = tact::runBenchmark(opt);

    std::cout << "TACT benchmark result\n";
    std::cout << "  engine:      " << tact::engineName(opt.engine) << "\n";
    std::cout << "  layout:      " << tact::layoutName(opt.layout) << "\n";
    std::cout << "  functions:   " << tact::functionListName(opt.funcs) << "\n";
    std::cout << "  terms:       " << opt.terms << "\n";
    std::cout << "  N:           " << opt.n << " input x values per engine run\n";
    std::cout << "  inner repeat:" << opt.inner_repeat << " engine run(s) per timed iteration\n";
    std::cout << "  outputs/it:  " << r.timed_outputs << " function values per timed iteration\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  init ms:     " << r.init_ms << " (engine setup/descriptors/static coefficients)\n";
    std::cout << "  avg ms:      " << r.avg_ms << "\n";
    std::cout << "  min/max ms:  " << r.min_ms << " / " << r.max_ms << "\n";
    if (opt.profile_sections) {
      std::cout << "  section avg ms per timed iteration:\n";
      if (r.avg_baseline_ms > 0.0f) std::cout << "    baseline kernel: " << r.avg_baseline_ms << "\n";
      if (r.avg_fused_kernel_ms > 0.0f) std::cout << "    fused WMMA kernel: " << r.avg_fused_kernel_ms << "\n";
      if (r.avg_build_power_ms > 0.0f) std::cout << "    build power matrix: " << r.avg_build_power_ms << "\n";
      if (r.avg_matmul_ms > 0.0f) std::cout << "    matrix multiply:    " << r.avg_matmul_ms << "\n";
      if (r.avg_postprocess_ms > 0.0f) std::cout << "    postprocess/write:  " << r.avg_postprocess_ms << "\n";
    }
    std::cout << "  throughput:  " << r.throughput_mvals_s << " Mvalues/s\n";
    if (opt.validate) {
      std::cout << std::scientific << std::setprecision(3);
      std::cout << "  max abs err: " << r.max_abs << "\n";
      std::cout << "  max rel err: " << r.max_rel << "\n";
      std::cout << "  failures:    " << r.failures << " @ threshold " << opt.threshold << "\n";
      if (!r.func_failures.empty()) {
        std::cout << "  per-function validation:\n";
        for (size_t i = 0; i < opt.funcs.size(); ++i) {
          std::cout << "    " << tact::functionName(opt.funcs[i])
                    << ": max_abs=" << r.func_max_abs[i]
                    << ", max_rel=" << r.func_max_rel[i]
                    << ", failures=" << r.func_failures[i] << "\n";
        }
      }
    }
    if (!opt.csv_path.empty()) tact::appendCsv(opt.csv_path, opt, r);
    return r.failures == 0 ? 0 : 2;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
