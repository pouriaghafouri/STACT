#include "tact/common.cuh"

int main(int argc, char** argv) {
  try {
    tact::Options opt = tact::parseArgs(argc, argv);
    tact::RunResult r = tact::runBenchmark(opt);

    std::cout << "TACT benchmark result\n";
    std::cout << "  engine:        " << tact::engineName(opt.engine) << "\n";
    std::cout << "  layout:        " << tact::layoutName(opt.layout) << "\n";
    std::cout << "  functions:     " << tact::functionListName(opt.funcs) << "\n";
    std::cout << "  terms:         " << opt.terms << "\n";
    std::cout << "  N:             " << opt.n << " input values\n";
    std::cout << "  stress repeat: " << opt.stress_repeat;
    if (opt.engine == tact::ENGINE_CUSPARSE) {
      std::cout << " physical wide SpMM repetitions over 16x16 tile groups";
    } else if (opt.engine == tact::ENGINE_SPARSE_TC) {
      std::cout << " logical tile updates; 2 physical mma.sp.sync instructions per update";
    } else {
      std::cout << " physical core-operation repetitions";
    }
    std::cout << "\n";

    const bool canonical_single =
        opt.funcs.size() == 1 && opt.funcs[0] == tact::F_COS;
    const bool canonical_triple =
        opt.funcs.size() == 3 && opt.funcs[0] == tact::F_SIN &&
        opt.funcs[1] == tact::F_COS && opt.funcs[2] == tact::F_TAN;
    if (opt.engine == tact::ENGINE_SP || opt.engine == tact::ENGINE_SFU) {
      std::cout << "  kernel path:   old-TACT scalar state-chain kernel\n";
    } else if (opt.engine == tact::ENGINE_WMMA &&
               opt.layout == tact::LAYOUT_DENSE &&
               (canonical_single || canonical_triple)) {
      std::cout << "  kernel path:   compile-time old-TACT WMMA tile\n";
    } else if (opt.engine == tact::ENGINE_WMMA) {
      std::cout << "  kernel path:   generalized old-TACT WMMA tile\n";
    } else if (opt.engine == tact::ENGINE_CUSPARSE) {
      std::cout << "  kernel path:   wide cuSPARSE CSR(B^T) * [A_0^T|A_1^T|...], "
                   "16-column WMMA-equivalent tile groups\n";
    } else if (opt.engine == tact::ENGINE_SPARSE_TC) {
      std::cout << "  kernel path:   custom Ampere 2:4 sparse Tensor Core PTX; "
                   "register-resident stress mainloop\n";
    }

    if (opt.engine == tact::ENGINE_SPARSE_TC) {
      std::cout << "  sparse MMA/tile: "
                << (2LL * opt.stress_repeat)
                << " physical mma.sp.sync instructions\n";
    }
    std::cout << "  outputs/it:    " << r.timed_outputs << " function values\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  init ms:       " << r.init_ms << "\n";
    std::cout << "  HtoD avg ms:   " << r.avg_h2d_ms << "\n";
    std::cout << "  kernel avg ms: " << r.avg_kernel_ms << "\n";
    std::cout << "  DtoH avg ms:   " << r.avg_d2h_ms << "\n";
    std::cout << "  total avg ms:  " << r.avg_ms << "\n";
    std::cout << "  total min/max: " << r.min_ms << " / " << r.max_ms << " ms\n";

    if (opt.profile_sections) {
      std::cout << "  separate profile-pass ms:\n";
      if (r.avg_scalar_ms > 0.0f)
        std::cout << "    scalar kernel: " << r.avg_scalar_ms << "\n";
      if (r.avg_fused_kernel_ms > 0.0f)
        std::cout << "    fused matrix:  " << r.avg_fused_kernel_ms << "\n";
      if (r.avg_build_power_ms > 0.0f)
        std::cout << "    build/init C:  " << r.avg_build_power_ms << "\n";
      if (r.avg_matmul_ms > 0.0f)
        std::cout << "    sparse matmul:  " << r.avg_matmul_ms << "\n";
      if (r.avg_postprocess_ms > 0.0f)
        std::cout << "    postprocess:   " << r.avg_postprocess_ms << "\n";
      std::cout << "    profile-pass total: " << r.avg_profiled_ms << "\n";
    }

    std::cout << "  throughput:    " << r.throughput_mvals_s << " Mvalues/s\n";
    std::cout << std::scientific << std::setprecision(3);
    std::cout << "  validation:    ALWAYS ON ("
              << r.validated_outputs << " output values checked)\n";
    if (opt.engine == tact::ENGINE_SP || opt.engine == tact::ENGINE_SFU) {
      std::cout << "  validation ref: full deterministic device replay\n";
    } else {
      std::cout << "  validation ref: independent host FP16-rounded TACT model\n";
    }
    std::cout << "  max abs err:   " << r.max_abs << "\n";
    std::cout << "  max rel err:   " << r.max_rel << "\n";
    std::cout << "  failures:      " << r.failures
              << " @ threshold " << opt.threshold << "\n";
    for (size_t i = 0; i < opt.funcs.size(); ++i) {
      std::cout << "    " << tact::functionName(opt.funcs[i])
                << ": max_abs=" << r.func_max_abs[i]
                << ", max_rel=" << r.func_max_rel[i]
                << ", failures=" << r.func_failures[i] << "\n";
    }
    if (opt.engine == tact::ENGINE_CUSPARSE ||
        opt.engine == tact::ENGINE_SPARSE_TC) {
      std::cout << "  "
                << (opt.engine == tact::ENGINE_CUSPARSE
                        ? "cuSPARSE/WMMA parity:\n"
                        : "sparse-TC/WMMA parity:\n");
      std::cout << "    max abs err: " << r.parity_max_abs << "\n";
      std::cout << "    max rel err: " << r.parity_max_rel << "\n";
      std::cout << "    failures:    " << r.parity_failures
                << " @ threshold " << opt.threshold << "\n";
    }

    if (!opt.csv_path.empty()) tact::appendCsv(opt.csv_path, opt, r);
    return (r.failures == 0 && r.parity_failures == 0) ? 0 : 2;
  } catch (const tact::UnsupportedError& e) {
    std::cerr << "Unsupported: " << e.what() << "\n";
    return 3;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
