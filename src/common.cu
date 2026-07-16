#include "tact/common.cuh"

namespace tact {

const char* functionName(FunctionId f) {
  switch (f) {
    case F_EXP: return "exp";
    case F_LOG1P: return "ln1p";
    case F_SIN: return "sin";
    case F_COS: return "cos";
    case F_TAN: return "tan";
    case F_SINH: return "sinh";
    case F_COSH: return "cosh";
    case F_TANH: return "tanh";
  }
  return "unknown";
}

std::string functionListName(const std::vector<FunctionId>& fs) {
  std::ostringstream oss;
  for (size_t i = 0; i < fs.size(); ++i) {
    if (i) oss << "+";
    oss << functionName(fs[i]);
  }
  return oss.str();
}

std::string engineName(EngineKind e) {
  switch (e) {
    case ENGINE_SFU: return "sfu";
    case ENGINE_SP: return "sp";
    case ENGINE_WMMA: return "wmma";
    case ENGINE_CUSPARSE: return "cusparse";
  }
  return "unknown";
}

std::string layoutName(LayoutKind layout) {
  switch (layout) {
    case LAYOUT_DENSE: return "dense";
    case LAYOUT_CSR: return "csr";
  }
  return "unknown";
}

static std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::tolower(c); });
  return s;
}

static std::vector<std::string> split(const std::string& s, char delim) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    if (!item.empty()) out.push_back(item);
  }
  return out;
}

FunctionId parseFunction(const std::string& raw) {
  std::string s = lower(raw);
  if (s == "exp" || s == "exponential" || s == "xponential") return F_EXP;
  if (s == "ln" || s == "log" || s == "log1p" || s == "ln1p" || s == "ln(x+1)") return F_LOG1P;
  if (s == "sin" || s == "sine") return F_SIN;
  if (s == "cos" || s == "cosine") return F_COS;
  if (s == "tan" || s == "tangent") return F_TAN;
  if (s == "sinh") return F_SINH;
  if (s == "cosh") return F_COSH;
  if (s == "tanh") return F_TANH;
  throw std::runtime_error("Unknown function: " + raw);
}

EngineKind parseEngine(const std::string& raw) {
  std::string s = lower(raw);
  if (s == "sfu" || s == "fast" || s == "baseline_fast") return ENGINE_SFU;
  if (s == "sp" || s == "libdevice" || s == "baseline_sp") return ENGINE_SP;
  if (s == "wmma" || s == "tensor" || s == "dense") return ENGINE_WMMA;
  if (s == "cusparse") return ENGINE_CUSPARSE;
  throw std::runtime_error("Unknown engine: " + raw);
}

LayoutKind parseLayout(const std::string& raw) {
  std::string s = lower(raw);
  if (s == "dense" || s == "sequential" || s == "normal") {
    return LAYOUT_DENSE;
  }
  if (s == "csr" || s == "sparse") return LAYOUT_CSR;
  throw std::runtime_error(
      "Unknown layout: " + raw + ". Use dense for WMMA or csr for cuSPARSE.");
}

bool isEvenGroup(FunctionId f) {
  return f == F_SIN || f == F_COS || f == F_TAN || f == F_SINH || f == F_COSH || f == F_TANH;
}

bool isMixedGroup(FunctionId f) {
  return f == F_EXP || f == F_LOG1P;
}

bool isSfuSupported(FunctionId f) {
  // Deliberately conservative: only the three intrinsics used by the old
  // TACT SFU reference programs are classified as SFU in this project.
  // exp/log/hyperbolic functions are not silently relabeled as SFU baselines.
  return f == F_SIN || f == F_COS || f == F_TAN;
}

BasisKind determineBasis(const std::vector<FunctionId>& funcs) {
  bool all_even = std::all_of(funcs.begin(), funcs.end(), isEvenGroup);
  bool all_mixed = std::all_of(funcs.begin(), funcs.end(), isMixedGroup);
  if (all_even) return BASIS_EVEN;
  if (all_mixed) return BASIS_MIXED;
  throw std::runtime_error("Requested functions are not compatible for one shared Tile A. Use only {sin,cos,tan,sinh,cosh,tanh} or only {exp,ln1p}.");
}

int inputsPerRow(int terms, int num_funcs) {
  if (terms < 1 || terms > 8) {
    throw std::runtime_error("terms must be in [1,8] for the fixed K=16 TACT tile.");
  }
  int by_k = WMMA_K / terms;
  int by_n = WMMA_N / num_funcs;
  int ipr = std::min(by_k, by_n);
  if (ipr < 1) throw std::runtime_error("Invalid terms/function count: no input can be packed into a 16x16 MMA tile.");
  return ipr;
}

int roundUp(int x, int a) { return ((x + a - 1) / a) * a; }

std::vector<int> makePositions(int terms, int ipr, LayoutKind layout) {
  if (layout != LAYOUT_DENSE && layout != LAYOUT_CSR) {
    throw std::runtime_error("Only dense and CSR layouts are supported.");
  }
  if (terms * ipr > WMMA_K) {
    throw std::runtime_error("terms * inputs_per_row exceeds K=16");
  }
  std::vector<int> positions(ipr * terms, -1);
  for (int slot = 0; slot < ipr; ++slot) {
    for (int term = 0; term < terms; ++term) {
      positions[slot * terms + term] = slot * terms + term;
    }
  }
  return positions;
}

ProblemSpec makeProblemSpec(const Options& opt) {
  ProblemSpec spec;
  spec.basis = determineBasis(opt.funcs);
  spec.num_funcs = int(opt.funcs.size());
  spec.ipr = inputsPerRow(opt.terms, spec.num_funcs);
  spec.positions = makePositions(opt.terms, spec.ipr, opt.layout);
  spec.output_groups.resize(spec.num_funcs);

  // TACT physical function-column order uses cosine columns [0,ipr), sine
  // columns [ipr,2*ipr), then tangent. Other
  // combinations use the same stable canonical order, which makes dual and
  // quad layouts direct subsets/extensions of the reference triple layout.
  auto rank = [](FunctionId f) {
    switch (f) {
      case F_COS: return 0;
      case F_SIN: return 1;
      case F_TAN: return 2;
      case F_SINH: return 3;
      case F_COSH: return 4;
      case F_TANH: return 5;
      case F_EXP: return 0;
      case F_LOG1P: return 1;
    }
    return 99;
  };
  std::vector<int> order(spec.num_funcs);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    return rank(opt.funcs[a]) < rank(opt.funcs[b]);
  });
  for (int group = 0; group < spec.num_funcs; ++group) {
    spec.output_groups[order[group]] = group;
  }
  return spec;
}

void printProblemSpec(const ProblemSpec& spec, const Options& opt) {
  std::cerr << "Basis: " << (spec.basis == BASIS_EVEN ? "even" : "mixed") << "\n";
  std::cerr << "Inputs per Tile-A row: " << spec.ipr << "\n";
  for (int s = 0; s < spec.ipr; ++s) {
    std::cerr << "slot " << s << ":";
    for (int t = 0; t < opt.terms; ++t) std::cerr << " k" << spec.positions[s * opt.terms + t];
    std::cerr << "\n";
  }
  std::cerr << "Coefficient column groups:";
  for (int f = 0; f < spec.num_funcs; ++f) {
    std::cerr << " " << functionName(opt.funcs[f]) << "->" << spec.output_groups[f];
  }
  std::cerr << "\n";
}

static bool containsFunction(const std::vector<FunctionId>& fs, FunctionId target) {
  return std::find(fs.begin(), fs.end(), target) != fs.end();
}

std::vector<float> makeInputs(const Options& opt, BasisKind basis) {
  // With no explicit interval, reproduce each old standalone program's host
  // initialization. A supplied --min/--max overrides it so a multi-engine
  // sweep can feed every method the identical input distribution.
  if (!opt.min_set && !opt.max_set) {
    std::vector<float> h(opt.n);
    bool matrix_engine = opt.engine == ENGINE_WMMA ||
                         opt.engine == ENGINE_CUSPARSE;
    if (matrix_engine) {
      // TACT matrix engines use the Tensor Core reference initialization.
      std::fill(h.begin(), h.end(), opt.funcs.size() == 1 ? 5.0f : 1.0f);
    } else {
      // Scalar SFU/SP stress kernels use pi * (2*i + 1).
      constexpr double pi = 3.14159265358979323846264338327950288;
      for (int i = 0; i < opt.n; ++i) {
        h[i] = static_cast<float>(pi * (2.0 * static_cast<double>(i) + 1.0));
      }
    }
    return h;
  }
  float xmin = opt.xmin, xmax = opt.xmax;
  if (!opt.min_set || !opt.max_set) {
    // These defaults are intentionally validation-safe for the truncated Taylor
    // series and FP16 Tensor-Core inputs used by the TACT engines. A wider range,
    // especially for tan/tanh close to pi/2, is a stress test and should be
    // requested explicitly with --min/--max and/or a looser threshold.
    if (basis == BASIS_EVEN) {
      if (containsFunction(opt.funcs, F_TAN) || containsFunction(opt.funcs, F_TANH)) {
        xmin = -0.5f;
        xmax =  0.5f;
      } else {
        xmin = -1.25f;
        xmax =  1.25f;
      }
    } else {
      // log1p Taylor is slow to converge for large |x|; keep the default small
      // so --threshold 1e-5 is meaningful for the default --terms 5.
      xmin = -0.1f;
      xmax =  0.1f;
    }
  }
  std::mt19937_64 rng(opt.seed);
  std::uniform_real_distribution<float> dist(xmin, xmax);
  std::vector<float> h(opt.n);
  for (int i = 0; i < opt.n; ++i) h[i] = dist(rng);
  return h;
}

RunResult computeValidation(const std::vector<float>& out,
                            const std::vector<float>& ref,
                            int n,
                            int num_funcs,
                            float threshold) {
  RunResult r;
  r.func_max_abs.assign(num_funcs, 0.0);
  r.func_max_rel.assign(num_funcs, 0.0);
  r.func_failures.assign(num_funcs, 0);
  for (int f = 0; f < num_funcs; ++f) {
    for (int i = 0; i < n; ++i) {
      double a = out[size_t(f) * n + i];
      double b = ref[size_t(f) * n + i];
      if (std::isnan(a) && std::isnan(b)) continue;
      if (std::isinf(a) && std::isinf(b) && std::signbit(a) == std::signbit(b)) continue;
      if (!std::isfinite(a) || !std::isfinite(b)) {
        r.failures++;
        r.func_failures[f]++;
        continue;
      }
      double abs_err = std::abs(a - b);
      double rel_err = abs_err / std::max(1e-12, std::abs(b));
      r.max_abs = std::max(r.max_abs, abs_err);
      r.max_rel = std::max(r.max_rel, rel_err);
      r.func_max_abs[f] = std::max(r.func_max_abs[f], abs_err);
      r.func_max_rel[f] = std::max(r.func_max_rel[f], rel_err);
      if (!(abs_err <= threshold || rel_err <= threshold)) {
        r.failures++;
        r.func_failures[f]++;
      }
    }
  }
  return r;
}

Options parseArgs(int argc, char** argv) {
  Options opt;
  bool layout_set = false;
  auto need = [&](int& i) -> std::string {
    if (i + 1 >= argc) {
      throw std::runtime_error(std::string("Missing value for ") + argv[i]);
    }
    return argv[++i];
  };

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--engine") opt.engine = parseEngine(need(i));
    else if (a == "--layout") {
      opt.layout = parseLayout(need(i));
      layout_set = true;
    }
    else if (a == "--functions" || a == "--funcs") {
      opt.funcs.clear();
      for (const auto& item : split(need(i), ',')) {
        opt.funcs.push_back(parseFunction(item));
      }
    } else if (a == "--terms") opt.terms = std::stoi(need(i));
    else if (a == "--n") opt.n = std::stoi(need(i));
    else if (a == "--warmup") opt.warmup = std::stoi(need(i));
    else if (a == "--iterations" || a == "--iters") opt.iterations = std::stoi(need(i));
    else if (a == "--stress-repeat") {
      opt.stress_repeat = std::stoi(need(i));
    } else if (a == "--threshold") opt.threshold = std::stof(need(i));
    else if (a == "--seed") opt.seed = std::stoull(need(i));
    else if (a == "--min") { opt.xmin = std::stof(need(i)); opt.min_set = true; }
    else if (a == "--max") { opt.xmax = std::stof(need(i)); opt.max_set = true; }
    else if (a == "--csv") opt.csv_path = need(i);
    else if (a == "--profile-sections") opt.profile_sections = true;
    else if (a == "--no-profile-sections") opt.profile_sections = false;
    else if (a == "--print-layout") opt.print_layout = true;
    else if (a == "--debug") opt.debug = true;
    else if (a == "--debug-values") { opt.debug = true; opt.debug_values = std::stoi(need(i)); }
    else if (a == "--debug-rows") { opt.debug = true; opt.debug_rows = std::stoi(need(i)); }
    else if (a == "--debug-file") { opt.debug = true; opt.debug_file = need(i); }
    else if (a == "--clean" || a == "--clean-math" || a == "--stress" || a == "--tact") {
      throw std::runtime_error(
          "This project has one execution model only: the original TACT stress "
          "workload. Remove --clean/--stress/--tact; use --stress-repeat to set "
          "the loop count (default 2000).");
    } else if (a == "--inner-repeat" || a == "--repeat") {
      throw std::runtime_error(
          "--inner-repeat was removed because it repeats the whole engine, not "
          "only the TACT core operation. Use --iterations for benchmark samples.");
    } else if (a == "--help" || a == "-h") {
      std::cout <<
        "TACT benchmark (original stress workload only)\n"
        "  --engine sfu|sp|wmma|cusparse\n"
        "  --functions cos | sin,cos | sin,cos,tan | sin,cos,tan,sinh\n"
        "  --terms <1..8> --layout dense|csr\n"
        "  --n <count> --warmup <k> --iterations <k>\n"
        "  --stress-repeat <k>  (default 2000; core operation only)\n"
        "  --threshold <eps> --min <x> --max <x> --csv results.csv\n"
        "  validation is always enabled\n"
        "  --profile-sections | --no-profile-sections\n"
        "  --print-layout --debug [--debug-values <k>] [--debug-rows <k>]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown option: " + a);
    }
  }

  if (opt.funcs.empty() || opt.funcs.size() > MAX_FUNCS) {
    throw std::runtime_error("Use between 1 and 4 functions.");
  }
  if ((opt.min_set && !opt.max_set) || (!opt.min_set && opt.max_set)) {
    throw std::runtime_error("Set both --min and --max or neither.");
  }
  if (opt.min_set && !(opt.xmin < opt.xmax)) {
    throw std::runtime_error("--min must be less than --max.");
  }
  if (opt.n <= 0 || opt.warmup < 0 || opt.iterations <= 0 ||
      opt.stress_repeat <= 0 || opt.debug_values <= 0 || opt.debug_rows <= 0) {
    throw std::runtime_error(
        "Invalid n/warmup/iterations/stress-repeat/debug count.");
  }

  if (opt.engine == ENGINE_CUSPARSE) {
    if (!layout_set) opt.layout = LAYOUT_CSR;
    if (opt.layout != LAYOUT_CSR) {
      throw std::runtime_error("cuSPARSE requires --layout csr.");
    }
  } else if (opt.layout != LAYOUT_DENSE) {
    throw std::runtime_error("SP, SFU, and WMMA require --layout dense.");
  }

  const bool single = opt.funcs == std::vector<FunctionId>{F_COS};
  const bool dual = opt.funcs == std::vector<FunctionId>{F_SIN, F_COS};
  const bool triple =
      opt.funcs == std::vector<FunctionId>{F_SIN, F_COS, F_TAN};
  const bool quad = opt.funcs ==
      std::vector<FunctionId>{F_SIN, F_COS, F_TAN, F_SINH};

  if (opt.engine == ENGINE_SFU) {
    if (!single && !dual && !triple) {
      throw UnsupportedError(
          "SFU old-TACT kernels support only cos; sin,cos; or sin,cos,tan. "
          "The quad case contains sinh and is intentionally unsupported.");
    }
    for (FunctionId function : opt.funcs) {
      if (!isSfuSupported(function)) {
        throw UnsupportedError(
            std::string("SFU engine does not support function ") +
            functionName(function));
      }
    }
  }
  if (opt.engine == ENGINE_SP && !single && !dual && !triple && !quad) {
    throw UnsupportedError(
        "SP old-TACT kernels support only cos; sin,cos; sin,cos,tan; or "
        "sin,cos,tan,sinh.");
  }

  // Matrix engines can accept any compatible 1..4-function set; this check
  // also rejects mixed power bases before device allocation.
  (void)determineBasis(opt.funcs);
  return opt;
}

void appendCsv(const std::string& path, const Options& opt, const RunResult& r) {
  bool exists = false;
  { std::ifstream f(path); exists = f.good(); }
  std::ofstream out(path, std::ios::app);
  if (!exists) {
    out << "engine,layout,functions,terms,n,warmup,iterations,stress_repeat,"
           "timed_values_per_iteration,timed_outputs_per_iteration,init_ms,"
           "avg_h2d_ms,avg_kernel_ms,avg_d2h_ms,avg_ms,min_ms,max_ms,"
           "avg_scalar_ms,avg_fused_kernel_ms,avg_build_power_ms,avg_matmul_ms,"
           "avg_postprocess_ms,avg_profiled_ms,throughput_mvalues_s,max_abs,"
           "max_rel,failures,validated_outputs,parity_max_abs,parity_max_rel,"
           "parity_failures,threshold\n";
  }
  out << engineName(opt.engine) << ','
      << layoutName(opt.layout) << ','
      << functionListName(opt.funcs) << ','
      << opt.terms << ',' << opt.n << ',' << opt.warmup << ','
      << opt.iterations << ',' << opt.stress_repeat << ','
      << std::setprecision(12) << r.timed_values << ',' << r.timed_outputs << ','
      << std::setprecision(8) << r.init_ms << ',' << r.avg_h2d_ms << ','
      << r.avg_kernel_ms << ',' << r.avg_d2h_ms << ',' << r.avg_ms << ','
      << r.min_ms << ',' << r.max_ms << ','
      << r.avg_scalar_ms << ',' << r.avg_fused_kernel_ms << ','
      << r.avg_build_power_ms << ',' << r.avg_matmul_ms << ','
      << r.avg_postprocess_ms << ',' << r.avg_profiled_ms << ','
      << r.throughput_mvals_s << ',' << r.max_abs << ',' << r.max_rel << ','
      << r.failures << ',' << r.validated_outputs << ','
      << r.parity_max_abs << ',' << r.parity_max_rel << ','
      << r.parity_failures << ',' << opt.threshold << "\n";
}

}  // namespace tact
