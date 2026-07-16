#include "tact/common.cuh"
#include "tact/matrix_builders.cuh"

#include <set>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

tact::Options parseOptions(std::vector<std::string> arguments) {
  std::vector<char*> argv;
  argv.reserve(arguments.size());
  for (std::string& argument : arguments) argv.push_back(argument.data());
  return tact::parseArgs(int(argv.size()), argv.data());
}

void checkCase(const std::vector<tact::FunctionId>& functions, int terms,
               tact::LayoutKind layout_kind) {
  tact::Options options;
  options.funcs = functions;
  options.terms = terms;
  options.layout = layout_kind;
  tact::ProblemSpec spec = tact::makeProblemSpec(options);
  tact::CoefficientLayout layout = tact::buildCoefficientLayout(
      functions, terms, spec.ipr, spec.positions, spec.output_groups);
  const int output_rows = int(functions.size()) * spec.ipr;
  const int k_rows = terms * spec.ipr;

  require(layout.logical.size() == tact::WMMA_K * tact::OUT_PAD,
          "logical coefficient shape mismatch");
  require(layout.physical.size() == layout.logical.size(),
          "physical coefficient shape mismatch");
  require(layout.logical_to_physical.size() == tact::WMMA_K,
          "K permutation size mismatch");

  std::set<int> mapped_rows(layout.logical_to_physical.begin(),
                            layout.logical_to_physical.end());
  require(mapped_rows.size() == tact::WMMA_K && *mapped_rows.begin() == 0 &&
              *mapped_rows.rbegin() == tact::WMMA_K - 1,
          "logical-to-physical K map is not a bijection");

  for (int logical_k = 0; logical_k < tact::WMMA_K; ++logical_k) {
    int physical_k = layout.logical_to_physical[logical_k];
    for (int col = 0; col < tact::OUT_PAD; ++col) {
      float logical_value =
          __half2float(layout.logical[logical_k * tact::OUT_PAD + col]);
      float physical_value =
          __half2float(layout.physical[physical_k * tact::OUT_PAD + col]);
      require(logical_value == physical_value,
              "coefficient changed during K permutation");
    }
  }

  std::vector<__half> logical_a(tact::WMMA_K, __float2half(0.0f));
  std::vector<__half> physical_a(tact::WMMA_K, __float2half(0.0f));
  for (int logical_k = 0; logical_k < tact::WMMA_K; ++logical_k) {
    __half value = __float2half(0.125f * float(logical_k + 1));
    logical_a[logical_k] = value;
    physical_a[layout.logical_to_physical[logical_k]] = value;
  }
  for (int col = 0; col < tact::OUT_PAD; ++col) {
    float logical_dot = 0.0f;
    float physical_dot = 0.0f;
    for (int k = 0; k < k_rows; ++k) {
      logical_dot += __half2float(logical_a[k]) *
                     __half2float(layout.logical[k * tact::OUT_PAD + col]);
      physical_dot += __half2float(physical_a[k]) *
                      __half2float(layout.physical[k * tact::OUT_PAD + col]);
    }
    require(logical_dot == physical_dot,
            "logical and physical coefficient products differ");
  }

  std::vector<int> row_ptr;
  std::vector<int> col_ind;
  std::vector<__half> values;
  tact::buildCoefficientCsrTranspose(
      layout.logical, output_rows, k_rows, row_ptr, col_ind, values);
  require(row_ptr.size() == size_t(output_rows + 1),
          "CSR row pointer shape mismatch");
  require(col_ind.size() == values.size() &&
              row_ptr.back() == int(values.size()),
          "CSR nonzero metadata mismatch");
  for (int row = 0; row < output_rows; ++row) {
    float csr_dot = 0.0f;
    float dense_dot = 0.0f;
    for (int entry = row_ptr[row]; entry < row_ptr[row + 1]; ++entry) {
      csr_dot += __half2float(values[entry]) *
                 __half2float(physical_a[col_ind[entry]]);
    }
    for (int k = 0; k < k_rows; ++k) {
      dense_dot += __half2float(layout.logical[k * tact::OUT_PAD + row]) *
                   __half2float(logical_a[k]);
    }
    require(csr_dot == dense_dot,
            "CSR(B^T) product differs from the physical dense product");
  }
}

}  // namespace

int main() {
  using namespace tact;
  const std::vector<std::vector<FunctionId>> even_cases = {
      {F_COS},
      {F_SIN, F_COS},
      {F_SIN, F_COS, F_TAN},
      {F_SIN, F_COS, F_TAN, F_SINH},
  };
  for (const auto& functions : even_cases) {
    for (int terms = 1; terms <= 8; ++terms) {
      checkCase(functions, terms, LAYOUT_DENSE);
      checkCase(functions, terms, LAYOUT_CSR);
    }
  }
  for (int terms = 1; terms <= 8; ++terms) {
    checkCase({F_EXP}, terms, LAYOUT_DENSE);
    checkCase({F_EXP}, terms, LAYOUT_CSR);
    checkCase({F_EXP, F_LOG1P}, terms, LAYOUT_DENSE);
    checkCase({F_EXP, F_LOG1P}, terms, LAYOUT_CSR);
  }

  require(isSfuSupported(F_SIN) && isSfuSupported(F_COS) &&
              isSfuSupported(F_TAN),
          "old-TACT SFU support list is incomplete");
  require(!isSfuSupported(F_EXP) && !isSfuSupported(F_LOG1P) &&
              !isSfuSupported(F_SINH) && !isSfuSupported(F_COSH) &&
              !isSfuSupported(F_TANH),
          "SFU support list contains a non-reference fallback");

  Options old_triple;
  old_triple.funcs = {F_SIN, F_COS, F_TAN};
  old_triple.terms = 3;
  old_triple.layout = LAYOUT_DENSE;
  ProblemSpec old_triple_spec = makeProblemSpec(old_triple);
  require(old_triple_spec.ipr == 5,
          "reference triple packing must retain five inputs per WMMA row");
  require(old_triple_spec.output_groups == std::vector<int>({1, 0, 2}),
          "reference triple columns must remain cos, sin, tan");
  for (int k = 0; k < 15; ++k) {
    require(old_triple_spec.positions[k] == k,
            "reference triple Tile A K ordering changed");
  }

  Options stress_options = parseOptions({
      "tact_bench", "--engine", "baseline_sp", "--functions", "sin,cos",
      "--terms", "3", "--stress-repeat", "17"});
  require(stress_options.engine == ENGINE_SP &&
              stress_options.stress_repeat == 17 &&
              stress_options.terms == 3,
          "old-TACT-only CLI contract is incorrect");

  Options sparse_options = parseOptions({
      "tact_bench", "--engine", "cusparse", "--functions", "sin,cos,tan",
      "--terms", "3", "--layout", "csr"});
  require(sparse_options.engine == ENGINE_CUSPARSE &&
              sparse_options.layout == LAYOUT_CSR,
          "cuSPARSE CSR layout selection is incorrect");

  bool validation_flag_rejected = false;
  try {
    (void)parseOptions({
        "tact_bench", "--engine", "wmma", "--validate"});
  } catch (const std::runtime_error&) {
    validation_flag_rejected = true;
  }
  require(validation_flag_rejected,
          "validation enable/disable flags must not be accepted");

  bool clean_rejected = false;
  try {
    (void)parseOptions({
        "tact_bench", "--engine", "baseline_fast", "--functions",
        "sin,cos,tan", "--clean"});
  } catch (const std::runtime_error&) {
    clean_rejected = true;
  }
  require(clean_rejected, "removed clean mode was still accepted");

  bool unsupported_sfu_rejected = false;
  try {
    (void)parseOptions({
        "tact_bench", "--engine", "baseline_fast", "--functions",
        "sin,cos,tan,sinh", "--stress-repeat", "2000"});
  } catch (const UnsupportedError&) {
    unsupported_sfu_rejected = true;
  }
  require(unsupported_sfu_rejected,
          "unsupported SFU stress fusion was accepted");

  bool incompatible_rejected = false;
  try {
    Options options;
    options.funcs = {F_SIN, F_EXP};
    (void)makeProblemSpec(options);
  } catch (const std::runtime_error&) {
    incompatible_rejected = true;
  }
  require(incompatible_rejected, "incompatible shared power bases were accepted");

  std::cout << "TACT matrix layout tests passed\n";
  return 0;
}
