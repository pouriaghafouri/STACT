#include "tact/common.cuh"
#include "tact/matrix_builders.cuh"

#include <set>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
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
    for (int k = 0; k < tact::WMMA_K; ++k) {
      logical_dot += __half2float(logical_a[k]) *
                     __half2float(layout.logical[k * tact::OUT_PAD + col]);
      physical_dot += __half2float(physical_a[k]) *
                      __half2float(layout.physical[k * tact::OUT_PAD + col]);
    }
    require(logical_dot == physical_dot,
            "logical and physical coefficient products differ");
  }

  if (layout_kind == tact::LAYOUT_SPARSE24) {
    require(tact::isCoefficientSparse24(layout.physical),
            "physical coefficient matrix is not 2:4 valid");
  }

  std::vector<int> row_ptr;
  std::vector<int> col_ind;
  std::vector<__half> values;
  tact::buildCoefficientCsrTranspose(
      layout.physical, row_ptr, col_ind, values);
  require(row_ptr.size() == tact::OUT_PAD + 1,
          "CSR row pointer shape mismatch");
  require(col_ind.size() == values.size() &&
              row_ptr.back() == int(values.size()),
          "CSR nonzero metadata mismatch");
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
      checkCase(functions, terms, LAYOUT_SPARSE24);
    }
  }
  for (int terms = 1; terms <= 8; ++terms) {
    checkCase({F_EXP}, terms, LAYOUT_DENSE);
    checkCase({F_EXP}, terms, LAYOUT_SPARSE24);
    checkCase({F_EXP, F_LOG1P}, terms, LAYOUT_DENSE);
    checkCase({F_EXP, F_LOG1P}, terms, LAYOUT_SPARSE24);
  }

  require(isSfuSupported(F_EXP) && isSfuSupported(F_LOG1P) &&
              isSfuSupported(F_SIN) && isSfuSupported(F_COS) &&
              isSfuSupported(F_TAN),
          "explicit SFU support list is incomplete");
  require(!isSfuSupported(F_SINH) && !isSfuSupported(F_COSH) &&
              !isSfuSupported(F_TANH),
          "SFU support list contains a standard-math fallback");

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
