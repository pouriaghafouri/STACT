#include "tact/matrix_builders.cuh"

namespace tact {

CoefficientLayout buildCoefficientLayout(
    const std::vector<FunctionId>& funcs, int terms, int ipr,
    const std::vector<int>& physical_positions,
    const std::vector<int>& output_groups) {
  if (physical_positions.size() != size_t(terms * ipr)) {
    throw std::runtime_error("physical K-position count does not match terms*inputs_per_row");
  }

  CoefficientLayout layout;
  layout.logical.assign(WMMA_K * OUT_PAD, __float2half(0.0f));
  layout.physical.assign(WMMA_K * OUT_PAD, __float2half(0.0f));
  layout.logical_to_physical.assign(WMMA_K, -1);

  bool physical_used[WMMA_K] = {};
  int used_k = terms * ipr;
  for (int logical_k = 0; logical_k < used_k; ++logical_k) {
    int physical_k = physical_positions[logical_k];
    if (physical_k < 0 || physical_k >= WMMA_K || physical_used[physical_k]) {
      throw std::runtime_error("physical K positions must be a unique in-range mapping");
    }
    layout.logical_to_physical[logical_k] = physical_k;
    physical_used[physical_k] = true;
  }
  int next_physical = 0;
  for (int logical_k = used_k; logical_k < WMMA_K; ++logical_k) {
    while (physical_used[next_physical]) ++next_physical;
    layout.logical_to_physical[logical_k] = next_physical;
    physical_used[next_physical] = true;
  }

  for (int f = 0; f < int(funcs.size()); ++f) {
    for (int s = 0; s < ipr; ++s) {
      int col = output_groups[f] * ipr + s;
      for (int t = 0; t < terms; ++t) {
        int logical_k = s * terms + t;
        layout.logical[logical_k * OUT_PAD + col] =
            __float2half(coeffFor(funcs[f], t));
      }
    }
  }

  for (int logical_k = 0; logical_k < WMMA_K; ++logical_k) {
    int physical_k = layout.logical_to_physical[logical_k];
    for (int col = 0; col < OUT_PAD; ++col) {
      layout.physical[physical_k * OUT_PAD + col] =
          layout.logical[logical_k * OUT_PAD + col];
    }
  }
  return layout;
}

void buildCoefficientCsrTranspose(const std::vector<__half>& b,
                                  int output_rows,
                                  int k_rows,
                                  std::vector<int>& row_ptr,
                                  std::vector<int>& col_ind,
                                  std::vector<__half>& values) {
  if (b.size() != size_t(WMMA_K * OUT_PAD)) {
    throw std::runtime_error("coefficient matrix must have shape 16x16");
  }
  if (output_rows < 1 || output_rows > OUT_PAD ||
      k_rows < 1 || k_rows > WMMA_K) {
    throw std::runtime_error("invalid used coefficient submatrix dimensions");
  }

  row_ptr.assign(output_rows + 1, 0);
  col_ind.clear();
  values.clear();
  for (int row = 0; row < output_rows; ++row) {
    for (int k = 0; k < k_rows; ++k) {
      __half entry = b[k * OUT_PAD + row];
      if (__half2float(entry) != 0.0f) {
        col_ind.push_back(k);
        values.push_back(entry);
      }
    }
    row_ptr[row + 1] = int(col_ind.size());
  }
}

}  // namespace tact
