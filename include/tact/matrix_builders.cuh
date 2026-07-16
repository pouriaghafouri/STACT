#pragma once
#include "tact/common.cuh"

namespace tact {

struct CoefficientLayout {
  // Canonical dense TACT coefficient matrix B[K,16].
  std::vector<__half> logical;
  // Physical view used by matrix engines; currently identical to logical.
  std::vector<__half> physical;
  std::vector<int> logical_to_physical;
};

// The single source of truth for WMMA and cuSPARSE coefficient matrices.
CoefficientLayout buildCoefficientLayout(
    const std::vector<FunctionId>& funcs, int terms, int ipr,
    const std::vector<int>& physical_positions,
    const std::vector<int>& output_groups);


// Build the used submatrix of CSR(B^T) for cuSPARSE's B^T * A^T
// formulation. output_rows and k_rows omit zero-only padding rows/columns.
void buildCoefficientCsrTranspose(const std::vector<__half>& b,
                                  int output_rows,
                                  int k_rows,
                                  std::vector<int>& row_ptr,
                                  std::vector<int>& col_ind,
                                  std::vector<__half>& values);

}  // namespace tact
