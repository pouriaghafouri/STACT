#pragma once
#include "tact/common.cuh"

namespace tact {

struct CoefficientLayout {
  // Canonical dense TACT matrix B[K,16] with sequential logical K rows.
  std::vector<__half> logical;
  // Physical B after applying the paired K permutation used by power matrix A.
  std::vector<__half> physical;
  // Full bijection: logical K row -> physical K row.
  std::vector<int> logical_to_physical;
};

// The single source of truth for every matrix engine. Dense mode produces an
// identity permutation. sparse24 mode preserves logical B while distributing
// its physical K rows into a 2:4-compatible arrangement.
CoefficientLayout buildCoefficientLayout(
    const std::vector<FunctionId>& funcs, int terms, int ipr,
    const std::vector<int>& physical_positions,
    const std::vector<int>& output_groups);

bool isCoefficientSparse24(const std::vector<__half>& physical_b);

// Build CSR(B^T), shape 16 x K, for cuSPARSE's B^T * A^T formulation.
void buildCoefficientCsrTranspose(const std::vector<__half>& b,
                                  std::vector<int>& row_ptr,
                                  std::vector<int>& col_ind,
                                  std::vector<__half>& values);

}  // namespace tact
