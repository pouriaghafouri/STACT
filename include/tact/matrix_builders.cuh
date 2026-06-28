#pragma once
#include "tact/common.cuh"

namespace tact {

// Dense row-major logical B: K x N. Used by WMMA and by cuSPARSELt when opA=T.
std::vector<__half> buildCoefficientB(const std::vector<FunctionId>& funcs,
                                      int terms,
                                      int ipr,
                                      const std::vector<int>& positions);

// Dense row-major logical B^T: N x K. Used by cuSPARSE CSR SpMM.
std::vector<__half> buildCoefficientBT(const std::vector<FunctionId>& funcs,
                                       int terms,
                                       int ipr,
                                       const std::vector<int>& positions);

void buildCSRFromBT(const std::vector<__half>& bt,
                    std::vector<int>& rowPtr,
                    std::vector<int>& colInd,
                    std::vector<__half>& vals);

}  // namespace tact
