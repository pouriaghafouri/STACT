#include "tact/matrix_builders.cuh"

namespace tact {

std::vector<__half> buildCoefficientB(const std::vector<FunctionId>& funcs,
                                      int terms,
                                      int ipr,
                                      const std::vector<int>& positions,
                                      const std::vector<int>& output_groups) {
  std::vector<__half> b(WMMA_K * OUT_PAD);
  for (auto& h : b) h = __float2half(0.0f);
  for (int f = 0; f < int(funcs.size()); ++f) {
    for (int s = 0; s < ipr; ++s) {
      int col = output_groups[f] * ipr + s;
      for (int t = 0; t < terms; ++t) {
        int k = positions[s * terms + t];
        b[k * OUT_PAD + col] = __float2half(coeffFor(funcs[f], t));
      }
    }
  }
  return b;
}

std::vector<__half> buildCoefficientBT(const std::vector<FunctionId>& funcs,
                                       int terms,
                                       int ipr,
                                       const std::vector<int>& positions,
                                       const std::vector<int>& output_groups) {
  std::vector<__half> bt(OUT_PAD * WMMA_K);
  for (auto& h : bt) h = __float2half(0.0f);
  for (int f = 0; f < int(funcs.size()); ++f) {
    for (int s = 0; s < ipr; ++s) {
      int row = output_groups[f] * ipr + s;
      for (int t = 0; t < terms; ++t) {
        int k = positions[s * terms + t];
        bt[row * WMMA_K + k] = __float2half(coeffFor(funcs[f], t));
      }
    }
  }
  return bt;
}

void buildCSRFromBT(const std::vector<__half>& bt,
                    std::vector<int>& rowPtr,
                    std::vector<int>& colInd,
                    std::vector<__half>& vals) {
  rowPtr.assign(OUT_PAD + 1, 0);
  colInd.clear();
  vals.clear();
  for (int r = 0; r < OUT_PAD; ++r) {
    for (int c = 0; c < WMMA_K; ++c) {
      float v = __half2float(bt[r * WMMA_K + c]);
      if (v != 0.0f) {
        colInd.push_back(c);
        vals.push_back(bt[r * WMMA_K + c]);
      }
    }
    rowPtr[r + 1] = int(colInd.size());
  }
}

}  // namespace tact
