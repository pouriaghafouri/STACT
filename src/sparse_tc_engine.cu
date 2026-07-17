#include "tact/engines.cuh"
#include "tact/matrix_builders.cuh"

#include <array>
#include <cstdint>

namespace tact {
namespace {

constexpr int SPARSE_TC_K = 16;
constexpr int SPARSE_TC_COMPRESSED_K = SPARSE_TC_K / 2;

__device__ __forceinline__ uint32_t packHalf2(__half low, __half high) {
  return uint32_t(__half_as_ushort(low)) |
         (uint32_t(__half_as_ushort(high)) << 16);
}

// Direct Ampere structured-sparse Tensor Core instruction. One instruction
// computes D(16x8) = A_sparse(16x16) * B(16x8) + C(16x8).
__device__ __forceinline__ void sparseMmaM16N8K16(
    uint32_t a0, uint32_t a1,
    uint32_t b0, uint32_t b1,
    uint32_t metadata,
    float& d0, float& d1, float& d2, float& d3) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  float o0, o1, o2, o3;
  asm volatile(
      "mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3}, "
      "{%4, %5}, "
      "{%6, %7}, "
      "{%8, %9, %10, %11}, %12, 0x0;\n"
      : "=f"(o0), "=f"(o1), "=f"(o2), "=f"(o3)
      : "r"(a0), "r"(a1),
        "r"(b0), "r"(b1),
        "f"(d0), "f"(d1), "f"(d2), "f"(d3),
        "r"(metadata));
  d0 = o0;
  d1 = o1;
  d2 = o2;
  d3 = o3;
#else
  asm volatile("trap;");
#endif
}

// The compressed coefficient operand contains two FP16 values for each
// four-wide K group, i.e. eight stored values per row.
__device__ __forceinline__ uint32_t loadCompressedCoefficientPair(
    const __half* coefficients_t_compressed, int row, int k_group) {
  const int offset = row * SPARSE_TC_COMPRESSED_K + 2 * k_group;
  return packHalf2(coefficients_t_compressed[offset],
                   coefficients_t_compressed[offset + 1]);
}

// Matrix B of mma.sp is the permuted transpose of the original WMMA power
// tile. shared_power is stored as [input_row][permuted_k].
__device__ __forceinline__ uint32_t loadPowerPair(
    const __half* shared_power, int input_row, int k_pair_start) {
  const int offset = input_row * SPARSE_TC_K + k_pair_start;
  return packHalf2(shared_power[offset], shared_power[offset + 1]);
}

__device__ __forceinline__ void storeSparseResult(
    float residual,
    int coefficient_row,
    int input_row,
    int base_row,
    const float* __restrict__ input,
    float* __restrict__ output,
    const int* __restrict__ functions,
    const int* __restrict__ output_groups,
    int num_functions,
    int n,
    int inputs_per_row) {
#pragma unroll
  for (int f = 0; f < MAX_FUNCS; ++f) {
    if (f >= num_functions) break;
    const int slot = coefficient_row - output_groups[f] * inputs_per_row;
    if (slot < 0 || slot >= inputs_per_row) continue;

    const int index = (base_row + input_row) * inputs_per_row + slot;
    if (index >= n) return;
    const FunctionId function = static_cast<FunctionId>(functions[f]);
    const float x = input[index];
    output[size_t(f) * n + index] =
        needsMultiplyByX(function) ? residual * x : residual;
    return;
  }
}

// One warp owns one logical 16x16 TACT tile, matching the WMMA backend. The
// stress loop contains exactly two physical mma.sp.sync instructions per
// logical full-tile update: one instruction for columns 0..7 and one for
// columns 8..15. All operands and accumulators remain in registers.
__global__ __launch_bounds__(THREADS_PER_BLOCK)
void sparseTensorCoreKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int* __restrict__ functions,
    int num_functions,
    int n,
    int terms,
    int inputs_per_row,
    BasisKind basis,
    const int* __restrict__ physical_positions,
    const int* __restrict__ output_groups,
    const int* __restrict__ physical_to_sparse_k,
    const __half* __restrict__ coefficients_t_compressed,
    const uint32_t* __restrict__ metadata_by_row,
    int stress_repeat) {
  __shared__ __align__(16)
      __half shared_coefficients[WMMA_N * SPARSE_TC_COMPRESSED_K];
  __shared__ __align__(16)
      uint32_t shared_metadata[WMMA_N];
  __shared__ __align__(16)
      __half shared_power[WARPS_PER_BLOCK * WMMA_M * SPARSE_TC_K];

  for (int i = threadIdx.x;
       i < WMMA_N * SPARSE_TC_COMPRESSED_K; i += blockDim.x) {
    shared_coefficients[i] = coefficients_t_compressed[i];
  }
  for (int i = threadIdx.x; i < WMMA_N; i += blockDim.x) {
    shared_metadata[i] = metadata_by_row[i];
  }
  __syncthreads();

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int group = lane >> 2;
  const int thread_in_group = lane & 3;
  const int global_warp = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  const int base_row = global_warp * WMMA_M;
  const int base_input = base_row * inputs_per_row;

  __half* my_power =
      shared_power + warp_id * WMMA_M * SPARSE_TC_K;

  // Build A exactly as WMMA does, including FP16 rounding, but write each
  // physical K coordinate to the permutation that makes B^T legal 2:4.
  for (int i = lane; i < WMMA_M * SPARSE_TC_K; i += WARP_SIZE) {
    my_power[i] = __float2half(0.0f);
  }
  __syncwarp();

  if (lane < WMMA_M) {
    const int logical_row = base_row + lane;
    for (int slot = 0; slot < inputs_per_row; ++slot) {
      const int index = logical_row * inputs_per_row + slot;
      const float x = index < n ? input[index] : 0.0f;
      for (int term = 0; term < terms; ++term) {
        const int physical_k = physical_positions[slot * terms + term];
        const int sparse_k = physical_to_sparse_k[physical_k];
        const float power = index < n ? powerFor(x, term, basis) : 0.0f;
        my_power[lane * SPARSE_TC_K + sparse_k] = __float2half(power);
      }
    }
  }
  __syncwarp();

  if (base_input >= n) return;

  // m16n8k16 sparse-A fragment mapping:
  //   a0: coefficient row group,   K group thread_in_group
  //   a1: coefficient row group+8, K group thread_in_group
  const uint32_t a0 = loadCompressedCoefficientPair(
      shared_coefficients, group, thread_in_group);
  const uint32_t a1 = loadCompressedCoefficientPair(
      shared_coefficients, group + 8, thread_in_group);

  // With sparsity selector 0, lane T0 in every four-lane group contributes
  // metadata. The selected lane's low 16 bits describe coefficient row group;
  // its high 16 bits describe row group+8. Supplying the same register from
  // every lane is harmless and avoids a divergent metadata path.
  const uint32_t metadata =
      shared_metadata[group] | (shared_metadata[group + 8] << 16);

  // Dense B fragment mapping. group selects the N column (one packed input
  // row); thread_in_group selects adjacent K pairs.
  const int input_row_lo = group;
  const int input_row_hi = group + 8;
  const uint32_t b_lo0 = loadPowerPair(
      my_power, input_row_lo, 2 * thread_in_group);
  const uint32_t b_lo1 = loadPowerPair(
      my_power, input_row_lo, 8 + 2 * thread_in_group);
  const uint32_t b_hi0 = loadPowerPair(
      my_power, input_row_hi, 2 * thread_in_group);
  const uint32_t b_hi1 = loadPowerPair(
      my_power, input_row_hi, 8 + 2 * thread_in_group);

  float lo0 = 1.0f;
  float lo1 = 1.0f;
  float lo2 = 1.0f;
  float lo3 = 1.0f;
  float hi0 = 1.0f;
  float hi1 = 1.0f;
  float hi2 = 1.0f;
  float hi3 = 1.0f;

  // The loop count is runtime data and the two instructions are volatile PTX.
  // Consequently stress_repeat=2000 produces 4000 physical mma.sp.sync
  // instructions per logical tile, with exactly 2000 updates per output.
#pragma unroll 1
  for (int repeat = 0; repeat < stress_repeat; ++repeat) {
    sparseMmaM16N8K16(
        a0, a1, b_lo0, b_lo1, metadata,
        lo0, lo1, lo2, lo3);
    sparseMmaM16N8K16(
        a0, a1, b_hi0, b_hi1, metadata,
        hi0, hi1, hi2, hi3);
  }

  // mma.sp computes C^T = B^T * A^T. Emit the transposed logical result
  // directly in function-major output order; there is no global C matrix and
  // no separate postprocess kernel.
  const int input_pair_lo = 2 * thread_in_group;
  storeSparseResult(lo0, group, input_pair_lo, base_row,
                    input, output, functions, output_groups,
                    num_functions, n, inputs_per_row);
  storeSparseResult(lo1, group, input_pair_lo + 1, base_row,
                    input, output, functions, output_groups,
                    num_functions, n, inputs_per_row);
  storeSparseResult(lo2, group + 8, input_pair_lo, base_row,
                    input, output, functions, output_groups,
                    num_functions, n, inputs_per_row);
  storeSparseResult(lo3, group + 8, input_pair_lo + 1, base_row,
                    input, output, functions, output_groups,
                    num_functions, n, inputs_per_row);

  const int input_pair_hi = 8 + 2 * thread_in_group;
  storeSparseResult(hi0, group, input_pair_hi, base_row,
                    input, output, functions, output_groups,
                    num_functions, n, inputs_per_row);
  storeSparseResult(hi1, group, input_pair_hi + 1, base_row,
                    input, output, functions, output_groups,
                    num_functions, n, inputs_per_row);
  storeSparseResult(hi2, group + 8, input_pair_hi, base_row,
                    input, output, functions, output_groups,
                    num_functions, n, inputs_per_row);
  storeSparseResult(hi3, group + 8, input_pair_hi + 1, base_row,
                    input, output, functions, output_groups,
                    num_functions, n, inputs_per_row);
}

struct SparseCoefficientData {
  std::array<int, WMMA_K> physical_to_sparse_k{};
  std::vector<__half> compressed_bt;
  std::array<uint32_t, WMMA_N> metadata_by_row{};
};

// Return the ordered metadata nibble for exactly two selected positions in a
// four-wide group. The legal values are 0x4,0x8,0x9,0xc,0xd,0xe.
uint32_t encodeMetadataPair(int first, int second) {
  if (first < 0 || first >= 4 || second < 0 || second >= 4 ||
      first >= second) {
    throw std::runtime_error("invalid sparse Tensor Core metadata pair");
  }
  const uint32_t code = uint32_t(first | (second << 2));
  switch (code) {
    case 0x4:
    case 0x8:
    case 0x9:
    case 0xc:
    case 0xd:
    case 0xe:
      return code;
    default:
      throw std::runtime_error("unsupported sparse Tensor Core metadata code");
  }
}

SparseCoefficientData buildSparseCoefficientData(
    const EngineContext& ctx,
    const CoefficientLayout& coefficients) {
  SparseCoefficientData data;
  data.physical_to_sparse_k.fill(-1);

  const int terms = ctx.opt->terms;
  const int ipr = ctx.spec.ipr;
  const int used_k = terms * ipr;

  // The WMMA layout is slot-major: logical_k = slot*terms + term.
  // Reorder it to term-major: sparse_k = term*ipr + slot. For all legal TACT
  // configurations (terms<=8 and terms*ipr<=16), this places at most two terms
  // of any one input slot in every four-wide group, making every coefficient
  // row valid 2:4 without dropping or changing a coefficient.
  std::array<bool, WMMA_K> sparse_position_used{};
  for (int slot = 0; slot < ipr; ++slot) {
    for (int term = 0; term < terms; ++term) {
      const int logical_k = slot * terms + term;
      const int physical_k = ctx.spec.positions[logical_k];
      const int sparse_k = term * ipr + slot;
      if (physical_k < 0 || physical_k >= WMMA_K ||
          sparse_k < 0 || sparse_k >= WMMA_K ||
          data.physical_to_sparse_k[physical_k] != -1 ||
          sparse_position_used[sparse_k]) {
        throw std::runtime_error(
            "sparse_tc failed to construct a bijective K permutation");
      }
      data.physical_to_sparse_k[physical_k] = sparse_k;
      sparse_position_used[sparse_k] = true;
    }
  }

  // Preserve all zero-only padding K coordinates as a complete permutation.
  int next_sparse = 0;
  for (int logical_k = used_k; logical_k < WMMA_K; ++logical_k) {
    const int physical_k = coefficients.logical_to_physical[logical_k];
    while (next_sparse < WMMA_K && sparse_position_used[next_sparse]) {
      ++next_sparse;
    }
    if (physical_k < 0 || physical_k >= WMMA_K || next_sparse >= WMMA_K ||
        data.physical_to_sparse_k[physical_k] != -1) {
      throw std::runtime_error(
          "sparse_tc failed to place padding K coordinates");
    }
    data.physical_to_sparse_k[physical_k] = next_sparse;
    sparse_position_used[next_sparse] = true;
  }

  for (int physical_k = 0; physical_k < WMMA_K; ++physical_k) {
    if (data.physical_to_sparse_k[physical_k] < 0) {
      throw std::runtime_error("sparse_tc K permutation is incomplete");
    }
  }

  // Build dense permuted B^T temporarily, then compress each 2:4 group.
  std::array<__half, WMMA_N * WMMA_K> permuted_bt{};
  permuted_bt.fill(__float2half(0.0f));
  for (int physical_k = 0; physical_k < WMMA_K; ++physical_k) {
    const int sparse_k = data.physical_to_sparse_k[physical_k];
    for (int row = 0; row < WMMA_N; ++row) {
      permuted_bt[row * WMMA_K + sparse_k] =
          coefficients.physical[physical_k * WMMA_N + row];
    }
  }

  data.compressed_bt.assign(
      WMMA_N * SPARSE_TC_COMPRESSED_K, __float2half(0.0f));

  for (int row = 0; row < WMMA_N; ++row) {
    uint32_t row_metadata = 0u;
    for (int group = 0; group < WMMA_K / 4; ++group) {
      int selected[2] = {-1, -1};
      int selected_count = 0;
      for (int local = 0; local < 4; ++local) {
        const __half value = permuted_bt[row * WMMA_K + group * 4 + local];
        if (__half2float(value) != 0.0f) {
          if (selected_count >= 2) {
            std::ostringstream message;
            message << "sparse_tc term-major permutation is not 2:4 at row "
                    << row << ", K group " << group;
            throw std::runtime_error(message.str());
          }
          selected[selected_count++] = local;
        }
      }
      for (int local = 0; selected_count < 2 && local < 4; ++local) {
        bool already_selected = false;
        for (int i = 0; i < selected_count; ++i) {
          already_selected |= selected[i] == local;
        }
        if (!already_selected) selected[selected_count++] = local;
      }
      if (selected[0] > selected[1]) std::swap(selected[0], selected[1]);

      const uint32_t nibble =
          encodeMetadataPair(selected[0], selected[1]);
      row_metadata |= nibble << (4 * group);

      const int compressed_offset =
          row * SPARSE_TC_COMPRESSED_K + 2 * group;
      data.compressed_bt[compressed_offset] =
          permuted_bt[row * WMMA_K + group * 4 + selected[0]];
      data.compressed_bt[compressed_offset + 1] =
          permuted_bt[row * WMMA_K + group * 4 + selected[1]];
    }
    data.metadata_by_row[row] = row_metadata;
  }

  // Independent structural reconstruction: decode every metadata nibble,
  // rebuild the 16x16 sparse operand, and compare all 256 FP16 entries.
  for (int row = 0; row < WMMA_N; ++row) {
    std::array<__half, WMMA_K> reconstructed{};
    reconstructed.fill(__float2half(0.0f));
    for (int group = 0; group < WMMA_K / 4; ++group) {
      const uint32_t code =
          (data.metadata_by_row[row] >> (4 * group)) & 0xfu;
      const int first = int(code & 0x3u);
      const int second = int((code >> 2) & 0x3u);
      (void)encodeMetadataPair(first, second);
      const int compressed_offset =
          row * SPARSE_TC_COMPRESSED_K + 2 * group;
      reconstructed[group * 4 + first] =
          data.compressed_bt[compressed_offset];
      reconstructed[group * 4 + second] =
          data.compressed_bt[compressed_offset + 1];
    }
    for (int sparse_k = 0; sparse_k < WMMA_K; ++sparse_k) {
      const float expected =
          __half2float(permuted_bt[row * WMMA_K + sparse_k]);
      const float actual = __half2float(reconstructed[sparse_k]);
      if (actual != expected) {
        std::ostringstream message;
        message << "sparse_tc compression reconstruction mismatch at ["
                << row << ',' << sparse_k << "]: got " << actual
                << ", expected " << expected;
        throw std::runtime_error(message.str());
      }
    }
  }

  return data;
}

void launchSparseTensorCore(
    const EngineContext& ctx,
    const int* physical_to_sparse_k,
    const __half* compressed_coefficients_t,
    const uint32_t* metadata_by_row,
    cudaStream_t stream) {
  const int rows =
      (ctx.opt->n + ctx.spec.ipr - 1) / ctx.spec.ipr;
  const int warp_tiles = (rows + WMMA_M - 1) / WMMA_M;
  const int grid =
      (warp_tiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

  sparseTensorCoreKernel<<<grid, THREADS_PER_BLOCK, 0, stream>>>(
      ctx.d_in, ctx.d_out, ctx.d_funcs, ctx.spec.num_funcs,
      ctx.opt->n, ctx.opt->terms, ctx.spec.ipr, ctx.spec.basis,
      ctx.d_positions, ctx.d_output_groups, physical_to_sparse_k,
      compressed_coefficients_t, metadata_by_row,
      ctx.opt->stress_repeat);
  TACT_CUDA_CHECK(cudaGetLastError());
}

class SparseTensorCoreEngine final : public Engine {
 public:
  explicit SparseTensorCoreEngine(EngineContext ctx) : ctx_(ctx) {
    if (ctx_.opt->layout != LAYOUT_DENSE) {
      throw std::runtime_error(
          "sparse_tc requires --layout dense; its sparse operand is an "
          "internal 2:4 Tensor Core fragment, not a CSR matrix");
    }

    int device = 0;
    TACT_CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    TACT_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    if (properties.major < 8) {
      throw UnsupportedError(
          "sparse_tc requires an Ampere-or-newer GPU with mma.sp.sync "
          "support (compute capability 8.0+)");
    }

    const CoefficientLayout coefficients = buildCoefficientLayout(
        ctx_.opt->funcs, ctx_.opt->terms, ctx_.spec.ipr,
        ctx_.spec.positions, ctx_.spec.output_groups);
    const SparseCoefficientData sparse =
        buildSparseCoefficientData(ctx_, coefficients);

    d_physical_to_sparse_k_.allocate(WMMA_K * sizeof(int));
    d_compressed_coefficients_t_.allocate(
        sparse.compressed_bt.size() * sizeof(__half));
    d_metadata_by_row_.allocate(WMMA_N * sizeof(uint32_t));

    TACT_CUDA_CHECK(cudaMemcpy(
        d_physical_to_sparse_k_.as<int>(),
        sparse.physical_to_sparse_k.data(), WMMA_K * sizeof(int),
        cudaMemcpyHostToDevice));
    TACT_CUDA_CHECK(cudaMemcpy(
        d_compressed_coefficients_t_.as<__half>(),
        sparse.compressed_bt.data(),
        sparse.compressed_bt.size() * sizeof(__half),
        cudaMemcpyHostToDevice));
    TACT_CUDA_CHECK(cudaMemcpy(
        d_metadata_by_row_.as<uint32_t>(),
        sparse.metadata_by_row.data(), WMMA_N * sizeof(uint32_t),
        cudaMemcpyHostToDevice));
  }

  void run(cudaStream_t stream = nullptr) override {
    launchSparseTensorCore(
        ctx_, d_physical_to_sparse_k_.as<int>(),
        d_compressed_coefficients_t_.as<__half>(),
        d_metadata_by_row_.as<uint32_t>(), stream);
  }

  SectionTimes runProfile(cudaStream_t stream = nullptr) override {
    SectionTimes times;
    GpuTimer timer;
    timer.tic(stream);
    run(stream);
    times.fused_kernel_ms = timer.toc(stream);
    times.total_ms = times.fused_kernel_ms;
    return times;
  }

 private:
  EngineContext ctx_;
  DeviceBuffer d_physical_to_sparse_k_;
  DeviceBuffer d_compressed_coefficients_t_;
  DeviceBuffer d_metadata_by_row_;
};

}  // namespace

std::unique_ptr<Engine> makeSparseTensorCoreEngine(
    const EngineContext& ctx) {
  return std::make_unique<SparseTensorCoreEngine>(ctx);
}

}  // namespace tact
