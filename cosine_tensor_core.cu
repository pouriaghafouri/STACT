#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_fp16.h>
#include <mma.h>

// Compile: nvcc -arch=sm_86 -O2 cosine_tensor_32x8x16.cu -o cosine_32x8x16

// ═══════════════════════════════════════════════════════════════════════════════
// WMMA shape: (M=32, N=8, K=16) with FP16 inputs, FP32 accumulator
//
// Per-warp tile: 32 rows × 8 output columns = 256 elements
//
// A: 32×32, row-major, stride=32
//   row r = [x0²,x0⁴,x0⁶,x0⁸ | x1²,...,x1⁸ | ... | x7²,...,x7⁸]
//            cols 0-3            cols 4-7             cols 28-31
//
// B: 32×8, block-diagonal, row-major, stride=8
//   8 blocks of 4×1; block i at rows [4i..4i+3], column i:
//     [-1/2!, +1/4!, -1/6!, +1/8!]ᵀ
//
// C[r][i] = 1 + dot(A[r,4i:4i+4], B_col_i) = cos_taylor(x_i)
//   (the 1 comes from fill_fragment(C_frag, 1.0f))
//
// K-split (2 steps of K=16):
//   k=0: A[:,  0:16] × B[ 0:16, :] → inputs 0-3
//   k=1: A[:,16:32]  × B[16:32, :] → inputs 4-7
// ═══════════════════════════════════════════════════════════════════════════════

#define WARP_SIZE        32
#define NUM_INPUTS        8    // inputs packed per A-row
#define NUM_TERMS         4    // Taylor terms per input (x², x⁴, x⁶, x⁸)
#define WMMA_M           32
#define WMMA_N            8
#define WMMA_K           16
#define K_TOTAL          32    // NUM_INPUTS * NUM_TERMS
#define K_STEPS           2    // K_TOTAL / WMMA_K
#define WARPS_PER_BLOCK   8
#define M           9000000

using namespace nvcuda;

// ─── CPU wall-clock timer ────────────────────────────────────────────────────
static double cpu_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

// ─── GPU event timer macros ──────────────────────────────────────────────────
#define GPU_TIMER_DECL(n)    cudaEvent_t n##_s, n##_e
#define GPU_TIMER_CREATE(n)  cudaEventCreate(&n##_s); cudaEventCreate(&n##_e)
#define GPU_TIMER_START(n)   cudaEventRecord(n##_s)
#define GPU_TIMER_STOP(n)    cudaEventRecord(n##_e)
#define GPU_TIMER_MS(n,out)  cudaEventSynchronize(n##_e); \
                             cudaEventElapsedTime(&out, n##_s, n##_e)
#define GPU_TIMER_DESTROY(n) cudaEventDestroy(n##_s); cudaEventDestroy(n##_e)

// ─── Kernel ──────────────────────────────────────────────────────────────────
__global__ void cosine_kernel(float * __restrict__ d_D)
{
    const int warp_id     = threadIdx.x / WARP_SIZE;
    const int lane        = threadIdx.x % WARP_SIZE;
    const int warp_global = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const int D_base      = warp_global * WMMA_M * WMMA_N;  // 256 elems/warp
    if (D_base >= M) return;

    // ══════════════════════════════════════════════════════════════════════════
    // Shared Memory Layout  (no float staging — convert in registers)
    // ──────────────────────────────────────────────────────────────────────────
    //   s_A_fp16 [WARPS_PER_BLOCK][WMMA_M][K_TOTAL]  = [8][32][32] halfs
    //            = 8192 × 2 B = 16384 B
    //   s_B_fp16 [K_TOTAL][WMMA_N]                   = [32][8]  halfs
    //            = 256  × 2 B =   512 B
    //   s_C      [WARPS_PER_BLOCK][WMMA_M][WMMA_N]   = [8][32][8]  floats
    //            = 2048 × 4 B =  8192 B
    //   ─────────────────────────────────────────────────────
    //   Total    = 25088 B ≈ 24.5 KB  (well under 48 KB)
    // ══════════════════════════════════════════════════════════════════════════
    __shared__ half  s_A_fp16[WARPS_PER_BLOCK * WMMA_M * K_TOTAL];
    __shared__ half  s_B_fp16[K_TOTAL * WMMA_N];
    __shared__ float s_C     [WARPS_PER_BLOCK * WMMA_M * WMMA_N];

    // Convenience pointers for this warp
    half  *my_A = s_A_fp16 + warp_id * WMMA_M * K_TOTAL; // [WMMA_M][K_TOTAL]
    float *my_C = s_C      + warp_id * WMMA_M * WMMA_N;  // [WMMA_M][WMMA_N]

    // ══════════════════════════════════════════════════════════════════════════
    // Fill A in FP16 directly (no float smem needed)
    // ──────────────────────────────────────────────────────────────────────────
    // Lane `lane` fills row `lane` of this warp's A tile.
    // The 8 inputs for that row start at global index D_base + lane*8.
    // ══════════════════════════════════════════════════════════════════════════

    // Zero the row first
    half *row_A = my_A + lane * K_TOTAL;
    #pragma unroll
    for (int c = 0; c < K_TOTAL; c++) row_A[c] = __float2half(0.0f);

    // Fill powers
    #pragma unroll
    for (int i = 0; i < NUM_INPUTS; i++) {
        int D_idx = D_base + lane * NUM_INPUTS + i;
        float x  = (D_idx < M) ? d_D[D_idx] : 0.0f;
        float x2 = x * x;
        float x4 = x2 * x2;
        int col  = i * NUM_TERMS;
        row_A[col + 0] = __float2half(x2);
        row_A[col + 1] = __float2half(x4);
        row_A[col + 2] = __float2half(x4 * x2);   // x6
        row_A[col + 3] = __float2half(x4 * x4);   // x8
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Fill B in FP16 directly (warp 0 only; each lane owns one row)
    // ──────────────────────────────────────────────────────────────────────────
    // B is 32×8 block-diagonal.  Row `lane`:
    //   block_i  = lane / 4   → which output column is non-zero
    //   term_idx = lane % 4   → which Taylor coefficient
    // ══════════════════════════════════════════════════════════════════════════
    if (warp_id == 0) {
        const float coeffs[NUM_TERMS] = {
            -1.0f /     2.0f,   // -1/2!
             1.0f /    24.0f,   // +1/4!
            -1.0f /   720.0f,   // -1/6!
             1.0f / 40320.0f    // +1/8!
        };
        int block_i  = lane / NUM_TERMS;
        int term_idx = lane % NUM_TERMS;

        half *row_B = s_B_fp16 + lane * WMMA_N;
        #pragma unroll
        for (int c = 0; c < WMMA_N; c++)
            row_B[c] = __float2half(0.0f);
        row_B[block_i] = __float2half(coeffs[term_idx]);
    }

    __syncthreads();

    // ── Debug print: A and B matrices (block 0, thread 0 only) ───────────────
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("\n╔══════════════════════════════════════════════════╗\n");
        printf("║  WMMA (%d×%d×%d) FP16→FP32  smem=24.5 KB        ║\n",
               WMMA_M, WMMA_N, WMMA_K);
        printf("╚══════════════════════════════════════════════════╝\n");

        printf("\n=== Matrix A (warp 0, rows 0-3 of %d×%d) ===\n", WMMA_M, K_TOTAL);
        for (int r = 0; r < 4; r++) {
            printf("r%02d: ", r);
            for (int c = 0; c < K_TOTAL; c++)
                printf("%7.3f", __half2float(s_A_fp16[r * K_TOTAL + c]));
            printf("\n");
        }

        printf("\n=== Matrix B (%d×%d, block-diagonal) ===\n", K_TOTAL, WMMA_N);
        for (int r = 0; r < K_TOTAL; r++) {
            printf("r%02d [in%d t%d]: ", r, r/NUM_TERMS, r%NUM_TERMS);
            for (int c = 0; c < WMMA_N; c++)
                printf("%9.5f", __half2float(s_B_fp16[r * WMMA_N + c]));
            printf("\n");
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // WMMA Computation
    // ══════════════════════════════════════════════════════════════════════════
    wmma::fragment<wmma::matrix_a,    WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag;
    wmma::fragment<wmma::matrix_b,    WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> B_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> C_frag;

    // Seed with 1.0f — the degree-0 term of cos Taylor series
    wmma::fill_fragment(C_frag, 1.0f);

    #pragma unroll
    for (int k = 0; k < K_STEPS; k++) {
        // A: column offset k*WMMA_K, leading dim = K_TOTAL
        wmma::load_matrix_sync(A_frag,
                               my_A + k * WMMA_K,
                               K_TOTAL);
        // B: row offset k*WMMA_K, leading dim = WMMA_N
        wmma::load_matrix_sync(B_frag,
                               s_B_fp16 + k * WMMA_K * WMMA_N,
                               WMMA_N);

        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
    }

    wmma::store_matrix_sync(my_C, C_frag, WMMA_N, wmma::mem_row_major);

    __syncthreads();

    // ── Debug print: C matrix (block 0, thread 0 only) ───────────────────────
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("\n=== Matrix C (%d×%d) — all values should ≈ -1.0 ===\n",
               WMMA_M, WMMA_N);
        for (int r = 0; r < WMMA_M; r++) {
            printf("r%02d: ", r);
            for (int c = 0; c < WMMA_N; c++)
                printf("%9.5f", s_C[r * WMMA_N + c]);
            printf("\n");
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Write Back: C[lane][0..7] → D[D_base + lane*8 .. D_base + lane*8 + 7]
    // ══════════════════════════════════════════════════════════════════════════
    float *C_row = my_C + lane * WMMA_N;
    #pragma unroll
    for (int i = 0; i < WMMA_N; i++) {
        int D_idx = D_base + lane * WMMA_N + i;
        if (D_idx < M)
            d_D[D_idx] = C_row[i];
    }
}

// ─── CPU reference ────────────────────────────────────────────────────────────
static float cos_taylor_ref(float x) {
    float x2 = x * x, x4 = x2 * x2;
    return 1.0f - x2/2.0f + x4/24.0f - x4*x2/720.0f + x4*x4/40320.0f;
}

// ─── Host wrapper ─────────────────────────────────────────────────────────────
void run(float *D, float *exact, float *ref)
{
    double t0 = cpu_ms();
    for (int i = 0; i < M; i++) {
        D[i] = (2.0f * (float)M_PI * i) / (M - 1);
        exact[i] = cosf(D[i]);
        ref[i]   = cos_taylor_ref(D[i]);
    }
    double cpu_init_ms = cpu_ms() - t0;

    GPU_TIMER_DECL(alloc); GPU_TIMER_DECL(h2d);
    GPU_TIMER_DECL(kern);  GPU_TIMER_DECL(d2h); GPU_TIMER_DECL(gfree);
    GPU_TIMER_CREATE(alloc); GPU_TIMER_CREATE(h2d);
    GPU_TIMER_CREATE(kern);  GPU_TIMER_CREATE(d2h); GPU_TIMER_CREATE(gfree);
    float t_alloc, t_h2d, t_kern, t_d2h, t_gfree;

    float *d_D;
    GPU_TIMER_START(alloc);
    cudaMalloc(&d_D, (size_t)M * sizeof(float));
    GPU_TIMER_STOP(alloc); GPU_TIMER_MS(alloc, t_alloc);

    GPU_TIMER_START(h2d);
    cudaMemcpy(d_D, D, (size_t)M * sizeof(float), cudaMemcpyHostToDevice);
    GPU_TIMER_STOP(h2d); GPU_TIMER_MS(h2d, t_h2d);

    int  elems_per_block = WARPS_PER_BLOCK * WMMA_M * WMMA_N;  // 2048
    dim3 block(WARPS_PER_BLOCK * WARP_SIZE);                    // 256 threads
    dim3 grid((M + elems_per_block - 1) / elems_per_block);

    printf("Grid : %d blocks × %d threads\n", grid.x, block.x);
    printf("Tile : %d elems/warp, %d elems/block\n",
           WMMA_M * WMMA_N, elems_per_block);

    GPU_TIMER_START(kern);
    cosine_kernel<<<grid, block>>>(d_D);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    GPU_TIMER_STOP(kern); GPU_TIMER_MS(kern, t_kern);

    GPU_TIMER_START(d2h);
    cudaMemcpy(D, d_D, (size_t)M * sizeof(float), cudaMemcpyDeviceToHost);
    GPU_TIMER_STOP(d2h); GPU_TIMER_MS(d2h, t_d2h);

    GPU_TIMER_START(gfree);
    cudaFree(d_D);
    GPU_TIMER_STOP(gfree); GPU_TIMER_MS(gfree, t_gfree);

    GPU_TIMER_DESTROY(alloc); GPU_TIMER_DESTROY(h2d);
    GPU_TIMER_DESTROY(kern);  GPU_TIMER_DESTROY(d2h); GPU_TIMER_DESTROY(gfree);

    printf("\n─── Timing ─────────────────────────────────────────\n");
    printf("CPU  init:       %8.3f ms\n", cpu_init_ms);
    printf("GPU  malloc:     %8.3f ms\n", t_alloc);
    printf("GPU  H2D:        %8.3f ms\n", t_h2d);
    printf("GPU  kernel:     %8.3f ms\n", t_kern);
    printf("GPU  D2H:        %8.3f ms\n", t_d2h);
    printf("GPU  free:       %8.3f ms\n", t_gfree);
    printf("GPU  total:      %8.3f ms\n",
           t_alloc + t_h2d + t_kern + t_d2h + t_gfree);
    printf("────────────────────────────────────────────────────\n\n");
}

int main()
{
    float *D = (float*)malloc((size_t)M * sizeof(float));
    float *exact = (float*)malloc((size_t)M * sizeof(float));
    float *ref = (float*)malloc((size_t)M * sizeof(float));
    if (!D) { fprintf(stderr, "malloc failed\n"); return 1; }
    run(D, exact, ref);

    float max_err = 0.0f;
    float error = 0.0f;
    float total_error = 0.0f;
    for (int i = 0; i < M; i++) {
        error = fabsf(D[i] - exact[i]);
        total_error += error;
        if (error > max_err) max_err = error;
        if (error > 1e-3f) {
            printf("ERROR: D[%d] = %f, ref[%d] = %f\n", i, D[i], i, ref[i]);
            break;
        }
    }
    printf("Max error: %.2e\n", max_err);
    printf("Average error: %.2e\n", total_error / M);

    printf("═══════════════════════════════════════════════════════════\n");
    printf("CORRECTNESS CHECK\n");
    printf("═══════════════════════════════════════════════════════════\n");
    for (int i = M/6; i < M/6+10 && i < M; i++)
        printf("  D[%3d] = %+.10f,  CPU[%3d] = %+.10f,  Taylor[%3d] = %+.10f, err: %.2e\n", i, D[i], i, ref[i], i, exact[i], fabsf(D[i] - ref[i]));
    printf("═══════════════════════════════════════════════════════════\n");

    free(D);
    free(exact);
    free(ref);
    return 0;
}
