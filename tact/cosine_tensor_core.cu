#include <stdio.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <time.h>
#include <sys/time.h>

#define USECPSEC 1000ULL
#define WARP_SIZE 32
#define NUM_COLS_B 5
#define NUM_TERMS 3
#define WMMA_M 16
#define M 100'000'000
#define Warps_in_row 8

using namespace nvcuda;

unsigned long long dtime_usec(unsigned long long start) {
    timeval tv;
    gettimeofday(&tv, 0);
    return ((tv.tv_sec * USECPSEC) + tv.tv_usec / 1000) - start;
}

void init_matrixD(float *mat, size_t row) {
    for (size_t i = 0; i < row; i++) {
        // mat[i] = (M_PI * (2 * i + 1));
        mat[i] = 5;
    }
}

__global__ void kernel(float *d_D) {
    int tile_x = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int D_col = tile_x * WMMA_M * NUM_COLS_B;
    int WARP_ID = threadIdx.x / WARP_SIZE;

    __shared__ float shared_tile_C[Warps_in_row * WMMA_M * WMMA_M];
    __shared__ __half shared_tile_A[Warps_in_row * WMMA_M * WMMA_M];
    __shared__ __half shared_tile_B[WMMA_M * WMMA_M];

    if (D_col < M) {
        if (threadIdx.x % WARP_SIZE < WMMA_M) {
            int row = threadIdx.x % WARP_SIZE;

            for (int i = 0; i < NUM_COLS_B; i++) {
                //init A
                int D_indx = D_col + row * NUM_COLS_B + i;
                int A_index = WARP_ID * WMMA_M * WMMA_M + row * WMMA_M + i * NUM_TERMS;
                if (D_indx < M) {
                    float x = d_D[D_indx];
                    float x_squared = x * x;
                    shared_tile_A[A_index] = __float2half(x_squared);
                    shared_tile_A[A_index + 1] = __float2half(x_squared * x_squared);
                    shared_tile_A[A_index + 2] = __float2half(x_squared * x_squared * x_squared);
                }
                // else {
                //     shared_tile_A[A_index] = __float2half(0.0f);
                //     shared_tile_A[A_index + 1] = __float2half(0.0f);
                //     shared_tile_A[A_index + 2] = __float2half(0.0f);
                // }
                // if (WMMA_M % NUM_TERMS != 0) {
                //     if (i == NUM_COLS_B - 1) {
                //         for (int j = 0; j < (WMMA_M % NUM_TERMS); j++) {
                //             shared_tile_A[A_index + 3 + j] = __float2half(0.0f);
                //         }
                //     }
                // }
                //init B
                if (WARP_ID == 0) {
                    // for (int b = 0; b < i; b++) {
                    //     shared_tile_B[b * NUM_TERMS * WMMA_M + b] = __float2half(0.0f);
                    //     shared_tile_B[b * NUM_TERMS * WMMA_M + 1 * WMMA_M + b] = __float2half(0.0f);
                    //     shared_tile_B[b * NUM_TERMS * WMMA_M + 2 * WMMA_M + b] = __float2half(0.0f);
                    // }
                    shared_tile_B[i * NUM_TERMS * WMMA_M + i] = __float2half(-1.0f / 2.0f);
                    shared_tile_B[i * NUM_TERMS * WMMA_M + 1 * WMMA_M + i] = __float2half(1.0f / 24.0f);
                    shared_tile_B[i * NUM_TERMS * WMMA_M + 2 * WMMA_M + i] = __float2half(-1.0f / 720.0f);
                    // for (int a = i * NUM_TERMS + 3; a < WMMA_M; a++) {
                    //     shared_tile_B[a * WMMA_M + a] = __float2half(0.0f);
                    // }
                }
            }
            //init C
            for (int cntr = 0; cntr < NUM_COLS_B; cntr++){
                shared_tile_C[WARP_ID * WMMA_M * WMMA_M + row * WMMA_M + cntr] = 0.0f;
            }
        }
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_M, WMMA_M, __half, wmma::row_major> A_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_M, WMMA_M, __half, wmma::row_major> B_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_M, WMMA_M, float> C_frag;
    wmma::fill_fragment(C_frag, 1.0f);

    wmma::load_matrix_sync(A_frag, shared_tile_A + WARP_ID * WMMA_M * WMMA_M, WMMA_M);
    wmma::load_matrix_sync(B_frag, shared_tile_B, WMMA_M);

    for (int j = 0; j < 2000; j++) 
    {
        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
    }

    wmma::store_matrix_sync(shared_tile_C + WARP_ID * WMMA_M * WMMA_M, C_frag, WMMA_M, wmma::mem_row_major);
    
    if (threadIdx.x % WARP_SIZE < WMMA_M) {
        int row = threadIdx.x % WARP_SIZE;
        for (int i = 0; i < NUM_COLS_B; i++) {
            int D_indx = D_col + row * NUM_COLS_B + i;
            int C_index = WARP_ID * WMMA_M * WMMA_M + row * WMMA_M + i;
            if (D_indx < M) {
                d_D[D_indx] = shared_tile_C[C_index];
            }
        }
    }
}

void matmul_inner_product(float *D) {
    unsigned long long tic[7];

    tic[0] = dtime_usec(0);

    size_t D_size = (size_t)M * sizeof(float);
    float *d_D;

    cudaMalloc((float**)&d_D, D_size);
    tic[1] = dtime_usec(0);
    cudaMemcpy(d_D, D, D_size, cudaMemcpyHostToDevice);
    tic[2] = dtime_usec(0);

    dim3 block(Warps_in_row * WARP_SIZE);
    dim3 grid((M + (NUM_COLS_B * WMMA_M * block.x / WARP_SIZE) - 1) / (NUM_COLS_B * WMMA_M * block.x / WARP_SIZE));

    tic[3] = dtime_usec(0);
    kernel<<<grid, block>>>(d_D);
    cudaDeviceSynchronize();
    tic[4] = dtime_usec(0);

    cudaMemcpy(D, d_D, D_size, cudaMemcpyDeviceToHost);
    tic[5] = dtime_usec(0);

    cudaFree(d_D);
    tic[6] = dtime_usec(0);

    printf("H2D copy time: %llu msec\n", tic[2] - tic[1]);
    printf("Kernel time: %llu msec\n", tic[4] - tic[3]);
    printf("D2H copy time: %llu msec\n", tic[5] - tic[4]);
    printf("GPU Tensor Core (inner product) total time: %llu msec\n", tic[6] - tic[0]);
}

int main() {
    size_t D_size = (size_t)M * sizeof(float);
    float *D = (float*)malloc(D_size);

    init_matrixD(D, M);
    matmul_inner_product(D);

    printf("%f\n", D[3]);
    
    free(D);

    return 0;
}
