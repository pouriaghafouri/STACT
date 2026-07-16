#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <time.h>

#define USECPSEC 1000ULL
#define M 90'000'000 * 10
#define K 16

using namespace nvcuda;

unsigned long long dtime_usec(unsigned long long start) {
    timeval tv;
    gettimeofday(&tv, 0);
    return ((tv.tv_sec * USECPSEC) + tv.tv_usec / 1000) - start;
}

void init_matrixD(float *mat, size_t col) {
    for (size_t i = 0; i < col; i++) {
        mat[i] = (M_PI * ((2 * i) + 1));
    }
}

__global__ void kernel(float *d_D, float *d_log_list) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M) {
        for (int j = 0; j < 2000; j++) {
            d_log_list[idx] = __logf(d_D[idx] + j);
            d_D[idx] = __expf(d_D[idx] + j);
        }
    }
}

void matmul_inner_product(float *D, float *log_list) {
    unsigned long long tic[7];

    tic[0] = dtime_usec(0);

    size_t D_size = (size_t)M * sizeof(float);
    float *d_D;
    float *d_log_list = (float*)malloc(D_size);

    cudaMalloc((float**)&d_D, D_size);
    cudaMalloc((float**)&d_log_list, D_size);
    
    tic[1] = dtime_usec(0);
    cudaMemcpy(d_D, D, D_size, cudaMemcpyHostToDevice);
    tic[2] = dtime_usec(0);

    dim3 block(64);
    dim3 grid((M + block.x - 1) / block.x);

    tic[3] = dtime_usec(0);
    kernel<<<grid, block>>>(d_D, d_log_list);
    cudaDeviceSynchronize();
    tic[4] = dtime_usec(0);

    cudaMemcpy(D, d_D, D_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(log_list, d_log_list, D_size, cudaMemcpyDeviceToHost);
    tic[5] = dtime_usec(0);

    cudaFree(d_D);
    cudaFree(d_log_list);
    tic[6] = dtime_usec(0);

    printf("H2D copy time: %llu msec\n", tic[2] - tic[1]);
    printf("Kernel time: %llu msec\n", tic[4] - tic[3]);
    printf("D2H copy time: %llu msec\n", tic[5] - tic[4]);
    printf("GPU SP (inner product) total time: %llu msec\n", tic[6] - tic[0]);
}

int main() {
    srand(time(0));

    size_t D_size = (size_t)M * sizeof(float);
    float *log_list = (float*)malloc(D_size);

    float *D = (float*)malloc(D_size);

    init_matrixD(D, M);
    matmul_inner_product(D, log_list);

    free(D);

    return 0;
}
