#include <cuda.h>
#include <sys/time.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cuda_fp16.h>
#include <mma.h>

#define WARP_SIZE 32
#define NUM_COLS_B 5
#define func_cols 5
#define NUM_TERMS 3
#define WMMA_M 16
#define Warps_in_row 8
#define N 16384
#define STRESS_TEST 2000


#define PI 3.14159265359

using namespace nvcuda;

typedef struct complex{
    double real;
    double imaginary;
} complex;

// Trigonometric functions are 2*pi periodic.  Keeping their argument in this
// interval prevents the x^4/x^6 Taylor terms from overflowing FP16 before the
// WMMA operation.
static inline float wrapToPi(double angle) {
    return (float)remainder(angle, 2.0 * PI);
}

void initExpList(float* expo_list) {
    int half_N = N / 2;
    for (int k = 0; k < half_N; ++k) {
        for (int m = 0; m < half_N; ++m) {
            double angle = -4.0 * PI * (double)m * (double)k / N;
            size_t index = (size_t)k * (size_t)half_N + (size_t)m;
            expo_list[index] = wrapToPi(angle);
            // expo_list[k * half_N + m] = 0;
        }
    }
}

void initTwiddleList(float* twiddle_list) {
    int half_N = N / 2;
    for (int k = 0; k < half_N; ++k) {
        double angle = -2.0 * PI * (double)k / N;
        twiddle_list[k] = wrapToPi(angle);
        // twiddle_list[k] = 0;
    }
}

__global__ void calcTranscKernel(float *d_exp_list, float *d_cos_list, float *d_sin_list) {
    size_t tile_x = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    size_t exp_col = tile_x * WMMA_M * NUM_COLS_B;
    int WARP_ID = threadIdx.x / WARP_SIZE;
	size_t total_elements = (size_t)(N / 2) * (size_t)(N / 2);

    __shared__ float shared_tile_C[Warps_in_row * WMMA_M * WMMA_M];
    __shared__ __half shared_tile_A[Warps_in_row * WMMA_M * WMMA_M];
    __shared__ __half shared_tile_B[WMMA_M * WMMA_M];

    for (int i = threadIdx.x;
         i < Warps_in_row * WMMA_M * WMMA_M; i += blockDim.x) {
        shared_tile_A[i] = __float2half(0.0f);
    }
    for (int i = threadIdx.x; i < WMMA_M * WMMA_M; i += blockDim.x) {
        shared_tile_B[i] = __float2half(0.0f);
    }
    __syncthreads();

    if (exp_col < total_elements) {
        if (threadIdx.x % WARP_SIZE < WMMA_M) {
            int row = threadIdx.x % WARP_SIZE;

            for (int i = 0; i < func_cols; i++) {
                size_t exp_indx = exp_col + row * func_cols + i;
                int A_index = WARP_ID * WMMA_M * WMMA_M + row * WMMA_M + i * NUM_TERMS;
                if (exp_indx < total_elements) {
                    float x = d_exp_list[exp_indx];
                    float x_squared = x * x;
                    shared_tile_A[A_index] = __float2half(x_squared);
                    shared_tile_A[A_index + 1] = __float2half(x_squared * x_squared);
                    shared_tile_A[A_index + 2] = __float2half(x_squared * x_squared * x_squared);
                }
                if (WARP_ID == 0) {
                    shared_tile_B[i * NUM_TERMS * WMMA_M + i] = __float2half(-1.0f / 2.0f);
                    shared_tile_B[i * NUM_TERMS * WMMA_M + 1 * WMMA_M + i] = __float2half(1.0f / 24.0f);
                    shared_tile_B[i * NUM_TERMS * WMMA_M + 2 * WMMA_M + i] = __float2half(-1.0f / 720.0f);
                    shared_tile_B[i * NUM_TERMS * WMMA_M + func_cols + i] = __float2half(-1.0f / 6.0f);
                    shared_tile_B[i * NUM_TERMS * WMMA_M + 1 * WMMA_M + func_cols + i] = __float2half(1.0f / 120.0f);
                    shared_tile_B[i * NUM_TERMS * WMMA_M + 2 * WMMA_M + func_cols + i] = __float2half(-1.0f / 5040.0f);
                }
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

    for (int j = 0; j < STRESS_TEST; j++) 
    {
        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
    }
    
    wmma::store_matrix_sync(shared_tile_C + WARP_ID * WMMA_M * WMMA_M, C_frag, WMMA_M, wmma::mem_row_major);
    
    if (threadIdx.x % WARP_SIZE < WMMA_M) {
        int row = threadIdx.x % WARP_SIZE;
        for (int i = 0; i < func_cols; i++) {
            size_t exp_indx = exp_col + row * func_cols + i;
            int C_index = WARP_ID * WMMA_M * WMMA_M + row * WMMA_M;
            if (exp_indx < total_elements) {
                d_sin_list[exp_indx] = shared_tile_C[C_index + func_cols + i] * d_exp_list[exp_indx];
                d_cos_list[exp_indx] = shared_tile_C[C_index + i];
            }
        }
    }
}

__global__ void calcTwidTranscKernel(float *d_twiddle_list, float *d_twiddle_cos_list, float *d_twiddle_sin_list) {
    int tile_x = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int twiddle_col = tile_x * WMMA_M * NUM_COLS_B;
    int WARP_ID = threadIdx.x / WARP_SIZE;
	int total_elements = (N / 2);

    __shared__ float shared_tile_C[Warps_in_row * WMMA_M * WMMA_M];
    __shared__ __half shared_tile_A[Warps_in_row * WMMA_M * WMMA_M];
    __shared__ __half shared_tile_B[WMMA_M * WMMA_M];

    for (int i = threadIdx.x;
         i < Warps_in_row * WMMA_M * WMMA_M; i += blockDim.x) {
        shared_tile_A[i] = __float2half(0.0f);
    }
    for (int i = threadIdx.x; i < WMMA_M * WMMA_M; i += blockDim.x) {
        shared_tile_B[i] = __float2half(0.0f);
    }
    __syncthreads();

    if (twiddle_col < total_elements) {
        if (threadIdx.x % WARP_SIZE < WMMA_M) {
            int row = threadIdx.x % WARP_SIZE;

            for (int i = 0; i < func_cols; i++) {
                int twiddle_indx = twiddle_col + row * func_cols + i;
                int A_index = WARP_ID * WMMA_M * WMMA_M + row * WMMA_M + i * NUM_TERMS;
                if (twiddle_indx < total_elements) {
                    float x = d_twiddle_list[twiddle_indx];
                    float x_squared = x * x;
                    shared_tile_A[A_index] = __float2half(x_squared);
                    shared_tile_A[A_index + 1] = __float2half(x_squared * x_squared);
                    shared_tile_A[A_index + 2] = __float2half(x_squared * x_squared * x_squared);
                }
                if (WARP_ID == 0) {
                    shared_tile_B[i * NUM_TERMS * WMMA_M + i] = __float2half(-1.0f / 2.0f);
                    shared_tile_B[i * NUM_TERMS * WMMA_M + 1 * WMMA_M + i] = __float2half(1.0f / 24.0f);
                    shared_tile_B[i * NUM_TERMS * WMMA_M + 2 * WMMA_M + i] = __float2half(-1.0f / 720.0f);
                    shared_tile_B[i * NUM_TERMS * WMMA_M + func_cols + i] = __float2half(-1.0f / 6.0f);
                    shared_tile_B[i * NUM_TERMS * WMMA_M + 1 * WMMA_M + func_cols + i] = __float2half(1.0f / 120.0f);
                    shared_tile_B[i * NUM_TERMS * WMMA_M + 2 * WMMA_M + func_cols + i] = __float2half(-1.0f / 5040.0f);
                }
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

    wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
    
    wmma::store_matrix_sync(shared_tile_C + WARP_ID * WMMA_M * WMMA_M, C_frag, WMMA_M, wmma::mem_row_major);
    
    if (threadIdx.x % WARP_SIZE < WMMA_M) {
        int row = threadIdx.x % WARP_SIZE;
        for (int i = 0; i < func_cols; i++) {
            int twiddle_indx = twiddle_col + row * func_cols + i;
            int C_index = WARP_ID * WMMA_M * WMMA_M + row * WMMA_M;
            if (twiddle_indx < total_elements) {
                d_twiddle_sin_list[twiddle_indx] = shared_tile_C[C_index + func_cols + i] * d_twiddle_list[twiddle_indx];
                d_twiddle_cos_list[twiddle_indx] = shared_tile_C[C_index + i];
            }
        }
    }
}

__device__
complex multiplyComplex(complex a, complex b)
{
    complex result;
    result.real = a.real*b.real - a.imaginary*b.imaginary;
    result.imaginary = a.real*b.imaginary + b.real*a.imaginary;

    return result;
}

__device__
complex addComplex(complex a, complex b)
{
    complex result = {a.real+b.real, a.imaginary+b.imaginary};
    return result;
}

__device__
complex subtractComplex(complex a, complex b)
{
    complex result = {a.real-b.real, a.imaginary-b.imaginary};
    return result;
}

__global__
void compute_Even_Odd(complex* xInput_d, float *d_cos_list, float *d_sin_list, complex *even_d, complex* odd_d)
{	
	size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	size_t m = idx % (N / 2);
	
	complex e_part;    

    //compute part with e^(i * exponent)
    e_part.real = d_cos_list[idx];
    e_part.imaginary = d_sin_list[idx];
        
    //save even part
	even_d[idx] = multiplyComplex(xInput_d[2 * m], e_part);
	
    //save odd part
    odd_d[idx] = multiplyComplex(xInput_d[(2 * m) + 1], e_part);
}

__device__
void compute_twiddle(float *d_twiddle_cos_list, float *d_twiddle_sin_list, complex* twiddle, int k)
{
	twiddle->real = d_twiddle_cos_list[k];
	twiddle->imaginary = d_twiddle_sin_list[k];
}

__global__
void sum_Regression(float *d_twiddle_cos_list, float *d_twiddle_sin_list, complex* even_d, complex* odd_d, complex* result_d)
{	
	int n = N / 2;
	size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	size_t k = idx / (n / 2); //we only need half the number of threads for regression
	size_t m = idx % (n / 2);
	size_t z = k * (size_t)n + m;
	
	while(n > 1)
	{
		__syncthreads();		
		if(m < n / 2)
		{
			even_d[z] = addComplex(even_d[z], even_d[z + n / 2]);
			odd_d[z] = addComplex(odd_d[z], odd_d[z + n / 2]);
		}		
		n = n / 2;
	}
	if(m == 0)
	{
		compute_twiddle(d_twiddle_cos_list, d_twiddle_sin_list, result_d + k, k);
		//calculate twiddle * O_k
		odd_d[z] = multiplyComplex(odd_d[z], result_d[k]);
		
		//compute X_k and X_k + N/2
		result_d[k] = addComplex(even_d[z], odd_d[z]);
		result_d[k + N / 2] = subtractComplex(even_d[z], odd_d[z]);
	}
}

int main()
{	
	complex* xInput, *result;
	
	complex* xInput_d, *result_d;
	complex *even_d, *odd_d;

	//allocate memory
	xInput = (complex *)malloc(N * sizeof(complex));
	result = (complex *)malloc(N * sizeof(complex));
	
	cudaMalloc((void**)&xInput_d, N * sizeof(complex));
	cudaMalloc((void**)&result_d, N * sizeof(complex));
	
	//set defined values from 0 to 7 for xInput
	xInput[0].real = 3.6;
	xInput[0].imaginary = 2.6;
	xInput[1].real = 2.9;
	xInput[1].imaginary = 6.3;
	xInput[2].real = 5.6;
	xInput[2].imaginary = 4;
	xInput[3].real = 4.8;
	xInput[3].imaginary = 9.1;
	xInput[4].real = 3.3;
	xInput[4].imaginary = 0.4;
	xInput[5].real = 5.9;
	xInput[5].imaginary = 4.8;
	xInput[6].real = 5;
	xInput[6].imaginary = 2.6;
	xInput[7].real = 4.3;
	xInput[7].imaginary = 4.1;
	
	//set defined values for x[8] to x[N-1] as 0
	for (int i = 8; i < N; i++)
	{
		xInput[i].real = 0;
		xInput[i].imaginary = 0;
	}
	
	clock_t st = clock();
	cudaMemcpy(xInput_d, xInput, sizeof(complex) * N, cudaMemcpyHostToDevice);
	
	size_t m_elements = (size_t)(N / 2) * (size_t)(N / 2);
	unsigned int m_blocks, m_threads;
	// k goes 0 to N - 1 (but same values used 2ice),
							//m goes 0 to N/2-1 for each k value
	
	cudaMalloc((void**)&even_d, sizeof(complex) * m_elements);
	cudaMalloc((void**)&odd_d, sizeof(complex) * m_elements);
	 
	m_blocks = (unsigned int)(m_elements / 1024); //number of blocks we need
	
	if(m_blocks == 0){
		m_blocks = 1;
		m_threads = (unsigned int)m_elements;
	}
	else{
		m_threads = 1024;
	}
	
	//Use half the number of blocks to compute results using regression
	dim3 dimSumGrid(m_blocks / 2, 1);
	dim3 dimSumBlock(m_threads, 1);
	if(m_blocks == 1)
	{
		dimSumGrid = dim3(m_blocks, 1);
		dimSumBlock = dim3(m_threads / 2, 1);
	}
		
	cudaEvent_t start, stop;
	float elapsedTime;
	size_t total_elements = (size_t)(N / 2) * (size_t)(N / 2);
    size_t list_size = total_elements * sizeof(float);
    float *exp_list = (float*)malloc(list_size);
	int twiddle_elements = (N / 2);
	size_t twiddle_size = (size_t)twiddle_elements * sizeof(float);
    float *twiddle_list = (float*)malloc(twiddle_size);
    float *d_exp_list, *d_cos_list, *d_sin_list, *d_twiddle_list, *d_twiddle_sin_list, *d_twiddle_cos_list;

    cudaMalloc((float**)&d_exp_list, list_size);
    cudaMalloc((float**)&d_cos_list, list_size);
    cudaMalloc((float**)&d_sin_list, list_size);
    cudaMalloc((float**)&d_twiddle_list, twiddle_size);
    cudaMalloc((float**)&d_twiddle_sin_list, twiddle_size);
    cudaMalloc((float**)&d_twiddle_cos_list, twiddle_size);

    dim3 block(Warps_in_row * WARP_SIZE);
    dim3 grid((unsigned int)((total_elements + (NUM_COLS_B * WMMA_M * block.x / WARP_SIZE) - 1) / (NUM_COLS_B * WMMA_M * block.x / WARP_SIZE)));
    
    initExpList(exp_list);
    cudaMemcpy(d_exp_list, exp_list, list_size, cudaMemcpyHostToDevice);

    dim3 twiddle_block(Warps_in_row * WARP_SIZE);
    dim3 twiddle_grid((twiddle_elements + (NUM_COLS_B * WMMA_M * block.x / WARP_SIZE) - 1) / (NUM_COLS_B * WMMA_M * block.x / WARP_SIZE));
    
    initTwiddleList(twiddle_list);
    cudaMemcpy(d_twiddle_list, twiddle_list, twiddle_size, cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    cudaEventRecord(start, 0);
    
    calcTwidTranscKernel<<<twiddle_grid, twiddle_block>>>(d_twiddle_list, d_twiddle_cos_list, d_twiddle_sin_list);
    cudaDeviceSynchronize();

    calcTranscKernel<<<grid, block>>>(d_exp_list, d_cos_list, d_sin_list);
    cudaDeviceSynchronize();

	compute_Even_Odd<<<m_blocks, m_threads>>>(xInput_d, d_cos_list, d_sin_list, even_d, odd_d);
    cudaDeviceSynchronize();

	sum_Regression<<<dimSumGrid, dimSumBlock>>>(d_twiddle_cos_list, d_twiddle_sin_list, even_d, odd_d, result_d);
    cudaDeviceSynchronize();

	clock_t end = clock();
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	
	printf("Cuda time : %f milli secs\n", elapsedTime);	
	
	cudaMemcpy(result, result_d, sizeof(complex) * N, cudaMemcpyDeviceToHost);
	
	cudaFree(xInput_d);
	cudaFree(result_d);
	cudaFree(even_d);
	cudaFree(odd_d);
	cudaFree(d_exp_list);
	cudaFree(d_cos_list);
	cudaFree(d_sin_list);
	cudaFree(d_twiddle_list);
	cudaFree(d_twiddle_sin_list);
	cudaFree(d_twiddle_cos_list);

	printf("PARALLEL VERSION\n\n");
	printf("TOTAL PROCESSED SAMPLES: %d\n", N);
	
	for(int i = 0; i < 8; i++)
	{
		printf("==================\n");
		printf("XR[%d]: %f\n", i, result[i].real);
		printf("XI[%d]: %f\n", i, result[i].imaginary);
	}
	
	printf("==================\n");
	
	//printf("\nC clock Time: %f\n", ((double)(end - st)) / CLOCKS_PER_SEC);
	
	return 0;
	
}
