// Utilities and system includes
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

// CUDA runtime and CUBLAS functions
#include <helper_string.h> 
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include <omp.h>

#define CHECK_CUDA_ERR(info_str) do {cudaError_t err = cudaGetLastError(); if (err != cudaSuccess) {printf("CUDA last Error: %s\n", cudaGetErrorString(err)); printf(info_str); assert(1 == 0);}} while (0);

#include "testKernel.cuh"

int main(int argc, char **argv)
{
	// Use non 2^p, different M, N, K default values to test the sanity
	int M = 8191, N = 8190, K = 8118, kernel = 1;
	if (argc >= 2) kernel = atoi(argv[1]);
	if (argc == 5)
	{
		M = atoi(argv[2]);
		N = atoi(argv[3]);
		K = atoi(argv[4]);
	}
	printf("Test kernel %d (0 for CUBLAS)\n", kernel);
	printf("Test size: M = %d, N = %d, K = %d\n", M, N, K);
	
	float alpha = 1.0, beta = 0.5;
	float *d_A, *d_B, *d_C, *d_refC, *h_A, *h_B, *h_C, *h_refC;
	int A_mem_size = sizeof(float) * M * K;
	int B_mem_size = sizeof(float) * K * N;
	int C_mem_size = sizeof(float) * M * N;
	
	h_A = (float*) malloc(A_mem_size);
	h_B = (float*) malloc(B_mem_size);
	h_C = (float*) malloc(C_mem_size);
	h_refC = (float*) malloc(C_mem_size);
	
	for (int i = 0; i < M * K; i++) h_A[i] = (float) (i / K + 1);
	for (int i = 0; i < K * N; i++) h_B[i] = (float) (i % N + 1);
	for (int i = 0; i < M * N; i++) { h_refC[i] = h_C[i] = (float) (i % 9 + 1); }
	
	cudaMalloc((void **) &d_A, A_mem_size);
	cudaMalloc((void **) &d_B, B_mem_size);
	cudaMalloc((void **) &d_C, C_mem_size);
	cudaMalloc((void **) &d_refC, C_mem_size);
	cudaMemcpy(d_A, h_A, A_mem_size, cudaMemcpyHostToDevice); 
	cudaMemcpy(d_B, h_B, B_mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C, C_mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_refC, h_C, C_mem_size, cudaMemcpyHostToDevice);
	
	// Get reference result
	printf("Generating reference result...");
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSgemm(
		handle, CUBLAS_OP_N, CUBLAS_OP_N, 
		N, M, K, 
		&alpha, d_B, N, d_A, K,
		&beta,  d_refC, N
	);
	cudaMemcpy(h_refC, d_refC, C_mem_size, cudaMemcpyDeviceToHost);
	printf(" Done.\n");
	
	// Check result sanity
	printf("Checking sanity of selected kernels on CUDA... ");
	testKernelSingleRun(kernel, M, N, K, alpha, beta, d_A, d_B, d_C);
	cudaMemcpy(h_C, d_C, C_mem_size, cudaMemcpyDeviceToHost);
	int cnt = 0;
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			float diff = fabs(h_C[i * N + j] - h_refC[i * N + j]);
			if (diff / fabs(h_refC[i * N + j]) > 0.00001)
			{
				cnt++;
				printf("Error %3d: Row %d Col %d, got %f, should be %f\n", cnt, i, j, h_C[i * N + j], h_refC[i * N + j]);
				break;
			}
		}
		if (cnt >= 1) break;
	}
	
	// Result is correct, test the performance
	if (cnt == 0) 
	{
		printf("Result is correct.\n");
		printf("Evaluating selected kernel performance... \n");
		testKernelPerformance(kernel, M, N, K, alpha, beta, d_A, d_B, d_C);
	} 
	
	return 0;
}
