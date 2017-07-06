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

#ifndef min
#define min(a,b) ((a < b) ? a : b)
#endif

#ifndef max
#define max(a,b) ((a > b) ? a : b)
#endif

#define block_size 32

#include "my_sgemm_kernels.cuh"  // my sgemm kernels
#include "my_sgemm_cm_kernels.cuh"  // my sgemm kernels
int target_kernel = 4;

typedef struct _matrixSize
{
	unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
} sMatrixSize;

void randomInit(float *data, int size)
{
	for (int i = 0; i < size; ++i)
		data[i] = rand() / (float)RAND_MAX;
}

void printDiff(float *data1, float *data2, int width, int height, int iListLength, float fListTol)
{
	printf("Listing first %d Differences > %.6f...\n", iListLength, fListTol);
	int i,j,k;
	int error_count=0;

	for (j = 0; j < height; j++)
	{
		if (error_count < iListLength)
		{
			printf("\n Row %d:\n", j);
		}

		for (i = 0; i < width; i++)
		{
			k = j * width + i;
			float fDiff = fabs(data1[k] - data2[k]);

			if (fDiff > fListTol)
			{
				if (error_count < iListLength)
				{
					printf("	Loc(%d,%d)\tRef=%.5f\tRes=%.5f\tDiff=%.6f\n", i, j, data1[k], data2[k], fDiff);
				}

				error_count++;
			}
		}
	}

	printf(" \n  Total Errors = %d\n", error_count);
}

void initializeCUDA(int argc, char **argv, int &devID, sMatrixSize &matrix_size)
{
	// By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
	cudaError_t error;
	devID = 0;
	
	unsigned int SizeMultiple = 2;
	unsigned int base_size = 512;

	if (checkCmdLineFlag(argc, (const char **)argv, "device"))
	{
		devID = getCmdLineArgumentInt(argc, (const char **)argv, "device");
		error = cudaSetDevice(devID);

		if (error != cudaSuccess)
		{
			printf("cudaSetDevice returned error code %d, line(%d)\n", error, __LINE__);
			exit(EXIT_FAILURE);
		}
	}

	// get number of SMs on this GPU
	error = cudaGetDevice(&devID);

	if (error != cudaSuccess)
	{
		printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
		exit(EXIT_FAILURE);
	}


	if (checkCmdLineFlag(argc, (const char **)argv, "sizemult"))
	{
		SizeMultiple = getCmdLineArgumentInt(argc, (const char **)argv, "sizemult");
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "kernel"))
	{
		target_kernel = getCmdLineArgumentInt(argc, (const char **)argv, "kernel");
	}
	
	if (checkCmdLineFlag(argc, (const char **)argv, "bs"))
	{
		base_size = getCmdLineArgumentInt(argc, (const char **)argv, "bs");
	}
	
	cudaDeviceProp deviceProp;

	error = cudaGetDeviceProperties(&deviceProp, devID);

	if (error != cudaSuccess)
	{
		printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
		exit(EXIT_FAILURE);
	}

	printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);

	matrix_size.uiWA = base_size * SizeMultiple;
	matrix_size.uiHA = base_size * SizeMultiple;
	matrix_size.uiWB = base_size * SizeMultiple;
	matrix_size.uiHB = base_size * SizeMultiple;
	matrix_size.uiWC = base_size * SizeMultiple;
	matrix_size.uiHC = base_size * SizeMultiple;

	printf("MatrixA(%u,%u), MatrixB(%u,%u), MatrixC(%u,%u)\n",
		   matrix_size.uiHA, matrix_size.uiWA,
		   matrix_size.uiHB, matrix_size.uiWB,
		   matrix_size.uiHC, matrix_size.uiWC);

	if( matrix_size.uiWA != matrix_size.uiHB ||
		matrix_size.uiHA != matrix_size.uiHC ||
		matrix_size.uiWB != matrix_size.uiWC)
	{
	   printf("ERROR: Matrix sizes do not match!\n");
	   exit(-1);
	}
}

////////////////////////////////////////////////////////////////////////////////
//! Run a test matrix multiply using CUBLAS as reference
////////////////////////////////////////////////////////////////////////////////
int matrixMultiply(int argc, char **argv, int devID, sMatrixSize &matrix_size)
{
	cudaDeviceProp deviceProp;

	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

	double flopsPerMatrixMul = 2.0 * (double)matrix_size.uiHC * (double)matrix_size.uiWC * (double)matrix_size.uiHB;
	
	// allocate host memory for matrices
	unsigned int size_A = matrix_size.uiWA * matrix_size.uiHA;
	unsigned int size_B = matrix_size.uiWB * matrix_size.uiHB;
	unsigned int size_C = matrix_size.uiWC * matrix_size.uiHC;
	unsigned int mem_size_A = sizeof(float) * size_A;
	unsigned int mem_size_B = sizeof(float) * size_B;
	unsigned int mem_size_C = sizeof(float) * size_C;
	float *h_A = (float *)malloc(mem_size_A);
	float *h_B = (float *)malloc(mem_size_B);
	float *h_C	    = (float *) malloc(mem_size_C);  // for my result
	float *h_CUBLAS = (float *) malloc(mem_size_C);  // for reference result

	// set seed for rand(), initialize host memory
	srand(12450);
	randomInit(h_A, size_A);
	randomInit(h_B, size_B);
	
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

	unsigned int M_pad = CEIL_DIV(matrix_size.uiHA, TSM) * TSM;
	unsigned int N_pad = CEIL_DIV(matrix_size.uiWB, TSN) * TSN;
	unsigned int K_pad = CEIL_DIV(matrix_size.uiWA, TSK) * TSK;
	unsigned int mem_size_PA = sizeof(float) * M_pad * K_pad;
	unsigned int mem_size_PB = sizeof(float) * K_pad * N_pad;
	unsigned int mem_size_PC = sizeof(float) * M_pad * N_pad;
	
	// allocate device memory
	float *d_A, *d_B, *d_C, *d_BT, *d_PA, *d_PBT, *d_PC;
	checkCudaErrors(cudaMalloc((void **) &d_A,   mem_size_A));
	checkCudaErrors(cudaMalloc((void **) &d_B,   mem_size_B));
	checkCudaErrors(cudaMalloc((void **) &d_C,   mem_size_C));
	checkCudaErrors(cudaMalloc((void **) &d_BT,  mem_size_B));
	checkCudaErrors(cudaMalloc((void **) &d_PA,  mem_size_PA));
	checkCudaErrors(cudaMalloc((void **) &d_PBT, mem_size_PB));
	checkCudaErrors(cudaMalloc((void **) &d_PC,  mem_size_PC));
	checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));
	
	int nIter = 30;
	
	cublasHandle_t handle;
	cudaEvent_t start, stop;
	
	/* ---------- CUBLAS version 2.0 test ---------- */
	#ifndef PROFILING
	{
		printf("Computing reference result using CUBLAS...");
		const float alpha = 1.0f;
		const float beta  = 0.0f;
		
		dim3 threads(block_size, block_size);
		dim3 grid(matrix_size.uiWC / threads.x, matrix_size.uiHC / threads.y);

		checkCudaErrors(cublasCreate(&handle));

		//Perform warmup operation with cublas
		checkCudaErrors(cublasSgemm(
			handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, 
			&alpha, d_B, matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWB
		));

		// Allocate CUDA events that we'll use for timing
		checkCudaErrors(cudaEventCreate(&start));
		checkCudaErrors(cudaEventCreate(&stop));

		// Record the start event
		checkCudaErrors(cudaEventRecord(start, NULL));

		for (int j = 0; j < nIter; j++)
		{
			//note cublas is column primary!
			//need to transpose the order
			if (target_kernel < 6)
			{
				checkCudaErrors(cublasSgemm(
					handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, 
					&alpha, d_B, matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWB
				));
			} else {
				checkCudaErrors(cublasSgemm(
					handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, 
					&alpha, d_A, matrix_size.uiHA, d_B, matrix_size.uiHB, &beta, d_C, matrix_size.uiHC
				));

			}
		}

		printf("done.\n");

		// Record the stop event
		checkCudaErrors(cudaEventRecord(stop, NULL));

		// Wait for the stop event to complete
		checkCudaErrors(cudaEventSynchronize(stop));

		float msecTotal = 0.0f;
		checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

		// Compute and print the performance
		float msecPerMatrixMul = msecTotal / nIter;
		double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
		printf("CUBLAS performance    = %.2f GFlop/s,\ttime= %.3f msec\n", gigaFlops, msecPerMatrixMul);

		// copy result from device to host
		checkCudaErrors(cudaMemcpy(h_CUBLAS, d_C, mem_size_C, cudaMemcpyDeviceToHost));

		// Destroy the handle
		checkCudaErrors(cublasDestroy(handle));
	}
	#endif
	/* ---------- CUBLAS version 2.0 test over ---------- */

	/* ---------- my implementation test ---------- */
	{
		printf("Selected kernel : %d, testing...", target_kernel);
		
		dim3 threads(block_size, block_size);
		int grid_y = matrix_size.uiHC / threads.y;
		int grid_x = matrix_size.uiWC / threads.x;
		if (grid_y * threads.y < matrix_size.uiHC) grid_y++;
		if (grid_x * threads.x < matrix_size.uiWC) grid_x++;
		
		dim3 grid(grid_x, grid_y);
		
		//Perform warmup operation with my kernel
		switch (target_kernel)
		{
			case 1: 
			{
				sgemm_1_naive<<<grid, threads>>> (
					d_A, matrix_size.uiWA,
					d_B, matrix_size.uiWB,
					d_C, matrix_size.uiWC,
					matrix_size.uiWA, matrix_size.uiHC, matrix_size.uiWC
				);
				break;
			}
			case 2: 
			{
				sgemm_2_tiling<<<grid, threads>>> (
					d_A, matrix_size.uiWA,
					d_B, matrix_size.uiWB,
					d_C, matrix_size.uiWC,
					matrix_size.uiWA, matrix_size.uiHC, matrix_size.uiWC
				);
				break;
			}
			case 3: 
			{
				sgemm_3_coalescing<<<grid, threads>>> (
					d_A, matrix_size.uiWA,
					d_B, matrix_size.uiWB,
					d_C, matrix_size.uiWC,
					matrix_size.uiWA, matrix_size.uiHC, matrix_size.uiWC
				);	
				break;
			}
			case 4: 
			{
				dim3 half_threads(block_size / 2, block_size / 2);
				sgemm_4_morework<<<grid, half_threads>>> (
					d_A, matrix_size.uiWA,
					d_B, matrix_size.uiWB,
					d_C, matrix_size.uiWC,
					matrix_size.uiWA, matrix_size.uiHC, matrix_size.uiWC
				);
				break;
			}
			case 5: 
			{
				dim3 quart_threads(block_size / 4, block_size / 4);
				sgemm_5_16xworks<<<grid, quart_threads>>> (
					d_A, matrix_size.uiWA,
					d_B, matrix_size.uiWB,
					d_C, matrix_size.uiWC,
					matrix_size.uiWA, matrix_size.uiHC, matrix_size.uiWC
				);
				break;
			}
			case 6: 
			{
				dim3 threads(block_size, block_size / 8);
				
				sgemm_6_cm<<<grid, threads>>> (
					d_A, matrix_size.uiHA,
					d_B, matrix_size.uiHB,
					d_C, matrix_size.uiHC,
					matrix_size.uiWA, matrix_size.uiHC, matrix_size.uiWC
				);
				break;
			}
			case 7: 
			{	
				unsigned int block_x = (matrix_size.uiHB + TRANSPOSEX - 1) / TRANSPOSEX;
				unsigned int block_y = (matrix_size.uiWB + TRANSPOSEY - 1) / TRANSPOSEY;
				dim3 blocksTRP(block_x, block_y);
				dim3 threadsTRP(TRANSPOSEX, TRANSPOSEY);
				transpose<<<blocksTRP, threadsTRP>>>(matrix_size.uiHA, matrix_size.uiWA, d_B, d_BT);
				
				dim3 blocks(matrix_size.uiHC / TSM, matrix_size.uiWC / TSN);
				dim3 threads(TSM / WPTM, TSN / WPTN);
				sgemm_7_reg2D<<<blocks, threads>>> (
					d_A, matrix_size.uiHA,
					d_BT, matrix_size.uiWB,
					d_C, matrix_size.uiHC,
					matrix_size.uiWA, matrix_size.uiHC, matrix_size.uiWC
				);
				break;
			}
			case 8: 
			{	
				unsigned int block_x = (matrix_size.uiHB + TRANSPOSEX - 1) / TRANSPOSEX;
				unsigned int block_y = (matrix_size.uiWB + TRANSPOSEY - 1) / TRANSPOSEY;
				dim3 blocksTRP(block_x, block_y);
				dim3 threadsTRP(TRANSPOSEX, TRANSPOSEY);
				transpose<<<blocksTRP, threadsTRP>>>(matrix_size.uiHB, matrix_size.uiWB, d_B, d_BT);
				
				dim3 blocks(matrix_size.uiHC / TSM, matrix_size.uiWC / TSN);
				dim3 threads(TSM / WPTM, TSN / WPTN);
				sgemm_8_reg2Dvec<<<blocks, threads>>> (
					(float4 *)d_A, matrix_size.uiHA,
					(float4 *)d_BT, matrix_size.uiWB,
					d_C, matrix_size.uiHC,
					matrix_size.uiWA, matrix_size.uiHC, matrix_size.uiWC
				);
				break;
			}
			case 9:
			{
				// transpose B
				unsigned int block_x = (matrix_size.uiHB + TRANSPOSEX - 1) / TRANSPOSEX;
				unsigned int block_y = (matrix_size.uiWB + TRANSPOSEY - 1) / TRANSPOSEY;
				dim3 blocksTRP(block_x, block_y);
				dim3 threadsTRP(TRANSPOSEX, TRANSPOSEY);
				transpose<<<blocksTRP, threadsTRP>>>(matrix_size.uiHB, matrix_size.uiWB, d_B, d_BT);
				
				// A pad 0
				dim3 blocksA(CEIL_DIV(M_pad,PADDINGX), CEIL_DIV(K_pad,PADDINGY));
				dim3 threadsA(PADDINGX, PADDINGY);
				padZeros<<<blocksA, threadsA>>>(matrix_size.uiHA, matrix_size.uiWA, M_pad, K_pad, d_A, d_PA);
				
				// B pad 0
				dim3 blocksB(CEIL_DIV(N_pad,PADDINGX), CEIL_DIV(K_pad,PADDINGY));
				dim3 threadsB(PADDINGX, PADDINGY);
				padZeros<<<blocksB, threadsB>>>(matrix_size.uiHB, matrix_size.uiWB, N_pad, K_pad, d_BT, d_PBT);
				
				// sgemm
				dim3 blocks(M_pad/TSM, N_pad/TSN);
				dim3 threads(TSM/WPTM, TSN/WPTN);
				sgemm_8_reg2Dvec<<<blocks, threads>>> (
					(float4 *)d_PA, M_pad,
					(float4 *)d_PBT, N_pad,
					d_PC, M_pad,
					K_pad, M_pad, N_pad
				);
				
				// C unpad 0
				dim3 blocksC(CEIL_DIV(matrix_size.uiHC,PADDINGX), CEIL_DIV(matrix_size.uiWC,PADDINGY));
				dim3 threadsC(PADDINGX, PADDINGY);
				removePadZeros<<<blocksC, threadsC>>>(M_pad, N_pad, matrix_size.uiHC, matrix_size.uiWC, d_PC, d_C);
			}
		}
		
		cudaError_t cudaError = cudaGetLastError();
		if (cudaError != cudaSuccess)
		{
			printf("  cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
		}

		// Allocate CUDA events that we'll use for timing
		checkCudaErrors(cudaEventCreate(&start));
		checkCudaErrors(cudaEventCreate(&stop));

		// Record the start event
		checkCudaErrors(cudaEventRecord(start, NULL));

		for (int j = 0; j < nIter; j++)
		{
			switch (target_kernel)
			{
				case 1: 
				{
					sgemm_1_naive<<<grid, threads>>> (
						d_A, matrix_size.uiWA,
						d_B, matrix_size.uiWB,
						d_C, matrix_size.uiWC,
						matrix_size.uiWA, matrix_size.uiHC, matrix_size.uiWC
					);
					break;
				}
				case 2: 
				{
					sgemm_2_tiling<<<grid, threads>>> (
						d_A, matrix_size.uiWA,
						d_B, matrix_size.uiWB,
						d_C, matrix_size.uiWC,
						matrix_size.uiWA, matrix_size.uiHC, matrix_size.uiWC
					);
					break;
				}
				case 3: 
				{
					sgemm_3_coalescing<<<grid, threads>>> (
						d_A, matrix_size.uiWA,
						d_B, matrix_size.uiWB,
						d_C, matrix_size.uiWC,
						matrix_size.uiWA, matrix_size.uiHC, matrix_size.uiWC
					);
					break;
				}
				case 4: 
				{
					dim3 half_threads(block_size / 2, block_size / 2);
					sgemm_4_morework<<<grid, half_threads>>> (
						d_A, matrix_size.uiWA,
						d_B, matrix_size.uiWB,
						d_C, matrix_size.uiWC,
						matrix_size.uiWA, matrix_size.uiHC, matrix_size.uiWC
					);
					break;
				}
				case 5: 
				{
					dim3 quart_threads(8, 8);
					sgemm_5_16xworks<<<grid, quart_threads>>> (
						d_A, matrix_size.uiWA,
						d_B, matrix_size.uiWB,
						d_C, matrix_size.uiWC,
						matrix_size.uiWA, matrix_size.uiHC, matrix_size.uiWC
					);
					break;
				}
				case 6: 
				{
					dim3 threads(block_size, block_size / 8);
					
					sgemm_6_cm<<<grid, threads>>> (
						d_A, matrix_size.uiHA,
						d_B, matrix_size.uiHB,
						d_C, matrix_size.uiHC,
						matrix_size.uiWA, matrix_size.uiHC, matrix_size.uiWC
					);
					/*
					sgemm_6_rm<<<grid, threads>>> (
						d_B, matrix_size.uiWB,
						d_A, matrix_size.uiWA,
						d_C, matrix_size.uiWC,
						matrix_size.uiWA, matrix_size.uiHC, matrix_size.uiWC
					);
					*/
					break;
				}
				case 7: 
				{
					unsigned int block_x = (matrix_size.uiHB + TRANSPOSEX - 1) / TRANSPOSEX;
					unsigned int block_y = (matrix_size.uiWB + TRANSPOSEY - 1) / TRANSPOSEY;
					dim3 blocksTRP(block_x, block_y);
					dim3 threadsTRP(TRANSPOSEX, TRANSPOSEY);
					transpose<<<blocksTRP, threadsTRP>>>(matrix_size.uiHB, matrix_size.uiWB, d_B, d_BT);
					
					dim3 blocks(matrix_size.uiHC / TSM, matrix_size.uiWC / TSN);
					dim3 threads(TSM / WPTM, TSN / WPTN);
					sgemm_7_reg2D<<<blocks, threads>>> (
						d_A, matrix_size.uiHA,
						d_BT, matrix_size.uiWB,
						d_C, matrix_size.uiHC,
						matrix_size.uiWA, matrix_size.uiHC, matrix_size.uiWC
					);
					break;
				}
				case 8: 
				{
					unsigned int block_x = (matrix_size.uiHB + TRANSPOSEX - 1) / TRANSPOSEX;
					unsigned int block_y = (matrix_size.uiWB + TRANSPOSEY - 1) / TRANSPOSEY;
					dim3 blocksTRP(block_x, block_y);
					dim3 threadsTRP(TRANSPOSEX, TRANSPOSEY);
					transpose<<<blocksTRP, threadsTRP>>>(matrix_size.uiHB, matrix_size.uiWB, d_B, d_BT);
					
					dim3 blocks(matrix_size.uiHC / TSM, matrix_size.uiWC / TSN);
					dim3 threads(TSM / WPTM, TSN / WPTN);
					sgemm_8_reg2Dvec<<<blocks, threads>>> (
						(float4 *)d_A, matrix_size.uiHA,
						(float4 *)d_BT, matrix_size.uiWB,
						d_C, matrix_size.uiHC,
						matrix_size.uiWA, matrix_size.uiHC, matrix_size.uiWC
					);
					break;
				}
				case 9:
				{
					// transpose B
					unsigned int block_x = (matrix_size.uiHB + TRANSPOSEX - 1) / TRANSPOSEX;
					unsigned int block_y = (matrix_size.uiWB + TRANSPOSEY - 1) / TRANSPOSEY;
					dim3 blocksTRP(block_x, block_y);
					dim3 threadsTRP(TRANSPOSEX, TRANSPOSEY);
					transpose<<<blocksTRP, threadsTRP>>>(matrix_size.uiHB, matrix_size.uiWB, d_B, d_BT);
					
					// A pad 0
					dim3 blocksA(CEIL_DIV(M_pad,PADDINGX), CEIL_DIV(K_pad,PADDINGY));
					dim3 threadsA(PADDINGX, PADDINGY);
					padZeros<<<blocksA, threadsA>>>(matrix_size.uiHA, matrix_size.uiWA, M_pad, K_pad, d_A, d_PA);
					
					// B pad 0
					dim3 blocksB(CEIL_DIV(N_pad,PADDINGX), CEIL_DIV(K_pad,PADDINGY));
					dim3 threadsB(PADDINGX, PADDINGY);
					padZeros<<<blocksB, threadsB>>>(matrix_size.uiHB, matrix_size.uiWB, N_pad, K_pad, d_BT, d_PBT);
					
					// sgemm
					dim3 blocks(M_pad/TSM, N_pad/TSN);
					dim3 threads(TSM/WPTM, TSN/WPTN);
					sgemm_8_reg2Dvec<<<blocks, threads>>> (
						(float4 *)d_PA, M_pad,
						(float4 *)d_PBT, N_pad,
						d_PC, M_pad,
						K_pad, M_pad, N_pad
					);
					
					// C unpad 0
					dim3 blocksC(CEIL_DIV(matrix_size.uiHC,PADDINGX), CEIL_DIV(matrix_size.uiWC,PADDINGY));
					dim3 threadsC(PADDINGX, PADDINGY);
					removePadZeros<<<blocksC, threadsC>>>(M_pad, N_pad, matrix_size.uiHC, matrix_size.uiWC, d_PC, d_C);
				}
			}
		}
		
		// Record the stop event
		checkCudaErrors(cudaEventRecord(stop, NULL));

		// Wait for the stop event to complete
		checkCudaErrors(cudaEventSynchronize(stop));

		float msecTotal = 0.0f;
		checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
		
		printf("done.\n");

		// Compute and print the performance
		float msecPerMatrixMul = msecTotal / nIter;
		double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
		printf("my kernel performance = %.2f GFlop/s,\ttime= %.3f msec\n", gigaFlops, msecPerMatrixMul);

		// copy result from device to host
		#ifndef PROFILING
		checkCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));
		
		bool res = sdkCompareL2fe(h_CUBLAS, h_C, size_C, 1.0e-6f);
		
		if (res != true)
		{
			printDiff(h_CUBLAS, h_C, matrix_size.uiWC, matrix_size.uiHC, 10, 1.0e-5f);
		}
		
		printf("Comparing CUBLAS SGEMM with my kernel results: %s\n", (true == res) ? "PASS" : "FAIL");
		#endif
	}
	/* ---------- my implementation test over ---------- */
	
	// clean up memory
	free(h_A);
	free(h_B);
	free(h_C);
	free(h_CUBLAS);
	checkCudaErrors(cudaFree(d_A));
	checkCudaErrors(cudaFree(d_B));
	checkCudaErrors(cudaFree(d_C));
	checkCudaErrors(cudaFree(d_BT));
	checkCudaErrors(cudaFree(d_PA));
	checkCudaErrors(cudaFree(d_PBT));
	checkCudaErrors(cudaFree(d_PC));

	return EXIT_SUCCESS;
}

int main(int argc, char **argv)
{
	int devID = 0;
	sMatrixSize matrix_size;

	initializeCUDA(argc, argv, devID, matrix_size);

	int matrix_result = matrixMultiply(argc, argv, devID, matrix_size);

	return matrix_result;
}
