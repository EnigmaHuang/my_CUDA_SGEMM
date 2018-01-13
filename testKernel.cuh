
#include "myCUDASgemm.cuh"

#define testKernelParam const unsigned int C_height, const unsigned int C_width, const unsigned int comm_dim, \
						const float alpha, const float beta, const float *d_A, const float *d_B, float *d_C

/* 
Input data format:
1. *d_A, *d_B, *d_C are pointers on device;
2. Given matrices are row major style;
3. A has C_height rows, comm_dim columns, leading dimension = comm_dim;
4. B has comm_dim rows, C_width  columns, leading dimension = C_width;
5. C has C_height rows, C_width  columns, leading dimension = C_width;
6. Need to returen C := alpha * A * B + beta * C, stored in *d_C

Some common regulation in my sgemm kernels and test routines:
1. For grids and blocks in the padding & sgemm & unpad kernels: 
	* `row` is related to `threadIdx.y` and blockIdx.y, 
	* `col` is related to `threadIdx.x` and blockIdx.x.
	* Notice: dim3(xdim, ydim, zdim)
2. Mat_memsize = sizeof(float) * Mat_num_rows * Mat_num_columns;
*/

void testKernel1(testKernelParam)
{
	dim3 kernel1_grid(CEIL_DIV(C_width, block_size), CEIL_DIV(C_height, block_size));
	dim3 kernel1_block(block_size, block_size);
	sgemm_1_naive<<<kernel1_grid, kernel1_block>>>(
		d_A, comm_dim,
		d_B, C_width,
		d_C, C_width,
		alpha, beta, 
		comm_dim, C_height, C_width
	);
}

void testKernel2(testKernelParam)
{
	dim3 kernel2_grid(CEIL_DIV(C_width, block_size), CEIL_DIV(C_height, block_size));
	dim3 kernel2_block(block_size, block_size);
	sgemm_2_tiling<<<kernel2_grid, kernel2_block>>>(
		d_A, comm_dim,
		d_B, C_width,
		d_C, C_width,
		alpha, beta, 
		comm_dim, C_height, C_width
	);
}

void testKernel3(testKernelParam)
{
	dim3 kernel3_grid(CEIL_DIV(C_width, block_size), CEIL_DIV(C_height, block_size));
	dim3 kernel3_block(block_size, block_size);
	sgemm_3_coalescing<<<kernel3_grid, kernel3_block>>>(
		d_A, comm_dim,
		d_B, C_width,
		d_C, C_width,
		alpha, beta,
		comm_dim, C_height, C_width
	);
}

void testKernel4(testKernelParam)
{
	dim3 kernel4_grid(CEIL_DIV(C_width, block_size), CEIL_DIV(C_height, block_size));
	dim3 kernel4_block(block_size / 2, block_size / 2);
	sgemm_4_morework<<<kernel4_grid, kernel4_block>>> (
		d_A, comm_dim,
		d_B, C_width,
		d_C, C_width,
		alpha, beta, 
		comm_dim, C_height, C_width
	);
}

void testKernel5(testKernelParam)
{
	// (1) Pad A & B with 0 in row major style
	unsigned int pad_C_height, pad_C_width, pad_comm_dim;
	pad_C_height = CEIL_DIV(C_height, tile_size) * tile_size;
	pad_C_width  = CEIL_DIV(C_width,  tile_size) * tile_size;
	pad_comm_dim = CEIL_DIV(comm_dim, tile_size) * tile_size;
	
	float *pad_A, *pad_B, *pad_C;
	unsigned int pad_A_memsize = sizeof(float) * pad_C_height * pad_comm_dim;
	unsigned int pad_B_memsize = sizeof(float) * pad_comm_dim * pad_C_width;
	unsigned int pad_C_memsize = sizeof(float) * pad_C_height * pad_C_width;
	cudaMalloc((void **) &pad_A, pad_A_memsize);
	cudaMalloc((void **) &pad_B, pad_B_memsize); 
	cudaMalloc((void **) &pad_C, pad_C_memsize); 

	dim3 pad_block(PADDINGX, PADDINGY);
	dim3 padA_grid(CEIL_DIV(pad_comm_dim, PADDINGX), CEIL_DIV(pad_C_height, PADDINGY));
	dim3 padB_grid(CEIL_DIV(pad_C_width,  PADDINGX), CEIL_DIV(pad_comm_dim, PADDINGY));
	dim3 padC_grid(CEIL_DIV(pad_C_width,  PADDINGX), CEIL_DIV(pad_C_height, PADDINGY));
	padZeros_rm<<<padA_grid, pad_block>>>(C_height, comm_dim, pad_C_height, pad_comm_dim, d_A, pad_A); 
	padZeros_rm<<<padB_grid, pad_block>>>(comm_dim, C_width,  pad_comm_dim, pad_C_width,  d_B, pad_B); 
	padZeros_rm<<<padC_grid, pad_block>>>(C_height, C_width,  pad_C_height, pad_C_width,  d_C, pad_C); 

	// (2) Perform C := alpha * C + A * B
	dim3 kernel5grid(CEIL_DIV(pad_C_width, block_size), CEIL_DIV(pad_C_height, block_size));
	dim3 kernel5block(block_size, block_size / 8);
	sgemm_5_morework8x<<<kernel5grid, kernel5block>>>(
		pad_A, pad_comm_dim,
		pad_B, pad_C_width,
		pad_C, pad_C_width,
		alpha, beta,
		pad_comm_dim, pad_C_height, pad_C_width
	);
	
	// (3) Remove padded 0
	removePadZeros_rm<<<padC_grid, pad_block>>>(pad_C_height, pad_C_width, C_height, C_width, pad_C, d_C); 

	cudaFree(pad_A); 
	cudaFree(pad_B); 
	cudaFree(pad_C);
}

void testKernel6(testKernelParam)
{
	// (1) Pad A & B with 0 in row major style
	unsigned int pad_C_height, pad_C_width, pad_comm_dim;
	pad_C_height = CEIL_DIV(C_height, TILE_SIZE) * TILE_SIZE;
	pad_C_width  = CEIL_DIV(C_width,  TILE_SIZE) * TILE_SIZE;
	pad_comm_dim = CEIL_DIV(comm_dim, TILE_SIZE) * TILE_SIZE;
	
	float *pad_A, *pad_B, *pad_C;
	unsigned int pad_A_memsize = sizeof(float) * pad_C_height * pad_comm_dim;
	unsigned int pad_B_memsize = sizeof(float) * pad_comm_dim * pad_C_width;
	unsigned int pad_C_memsize = sizeof(float) * pad_C_height * pad_C_width;
	cudaMalloc((void **) &pad_A, pad_A_memsize);
	cudaMalloc((void **) &pad_B, pad_B_memsize); 
	cudaMalloc((void **) &pad_C, pad_C_memsize); 

	dim3 pad_block(PADDINGX, PADDINGY);
	dim3 padA_grid(CEIL_DIV(pad_comm_dim, PADDINGX), CEIL_DIV(pad_C_height, PADDINGY));
	dim3 padB_grid(CEIL_DIV(pad_C_width,  PADDINGX), CEIL_DIV(pad_comm_dim, PADDINGY));
	dim3 padC_grid(CEIL_DIV(pad_C_width,  PADDINGX), CEIL_DIV(pad_C_height, PADDINGY));
	padZeros_rm<<<padA_grid, pad_block>>>(C_height, comm_dim, pad_C_height, pad_comm_dim, d_A, pad_A); 
	padZeros_rm<<<padB_grid, pad_block>>>(comm_dim, C_width,  pad_comm_dim, pad_C_width,  d_B, pad_B); 
	padZeros_rm<<<padC_grid, pad_block>>>(C_height, C_width,  pad_C_height, pad_C_width,  d_C, pad_C); 

	// (2) Perform C := alpha * C + A * B
	dim3 kernel6grid(CEIL_DIV(pad_C_width, TILE_SIZE), CEIL_DIV(pad_C_height, TILE_SIZE));
	dim3 kernel6block(TILE_SIZE / WPTN, TILE_SIZE / WPTM);
	sgemm_6_2Dreg<<<kernel6grid, kernel6block>>>(
		pad_A, pad_comm_dim,
		pad_B, pad_C_width,
		pad_C, pad_C_width,
		alpha, beta,
		pad_comm_dim, pad_C_height, pad_C_width
	);
	
	// (3) Remove padded 0
	removePadZeros_rm<<<padC_grid, pad_block>>>(pad_C_height, pad_C_width, C_height, C_width, pad_C, d_C); 

	cudaFree(pad_A); 
	cudaFree(pad_B); 
	cudaFree(pad_C);
}

void testKernel7(testKernelParam)
{
	// (1) Pad A & B with 0 in row major style
	unsigned int pad_C_height, pad_C_width, pad_comm_dim;
	pad_C_height = CEIL_DIV(C_height, TSM) * TSM;
	pad_C_width  = CEIL_DIV(C_width,  TSN) * TSN;
	pad_comm_dim = CEIL_DIV(comm_dim, TSK) * TSK;
	
	float *pad_A, *pad_B, *pad_C;
	unsigned int pad_A_memsize = sizeof(float) * pad_C_height * pad_comm_dim;
	unsigned int pad_B_memsize = sizeof(float) * pad_comm_dim * pad_C_width;
	unsigned int pad_C_memsize = sizeof(float) * pad_C_height * pad_C_width;
	cudaMalloc((void **) &pad_A, pad_A_memsize);
	cudaMalloc((void **) &pad_B, pad_B_memsize); 
	cudaMalloc((void **) &pad_C, pad_C_memsize); 

	dim3 pad_block(PADDINGX, PADDINGY);
	dim3 padA_grid(CEIL_DIV(pad_comm_dim, PADDINGX), CEIL_DIV(pad_C_height, PADDINGY));
	dim3 padB_grid(CEIL_DIV(pad_C_width,  PADDINGX), CEIL_DIV(pad_comm_dim, PADDINGY));
	dim3 padC_grid(CEIL_DIV(pad_C_width,  PADDINGX), CEIL_DIV(pad_C_height, PADDINGY));
	padZeros_rm<<<padA_grid, pad_block>>>(C_height, comm_dim, pad_C_height, pad_comm_dim, d_A, pad_A); 
	padZeros_rm<<<padB_grid, pad_block>>>(comm_dim, C_width,  pad_comm_dim, pad_C_width,  d_B, pad_B); 
	padZeros_rm<<<padC_grid, pad_block>>>(C_height, C_width,  pad_C_height, pad_C_width,  d_C, pad_C); 

	// (2) Perform C := alpha * C + A * B
	dim3 kernel7grid(CEIL_DIV(pad_C_width, TSN), CEIL_DIV(pad_C_height, TSM));
	dim3 kernel7block(RTS_N, RTS_M);
	sgemm_7_2Dreg_rect<<<kernel7grid, kernel7block>>>(
		pad_A, pad_comm_dim,
		pad_B, pad_C_width,
		pad_C, pad_C_width,
		alpha, beta,
		pad_comm_dim, pad_C_height, pad_C_width
	);
	cudaDeviceSynchronize();
	
	// (3) Remove padded 0
	removePadZeros_rm<<<padC_grid, pad_block>>>(pad_C_height, pad_C_width, C_height, C_width, pad_C, d_C); 

	cudaFree(pad_A); 
	cudaFree(pad_B); 
	cudaFree(pad_C);
}

void testKernel8(testKernelParam)
{
	// (1) Pad A & B with 0 in row major style
	unsigned int pad_C_height, pad_C_width, pad_comm_dim;
	pad_C_height = CEIL_DIV(C_height, TSM) * TSM;
	pad_C_width  = CEIL_DIV(C_width,  TSN) * TSN;
	pad_comm_dim = CEIL_DIV(comm_dim, TSK) * TSK;
	
	float *pad_A, *pad_B, *pad_C;
	unsigned int pad_A_memsize = sizeof(float) * pad_C_height * pad_comm_dim;
	unsigned int pad_B_memsize = sizeof(float) * pad_comm_dim * pad_C_width;
	unsigned int pad_C_memsize = sizeof(float) * pad_C_height * pad_C_width;
	cudaMalloc((void **) &pad_A, pad_A_memsize);
	cudaMalloc((void **) &pad_B, pad_B_memsize); 
	cudaMalloc((void **) &pad_C, pad_C_memsize); 

	dim3 pad_block(PADDINGX, PADDINGY);
	dim3 padA_grid(CEIL_DIV(pad_comm_dim, PADDINGX), CEIL_DIV(pad_C_height, PADDINGY));
	dim3 padB_grid(CEIL_DIV(pad_C_width,  PADDINGX), CEIL_DIV(pad_comm_dim, PADDINGY));
	dim3 padC_grid(CEIL_DIV(pad_C_width,  PADDINGX), CEIL_DIV(pad_C_height, PADDINGY));
	padZeros_rm<<<padA_grid, pad_block>>>(C_height, comm_dim, pad_C_height, pad_comm_dim, d_A, pad_A); 
	padZeros_rm<<<padB_grid, pad_block>>>(comm_dim, C_width,  pad_comm_dim, pad_C_width,  d_B, pad_B); 
	padZeros_rm<<<padC_grid, pad_block>>>(C_height, C_width,  pad_C_height, pad_C_width,  d_C, pad_C); 

	// (2) Perform C := alpha * C + A * B
	dim3 kernel8grid(CEIL_DIV(pad_C_width, TSN), CEIL_DIV(pad_C_height, TSM));
	dim3 kernel8block(RTS_N, RTS_M);
	double st = omp_get_wtime();
	sgemm_8_2Dreg_float4<<<kernel8grid, kernel8block>>>(
		(float4*) pad_A, pad_comm_dim,
		(float4*) pad_B, pad_C_width,
		pad_C, pad_C_width,
		alpha, beta,
		pad_comm_dim, pad_C_height, pad_C_width
	);
	cudaDeviceSynchronize();
	double ut = omp_get_wtime() - st;
	double GFlops = 2.0 * ((double) C_height * (double) C_width) * ((double) comm_dim + 1);
	GFlops /= 1000000000.0 * ut;
	//printf("Kernel 8 achieved performance = %.2lf GFlops, avg time = %.2lf ms\n", GFlops, ut * 1000.0);
	
	// (3) Remove padded 0
	removePadZeros_rm<<<padC_grid, pad_block>>>(pad_C_height, pad_C_width, C_height, C_width, pad_C, d_C); 

	cudaFree(pad_A); 
	cudaFree(pad_B); 
	cudaFree(pad_C);
}

#include <cublas_v2.h>

void testCUBLAS(testKernelParam)
{
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSgemm(
		handle, CUBLAS_OP_N, CUBLAS_OP_N, 
		C_width, C_height, comm_dim, 
		&alpha, d_B, C_width, d_A, comm_dim,
		&beta,  d_C, C_width
	);
	cudaDeviceSynchronize();
}

void testKernelSingleRun(
	const int kernel, const int C_height, const int C_width, const int comm_dim, 
	const float alpha, const float beta, const float *d_A, const float *d_B, float *d_C
)
{
	switch (kernel)
	{
		case 0: testCUBLAS (C_height, C_width, comm_dim, alpha, beta, d_A, d_B, d_C); break;
		case 1: testKernel1(C_height, C_width, comm_dim, alpha, beta, d_A, d_B, d_C); break;
		case 2: testKernel2(C_height, C_width, comm_dim, alpha, beta, d_A, d_B, d_C); break;
		case 3: testKernel3(C_height, C_width, comm_dim, alpha, beta, d_A, d_B, d_C); break;
		case 4: testKernel4(C_height, C_width, comm_dim, alpha, beta, d_A, d_B, d_C); break;
		case 5: testKernel5(C_height, C_width, comm_dim, alpha, beta, d_A, d_B, d_C); break;
		case 6: testKernel6(C_height, C_width, comm_dim, alpha, beta, d_A, d_B, d_C); break;
		case 7: testKernel7(C_height, C_width, comm_dim, alpha, beta, d_A, d_B, d_C); break;
		case 8: testKernel8(C_height, C_width, comm_dim, alpha, beta, d_A, d_B, d_C); break;
	}
	cudaDeviceSynchronize();
}


void testKernelPerformance(
	const int kernel, const int C_height, const int C_width, const int comm_dim, 
	const float alpha, const float beta, const float *d_A, const float *d_B, float *d_C
)
{	
	double GFlops = 2.0 * ((double) C_height * (double) C_width) * ((double) comm_dim + 1);
	GFlops *= 30.0;
	GFlops /= 1000000000.0;
	
	
	// Warm up
	testKernelSingleRun(kernel, C_height, C_width, comm_dim, alpha, beta, d_A, d_B, d_C);
	
	// Start testing
	double st, ut = 0;
	for (int i = 0; i < 30; i++)
	{
		st = omp_get_wtime();
		testKernelSingleRun(kernel, C_height, C_width, comm_dim, alpha, beta, d_A, d_B, d_C);
		ut += omp_get_wtime() - st;
	}
	GFlops /= ut;
	
	printf("Measured effective performance: %.2lf GFlops, avg time = %.2lf ms\n", GFlops, ut * 1000.0 / 30.0);
}
