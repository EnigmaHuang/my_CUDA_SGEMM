#ifndef _MY_SGEMM_KERNELS_H_
#define _MY_SGEMM_KERNELS_H_

__global__ 
void sgemm_1_naive(
	float *A, const unsigned int lda, 
	float *B, const unsigned int ldb, 
	float *C, const unsigned int ldc, 
	const unsigned int common_dim,
	const unsigned int c_height, 
	const unsigned int c_width
)
{
	// Accumulate row i of A and column j of B
	unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;
	
	if (row >= c_height || col >= c_width) return;

	float accu = 0.0;
	for (unsigned int k = 0; k < common_dim; k++)
	{
		accu += A[row * lda + k] * B[k * ldb + col];
	}

	// Write the block sub-matrix to device memory;
	// each thread writes one element
	C[row * ldc + col] = accu;
}

__global__ 
void sgemm_2_tiling(
	float *A, const unsigned int lda, 
	float *B, const unsigned int ldb, 
	float *C, const unsigned int ldc, 
	const unsigned int common_dim,
	const unsigned int c_height, 
	const unsigned int c_width
)
{
	__shared__ float As[block_size][block_size];
	__shared__ float Bs[block_size][block_size];
	
	unsigned int A_topleft_begin = blockIdx.y * block_size * lda;
	unsigned int A_topleft_end   = A_topleft_begin + common_dim;
	unsigned int A_step_size	 = block_size;
	
	unsigned int B_topleft_begin = blockIdx.x * block_size;
	unsigned int B_topleft_end   = B_topleft_begin + common_dim * ldb;
	unsigned int B_step_size	 = block_size * ldb;
	
	unsigned int a_topleft, b_topleft;
	float accumulated_Cij = 0;
	
	bool load_y_in_A = (blockIdx.y * block_size + threadIdx.y) < c_height;
	bool load_x_in_B = (blockIdx.x * block_size + threadIdx.x) < c_width;
	
	for (a_topleft = A_topleft_begin, b_topleft = B_topleft_begin;
		 a_topleft < A_topleft_end;
		 a_topleft += A_step_size, b_topleft += B_step_size
		)
	{
		if (a_topleft + threadIdx.x < A_topleft_end && load_y_in_A)
		{
			As[threadIdx.y][threadIdx.x] = A[a_topleft + lda * threadIdx.y + threadIdx.x];  
		} else {
			As[threadIdx.y][threadIdx.x] = 0;
		}
		
		if (load_x_in_B && b_topleft + threadIdx.y * ldb < B_topleft_end)
		{
			Bs[threadIdx.x][threadIdx.y] = B[b_topleft + ldb * threadIdx.y + threadIdx.x];  // shared mem bank conflict
		} else {
			Bs[threadIdx.x][threadIdx.y] = 0;
		}
		__syncthreads();
		
		for (unsigned int k = 0; k < block_size; k++)
			accumulated_Cij += As[threadIdx.y][k] * Bs[threadIdx.x][k]; // shared mem bank conflict
		__syncthreads();
	}
	
	// Accumulate row i of A and column j of B
	unsigned int row = block_size * blockIdx.y + threadIdx.y;
	unsigned int col = block_size * blockIdx.x + threadIdx.x;
	
	if (row < c_height && col < c_width)
		C[row * ldc + col] = accumulated_Cij;
}

__global__ 
void sgemm_3_coalescing(
	float *A, const unsigned int lda, 
	float *B, const unsigned int ldb, 
	float *C, const unsigned int ldc, 
	const unsigned int common_dim,
	const unsigned int c_height, 
	const unsigned int c_width
)
{
	__shared__ float As[block_size][block_size];
	__shared__ float Bs[block_size][block_size];
	
	unsigned int A_topleft_begin = blockIdx.y * block_size * lda;
	unsigned int A_topleft_end   = A_topleft_begin + common_dim;
	unsigned int A_step_size	 = block_size;
	
	unsigned int B_topleft_begin = blockIdx.x * block_size;
	unsigned int B_topleft_end   = B_topleft_begin + common_dim * ldb;
	unsigned int B_step_size	 = block_size * ldb;
	
	unsigned int a_topleft, b_topleft;
	float accumulated_Cij = 0;
	
	bool load_y_in_A = (blockIdx.y * block_size + threadIdx.y) < c_height;
	bool load_x_in_B = (blockIdx.x * block_size + threadIdx.x) < c_width;
	
	for (a_topleft = A_topleft_begin, b_topleft = B_topleft_begin;
		 a_topleft < A_topleft_end;
		 a_topleft += A_step_size, b_topleft += B_step_size
		)
	{
		if (a_topleft + threadIdx.x < A_topleft_end && load_y_in_A)
		{
			As[threadIdx.y][threadIdx.x] = A[a_topleft + lda * threadIdx.y + threadIdx.x];
		} else {
			As[threadIdx.y][threadIdx.x] = 0;
		}
		
		if (load_x_in_B && b_topleft + threadIdx.y * ldb < B_topleft_end)
		{
			Bs[threadIdx.y][threadIdx.x] = B[b_topleft + ldb * threadIdx.y + threadIdx.x];
		} else {
			Bs[threadIdx.y][threadIdx.x] = 0;
		}
		__syncthreads();
		
		for (unsigned int k = 0; k < block_size; k++)
			accumulated_Cij += As[threadIdx.y][k] * Bs[k][threadIdx.x];
		__syncthreads();
	}
	
	// Accumulate row i of A and column j of B
	unsigned int row = block_size * blockIdx.y + threadIdx.y;
	unsigned int col = block_size * blockIdx.x + threadIdx.x;
	
	if (row < c_height && col < c_width)
		C[row * ldc + col] = accumulated_Cij;
}

__global__ 
void sgemm_4_morework(
	float *A, const unsigned int lda, 
	float *B, const unsigned int ldb, 
	float *C, const unsigned int ldc, 
	const unsigned int common_dim,
	const unsigned int c_height, 
	const unsigned int c_width
)
{
	__shared__ float As[block_size][block_size];
	__shared__ float Bs[block_size][block_size];
	
	unsigned int A_topleft_begin = blockIdx.y * block_size * lda;
	unsigned int A_topleft_end   = A_topleft_begin + common_dim;
	unsigned int A_step_size	 = block_size;
	
	unsigned int B_topleft_begin = blockIdx.x * block_size;
	unsigned int B_topleft_end   = B_topleft_begin + common_dim * ldb;
	unsigned int B_step_size	 = block_size * ldb;
	
	unsigned int a_topleft, b_topleft;
	float Cs[4] = {0, 0, 0, 0};
	
	unsigned int ty  = threadIdx.y;
	unsigned int tx  = threadIdx.x;
	unsigned int ty1 = ty + 16;
	unsigned int tx1 = tx + 16;
	
	bool load_y_in_A  = (block_size * blockIdx.y + ty  < c_height);
	bool load_x_in_B  = (block_size * blockIdx.x + tx  < c_width);
	bool load_y1_in_A = (block_size * blockIdx.y + ty1 < c_height);
	bool load_x1_in_B = (block_size * blockIdx.x + tx1 < c_width);
	
	for (a_topleft = A_topleft_begin, b_topleft = B_topleft_begin;
		 a_topleft < A_topleft_end;
		 a_topleft += A_step_size, b_topleft += B_step_size
		)
	{
		As[ty][tx]   = 0;
		As[ty][tx1]  = 0;
		As[ty1][tx]  = 0;
		As[ty1][tx1] = 0;
		
		Bs[ty][tx]   = 0;
		Bs[ty][tx1]  = 0;
		Bs[ty1][tx]  = 0;
		Bs[ty1][tx1] = 0;
		
		if (a_topleft + tx  < A_topleft_end && load_y_in_A)  As[ty][tx]   = A[a_topleft + lda * ty  + tx];
		if (a_topleft + tx1 < A_topleft_end && load_y_in_A)  As[ty][tx1]  = A[a_topleft + lda * ty  + tx1];
		if (a_topleft + tx  < A_topleft_end && load_y1_in_A) As[ty1][tx]  = A[a_topleft + lda * ty1 + tx];
		if (a_topleft + tx1 < A_topleft_end && load_y1_in_A) As[ty1][tx1] = A[a_topleft + lda * ty1 + tx1];
		
		if (load_x_in_B  && b_topleft + ty  * ldb < B_topleft_end) Bs[ty][tx]   = B[b_topleft + ldb * ty  + tx];
		if (load_x1_in_B && b_topleft + ty  * ldb < B_topleft_end) Bs[ty][tx1]  = B[b_topleft + ldb * ty  + tx1];
		if (load_x_in_B  && b_topleft + ty1 * ldb < B_topleft_end) Bs[ty1][tx]  = B[b_topleft + ldb * ty1 + tx];
		if (load_x1_in_B && b_topleft + ty1 * ldb < B_topleft_end) Bs[ty1][tx1] = B[b_topleft + ldb * ty1 + tx1];
		
		__syncthreads();
		
		for (unsigned int k = 0; k < block_size; k++)
		{
			float As_ty_k = As[ty][k], As_ty1_k = As[ty1][k];
			float Bs_k_tx = Bs[k][tx], Bs_k_tx1 = Bs[k][tx1];
			Cs[0] += As_ty_k  * Bs_k_tx;
			Cs[1] += As_ty_k  * Bs_k_tx1;
			Cs[2] += As_ty1_k * Bs_k_tx;
			Cs[3] += As_ty1_k * Bs_k_tx1;
		}
		__syncthreads();
	}
	
	unsigned int row  = block_size * blockIdx.y + ty;
	unsigned int col  = block_size * blockIdx.x + tx;
	unsigned int row1 = row + 16;
	unsigned int col1 = col + 16;
	
	if (row  < c_height && col  < c_width) C[row  * ldc + col]  = Cs[0];
	if (row  < c_height && col1 < c_width) C[row  * ldc + col1] = Cs[1];
	if (row1 < c_height && col  < c_width) C[row1 * ldc + col]  = Cs[2];
	if (row1 < c_height && col1 < c_width) C[row1 * ldc + col1] = Cs[3];
}

__global__ 
void sgemm_5_16xworks(
	float *A, const unsigned int lda, 
	float *B, const unsigned int ldb, 
	float *C, const unsigned int ldc, 
	const unsigned int common_dim,
	const unsigned int c_height, 
	const unsigned int c_width
)
{
	__shared__ float As[block_size][block_size];
	__shared__ float Bs[block_size][block_size];
	
	const unsigned int A_topleft_begin = blockIdx.y * block_size * lda;
	const unsigned int A_topleft_end   = A_topleft_begin + common_dim;
	const unsigned int A_step_size	 = block_size;
	
	const unsigned int B_topleft_begin = blockIdx.x * block_size;
	const unsigned int B_topleft_end   = B_topleft_begin + common_dim * ldb;
	const unsigned int B_step_size	 = block_size * ldb;
	
	const unsigned int quart_bs = 8;
	float Cs[4][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
	float regA[4], regB[4];
	
	unsigned int a_topleft, b_topleft;
	for (a_topleft = A_topleft_begin, b_topleft = B_topleft_begin;
		 a_topleft < A_topleft_end;
		 a_topleft += A_step_size, b_topleft += B_step_size
		)
	{
		// Load sub-matrix of A & B into shared memory
		for (unsigned int x_offset = 0; x_offset < 4; x_offset++)
		{
			unsigned int tx = threadIdx.x + x_offset * quart_bs;
			bool tx_in_A = (a_topleft + tx) < A_topleft_end;
			bool tx_in_B = (block_size * blockIdx.x + tx) < c_width;
			
			for (unsigned int y_offset = 0; y_offset < 4; y_offset++)
			{
				unsigned int ty = threadIdx.y + y_offset * quart_bs;
				bool ty_in_A = (block_size * blockIdx.y + ty) < c_height;
				bool ty_in_B = (b_topleft + ty * ldb) < B_topleft_end;
				
				if (tx_in_A && ty_in_A) As[ty][tx] = A[a_topleft + lda * ty + tx]; else As[ty][tx] = 0;
				if (tx_in_B && ty_in_B) Bs[ty][tx] = B[b_topleft + ldb * ty + tx]; else Bs[ty][tx] = 0;
			}
		}
		__syncthreads();
		
		// Multiply & accumulate
		unsigned int ty0 = 0 * quart_bs + threadIdx.y;
		unsigned int ty1 = 1 * quart_bs + threadIdx.y;
		unsigned int ty2 = 2 * quart_bs + threadIdx.y;
		unsigned int ty3 = 3 * quart_bs + threadIdx.y;
		unsigned int tx0 = 0 * quart_bs + threadIdx.x;
		unsigned int tx1 = 1 * quart_bs + threadIdx.x;
		unsigned int tx2 = 2 * quart_bs + threadIdx.x;
		unsigned int tx3 = 3 * quart_bs + threadIdx.x;
		
		for (unsigned int k = 0; k < block_size; k++)
		{	
			regA[0] = As[ty0][k];
			regA[1] = As[ty1][k];
			regA[2] = As[ty2][k];
			regA[3] = As[ty3][k];
			
			regB[0] = Bs[k][tx0];
			regB[1] = Bs[k][tx1];
			regB[2] = Bs[k][tx2];
			regB[3] = Bs[k][tx3];

			Cs[0][0] += regA[0] * regB[0];
			Cs[0][1] += regA[0] * regB[1];
			Cs[0][2] += regA[0] * regB[2];
			Cs[0][3] += regA[0] * regB[3];
			
			Cs[1][0] += regA[1] * regB[0];
			Cs[1][1] += regA[1] * regB[1];
			Cs[1][2] += regA[1] * regB[2];
			Cs[1][3] += regA[1] * regB[3];
			
			Cs[2][0] += regA[2] * regB[0];
			Cs[2][1] += regA[2] * regB[1];
			Cs[2][2] += regA[2] * regB[2];
			Cs[2][3] += regA[2] * regB[3];
			
			Cs[3][0] += regA[3] * regB[0];
			Cs[3][1] += regA[3] * regB[1];
			Cs[3][2] += regA[3] * regB[2];
			Cs[3][3] += regA[3] * regB[3];
		}
		__syncthreads();
	}
	
	// Write back to C 
	for (unsigned int x_offset = 0; x_offset < 4; x_offset++)
	{
		unsigned int tx = threadIdx.x + x_offset * quart_bs;
		unsigned int col  = block_size * blockIdx.x + tx;
		
		for (unsigned int y_offset = 0; y_offset < 4; y_offset++)
		{
			unsigned int ty = threadIdx.y + y_offset * quart_bs;
			unsigned int row  = block_size * blockIdx.y + ty;
			
			if (row < c_height && col < c_width)
				C[row * ldc + col] = Cs[y_offset][x_offset];
		}
	}
}

#endif
