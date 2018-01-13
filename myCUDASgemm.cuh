#define KernelParameters const float *A, const unsigned int lda, \
	const float *B, const unsigned int ldb, \
	float *C, const unsigned int ldc, \
	const float alpha, const float beta, \
	const unsigned int common_dim, \
	const unsigned int c_height, \
	const unsigned int c_width

/* -------------- Prototype SGEMM Kernels -------------- */ 

#define work_per_thread 8
#define RTS				4
#define tile_size		32
#define block_size		32

__global__ 
void sgemm_1_naive(KernelParameters)
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
	C[row * ldc + col] = accu * alpha + beta * C[row * ldc + col];
}

__global__ 
void sgemm_2_tiling(KernelParameters)
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
		// Load the sub-block of A and B to shared memory
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
		
		// Compute 
		for (unsigned int k = 0; k < block_size; k++)
			accumulated_Cij += As[threadIdx.y][k] * Bs[threadIdx.x][k]; // shared mem bank conflict
		__syncthreads();
	}
	
	// Accumulate row i of A and column j of B and write to the output array
	unsigned int row = block_size * blockIdx.y + threadIdx.y;
	unsigned int col = block_size * blockIdx.x + threadIdx.x;
	if (row < c_height && col < c_width) 
		C[row * ldc + col] = alpha * accumulated_Cij + beta * C[row * ldc + col];
}

__global__ 
void sgemm_3_coalescing(KernelParameters)
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
		// Load the sub-block of A and B to shared memory
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
		
		// Compute 
		#pragma unroll
		for (unsigned int k = 0; k < block_size; k++)
			accumulated_Cij += As[threadIdx.y][k] * Bs[k][threadIdx.x];
		__syncthreads();
	}
	
	// Accumulate row i of A and column j of B
	unsigned int row = block_size * blockIdx.y + threadIdx.y;
	unsigned int col = block_size * blockIdx.x + threadIdx.x;
	
	// Write to the output array
	if (row < c_height && col < c_width)
		C[row * ldc + col] = alpha * accumulated_Cij + beta * C[row * ldc + col];
}

__global__ 
void sgemm_4_morework(KernelParameters)
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
	
	// Each thread is responsible for 4 elements in C and load 4 elements in A & B
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
		// Load the sub-block of A and B to shared memory
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
		
		// Compute 
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
	
	// Write to the output array
	unsigned int row  = block_size * blockIdx.y + ty;
	unsigned int col  = block_size * blockIdx.x + tx;
	unsigned int row1 = row + 16;
	unsigned int col1 = col + 16;
	
	if (row  < c_height && col  < c_width) C[row  * ldc + col]  = alpha * Cs[0] + beta * C[row  * ldc + col];
	if (row  < c_height && col1 < c_width) C[row  * ldc + col1] = alpha * Cs[1] + beta * C[row  * ldc + col1];
	if (row1 < c_height && col  < c_width) C[row1 * ldc + col]  = alpha * Cs[2] + beta * C[row1  * ldc + col];
	if (row1 < c_height && col1 < c_width) C[row1 * ldc + col1] = alpha * Cs[3] + beta * C[row1  * ldc + col1];
}

/* -------------- Auxilliary Kernels -------------- */ 
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define PADDINGX   16
#define PADDINGY   16
#define TRANSPOSEX 16
#define TRANSPOSEY 16

// Padding row-major matrix with 0 on boundary to given size
__global__
void padZeros_rm(
	const unsigned int rows, const unsigned int columns,
	const unsigned int pad_rows, const unsigned int pad_columns,
	const float *input, float *output
)
{
	const unsigned int col = blockIdx.x * PADDINGX + threadIdx.x;
	const unsigned int row = blockIdx.y * PADDINGY + threadIdx.y;
	if (col < pad_columns && row < pad_rows)
	{
		float val = 0.0f;
		if (col < columns && row < rows) val = input[row * columns + col];
		output[row * pad_columns + col] = val;
	}
}

// Inverse operation of padZeros(), the given matrix is row-major
__global__
void removePadZeros_rm(
	const unsigned int pad_rows, const unsigned int pad_columns,
	const unsigned int rows, const unsigned int columns,
	const float *input, float *output
)
{
	const unsigned int col = blockIdx.x * PADDINGX + threadIdx.x;
	const unsigned int row = blockIdx.y * PADDINGY + threadIdx.y;
	if (col < columns && row < rows)
	{
		output[row * columns + col] = input[row * pad_columns + col];
	}
}

/* -------------- Advanced SGEMM Kernels -------------- */ 

__global__ 
void sgemm_5_morework8x(KernelParameters)
{	
	// Thread identifiers
	const unsigned int col = threadIdx.x; // Local col ID (max: tile_size)
	const unsigned int row = threadIdx.y; // Local row ID (max: tile_size/work_per_thread == RTS)
	const unsigned int globalRow = tile_size * blockIdx.y + row; // Row ID of C (0..c_height)
	const unsigned int globalCol = tile_size * blockIdx.x + col; // Col ID of C (0..c_width)

	// Local memory to fit a tile of TS*TS elements of A and B
	__shared__ float As[tile_size][tile_size];
	__shared__ float Bs[tile_size][tile_size];

	// Initialise the accumulation registers
	float acc[work_per_thread];
	#pragma unroll
	for (int w = 0; w < work_per_thread; w++) acc[w] = 0;
	
	// Loop over all tiles
	const unsigned int numTiles = common_dim / tile_size;
	for (unsigned int t = 0; t < numTiles; t++) 
	{
		const float *read_A_topleft = A + tile_size * blockIdx.y * lda + t * tile_size;
		const float *read_B_topleft = B + tile_size * t * ldb + tile_size * blockIdx.x;
		
		// Load A tile and B tile to shm
		#pragma unroll
		for (unsigned int w = 0; w < work_per_thread; w++)
		{
			As[row + w * RTS][col] = read_A_topleft[(w * RTS + row) * lda + col];
			Bs[row + w * RTS][col] = read_B_topleft[(w * RTS + row) * ldb + col];
		}
		__syncthreads();

		// Perform the computation for a single tile
		#pragma unroll
		for (unsigned int k = 0; k < tile_size; k++) 
		{
			#pragma unroll
			for (unsigned int w = 0; w < work_per_thread; w++)
			{
				acc[w] += As[row + w * RTS][k] * Bs[k][col];
			}
		}
		__syncthreads();
	}
	
	// Store the final results in C
	#pragma unroll
	for (unsigned int w = 0; w < work_per_thread; w++) 
		C[(globalRow + w * RTS) * ldc + globalCol] = alpha * acc[w] + beta * C[(globalRow + w * RTS) * ldc + globalCol];
}

#define TILE_SIZE  64  // Tile size for loading into shared memory
#define WPTM       8   // Work per thread on dimension M (the height of C), == TILE_SIZE / RTSM
#define WPTN       4   // Work per thread on dimension N (the width of C),  == TILE_SIZE / RTSN
#define RTSM       8   // Reduced tile size on dimension M 
#define RTSN       16  // Reduced tile size on dimension N, should not be smaller than 16 for coalesced memory accessing 

__global__ 
void sgemm_6_2Dreg(KernelParameters)
{	
	// Thread identifiers
	const unsigned int col = threadIdx.x; // Local col ID (max: TILE_SIZE/WPTN == RTSN)
	const unsigned int row = threadIdx.y; // Local row ID (max: TILE_SIZE/WPTM == RTSM)
	const unsigned int globalRow = TILE_SIZE * blockIdx.y + row; // Row ID of C (0..c_height)
	const unsigned int globalCol = TILE_SIZE * blockIdx.x + col; // Col ID of C (0..c_width)

	// Local memory to fit a tile of TS*TS elements of A and B
	__shared__ float As[TILE_SIZE][TILE_SIZE];
	__shared__ float Bs[TILE_SIZE][TILE_SIZE];

	// Initialise the accumulation registers
	float acc[WPTM][WPTN];
	float Areg[WPTM], Breg[WPTN];
	#pragma unroll
	for (unsigned int wm = 0; wm < WPTM; wm++) 
		for (unsigned int wn = 0; wn < WPTN; wn++) 
			acc[wm][wn] = 0;
	
	// Loop over all tiles
	const unsigned int numTiles = common_dim / TILE_SIZE;
	for (unsigned int t = 0; t < numTiles; t++) 
	{
		const float *read_A_topleft = A + TILE_SIZE * blockIdx.y * lda + t * TILE_SIZE;
		const float *read_B_topleft = B + TILE_SIZE * t * ldb + TILE_SIZE * blockIdx.x;
		
		// Load A tile and B tile to the shm
		#pragma unroll
		for (unsigned int wm = 0; wm < WPTM; wm++)
		{
			#pragma unroll
			for (unsigned int wn = 0; wn < WPTN; wn++)
			{
				As[row + wm * RTSM][col + wn * RTSN] = read_A_topleft[(wm * RTSM + row) * lda + (col + wn * RTSN)];
				Bs[row + wm * RTSM][col + wn * RTSN] = read_B_topleft[(wm * RTSM + row) * ldb + (col + wn * RTSN)];
			}
		}
		__syncthreads();

		// Perform the computation for a single tile
		#pragma unroll
		for (unsigned int k = 0; k < TILE_SIZE; k++) 
		{
			// Cache the values of As in registers
			#pragma unroll
			for (unsigned int wm = 0; wm < WPTM; wm++) 
			{
				unsigned int irow = row + wm * RTSM;
				Areg[wm] = As[irow][k];
			}
			
			// Cache the values of Bs in registers
			#pragma unroll
			for (unsigned int wn = 0; wn < WPTN; wn++) 
			{
				unsigned int icol = col + wn * RTSN;
				Breg[wn] = Bs[k][icol];
			}
			
			// Perform the computation
			#pragma unroll
			for (unsigned int wm = 0; wm < WPTM; wm++) 
			{
				#pragma unroll
				for (unsigned int wn = 0; wn < WPTN; wn++)
				{
					acc[wm][wn] += Areg[wm] * Breg[wn];
				}
			}
		}
		__syncthreads();
	}
	
	// Store the final results in C
	#pragma unroll
	for (unsigned int wm = 0; wm < WPTM; wm++)
	{
		unsigned int c_dim1 = (globalRow + wm * RTSM) * ldc;
		#pragma unroll
		for (unsigned int wn = 0; wn < WPTN; wn++)
		{
			unsigned int c_dim2 = globalCol + wn * RTSN;
			C[c_dim1 + c_dim2] = alpha * acc[wm][wn] + beta * C[c_dim1 + c_dim2];
		}
	}
}

#define TSM   128  // Tile size in dimension M for loading into shared memory
#define TSN   128  // Tile size in dimension N for loading into shared memory
#define TSK   32   // Tile size in dimension K for loading into shared memory
#define RTS_M 16   // Reduced tile size on dimension M (how many thread on dimension M)
#define RTS_N 16   // Reduced tile size on dimension N (how many thread on dimension N)
#define WPT_M 8    // Each thread is responsible for how many elements on dimension M, == TSM / RTS_M
#define WPT_N 8    // Each thread is responsible for how many elements on dimension N, == TSN / RTS_N
#define WPTAM 8    // Each thread is responsible for how many elements on reading A on dimension M, == TSM / RTS_M
#define WPTAK 2    // Each thread is responsible for how many elements on reading A on dimension K, == TSK / RTS_N
#define WPTBK 2    // Each thread is responsible for how many elements on reading B on dimension K, == TSK / RTS_M
#define WPTBN 8    // Each thread is responsible for how many elements on reading B on dimension N, == TSN / RTS_N

__global__ 
void sgemm_7_2Dreg_rect(KernelParameters)
{	
	// Thread identifiers
	const unsigned int col = threadIdx.x; // Local col ID: 0, ..., RTS_N-1
	const unsigned int row = threadIdx.y; // Local row ID: 0, ..., RTS_M-1
	const unsigned int globalRow = TSM * blockIdx.y + row; // Row ID of C (0..c_height)
	const unsigned int globalCol = TSN * blockIdx.x + col; // Col ID of C (0..c_width)

	// Local memory to fit a tile of TS*TS elements of A and B
	__shared__ float As[TSM][TSK];
	__shared__ float Bs[TSK][TSN];

	// Initialise the accumulation registers
	float Cres[WPT_M][WPT_N];
	float Breg[WPT_N];
	#pragma unroll
	for (unsigned int wm = 0; wm < WPT_M; wm++)
		#pragma unroll	
		for (unsigned int wn = 0; wn < WPT_N; wn++) 
			Cres[wm][wn] = 0;
	
	// Loop over all tiles
	const unsigned int numTiles = common_dim / TSK;
	for (unsigned int t = 0; t < numTiles; t++) 
	{
		const float *read_A_topleft = A + TSM * blockIdx.y * lda + t * TSK;
		const float *read_B_topleft = B + TSK * t * ldb + TSN * blockIdx.x;
		
		// Load A tile and B tile to the shm
		#pragma unroll
		for (unsigned int wm = 0; wm < WPTAM; wm++)
		{
			int irow = row + wm * RTS_M;
			#pragma unroll
			for (unsigned int wn = 0; wn < WPTAK; wn++)
				As[irow][col + wn * RTS_N] = read_A_topleft[irow * lda + (col + wn * RTS_N)];
		}
		#pragma unroll
		for (unsigned int wm = 0; wm < WPTBK; wm++)
		{
			int irow = row + wm * RTS_M;
			#pragma unroll
			for (unsigned int wn = 0; wn < WPTBN; wn++)
				Bs[irow][col + wn * RTS_N] = read_B_topleft[irow * ldb + (col + wn * RTS_N)];
		}
		__syncthreads();

		// Perform the computation for a single tile
		#pragma unroll
		for (unsigned int k = 0; k < TSK; k++) 
		{
			// Cache the values of Bs in registers
			#pragma unroll
			for (unsigned int wn = 0; wn < WPT_N; wn++) 
				Breg[wn] = Bs[k][col + wn * RTS_N];
			
			// Perform the computation
			#pragma unroll
			for (unsigned int wm = 0; wm < WPT_M; wm++) 
			{
				float Areg_wm = As[row + wm * RTS_M][k];
				#pragma unroll
				for (unsigned int wn = 0; wn < WPT_N; wn++)
				{
					Cres[wm][wn] += Areg_wm * Breg[wn];
				}
			}
		}
		__syncthreads();
	}
	
	// Store the final results in C
	#pragma unroll
	for (unsigned int wm = 0; wm < WPT_M; wm++)
	{
		unsigned int c_dim1 = (globalRow + wm * RTS_M) * ldc;
		#pragma unroll
		for (unsigned int wn = 0; wn < WPT_N; wn++)
		{
			unsigned int c_coord = globalCol + wn * RTS_N + c_dim1;
			C[c_coord] = alpha * Cres[wm][wn] + beta * C[c_coord];
		}
	}
}

#define VPTAK 4    // Each thread is responsible for how many float4 on reading A on dimension K, == TSK * TSM / (RTS_M * RTS_N * 4)
#define TPAK  2    // How many thread can read a row of tile A, == TSK / (VPTAK * 4)
#define VPTBN 4    // Each thread is responsible for how many float4 on reading B on dimension N, == TSK * TSM / (RTS_M * RTS_N * 4)
#define TPBN  8    // How many thread can read a row of tile B, == TSN / (VPTBN * 4)

__global__ 
void sgemm_8_2Dreg_float4(
	const float4 *A, const unsigned int lda, 
	const float4 *B, const unsigned int ldb, 
	float *C, const unsigned int ldc, 
	const float alpha, const float beta, 
	const unsigned int common_dim,
	const unsigned int c_height, 
	const unsigned int c_width
)
{	
	// Thread identifiers
	const unsigned int col = threadIdx.x; // Local col ID: 0, ..., RTS_N-1
	const unsigned int row = threadIdx.y; // Local row ID: 0, ..., RTS_M-1
	const unsigned int globalRow = TSM * blockIdx.y + row; // Row ID of C (0..c_height)
	const unsigned int globalCol = TSN * blockIdx.x + col; // Col ID of C (0..c_width)
	const unsigned int block_id  = row * blockDim.x + col;

	// Local memory to fit a tile of TS*TS elements of A and B
	__shared__ float As[TSM][TSK];
	__shared__ float Bs[TSK][TSN];

	// Initialise the accumulation registers
	float Cres[WPT_M][WPT_N];
	float Breg[WPT_N];
	#pragma unroll
	for (unsigned int wm = 0; wm < WPT_M; wm++)
		#pragma unroll	
		for (unsigned int wn = 0; wn < WPT_N; wn++) 
			Cres[wm][wn] = 0;
	
	// Loop over all tiles
	const unsigned int numTiles = common_dim / TSK;
	for (unsigned int t = 0; t < numTiles; t++) 
	{
		// Load A tile and B tile to the shm
		unsigned int Arow = block_id / TPAK;
		unsigned int Acol = (block_id % TPAK) * 16;
		unsigned int Brow = block_id / TPBN;
		unsigned int Bcol = (block_id % TPBN) * 16;
		// unsigned int indexA = ((TSM * blockIdx.y * lda + t * TSK) + (Arow * lda + Acol)) / 4;
		// unsigned int indexB = ((TSK * t * ldb + TSN * blockIdx.x) + (Brow * ldb + Bcol)) / 4;
		unsigned int indexA = ((TSM * blockIdx.y + Arow) * lda + t * TSK + Acol) / 4;
		unsigned int indexB = ((TSK * t + Brow) * ldb + TSN * blockIdx.x + Bcol) / 4;
		#pragma unroll
		for (unsigned int vec = 0; vec < 4; vec++)
		{
			float4 vecA = __ldg(&A[indexA + vec]);
			float4 vecB = __ldg(&B[indexB + vec]);
			As[Arow][Acol + vec * 4 + 0] = vecA.x;
			As[Arow][Acol + vec * 4 + 1] = vecA.y;
			As[Arow][Acol + vec * 4 + 2] = vecA.z;
			As[Arow][Acol + vec * 4 + 3] = vecA.w;
			Bs[Brow][Bcol + vec * 4 + 0] = vecB.x;
			Bs[Brow][Bcol + vec * 4 + 1] = vecB.y;
			Bs[Brow][Bcol + vec * 4 + 2] = vecB.z;
			Bs[Brow][Bcol + vec * 4 + 3] = vecB.w;
		}
		__syncthreads();

		// Perform the computation for a single tile
		#pragma unroll
		for (unsigned int k = 0; k < TSK; k++) 
		{
			// Cache the values of Bs in registers
			#pragma unroll
			for (unsigned int wn = 0; wn < WPT_N; wn++) 
				Breg[wn] = Bs[k][col + wn * RTS_N];
			
			// Perform the computation
			#pragma unroll
			for (unsigned int wm = 0; wm < WPT_M; wm++) 
			{
				float Areg_wm = As[row + wm * RTS_M][k];
				#pragma unroll
				for (unsigned int wn = 0; wn < WPT_N; wn++)
				{
					Cres[wm][wn] += Areg_wm * Breg[wn];
				}
			}
		}
		__syncthreads();
	}
	
	// Store the final results in C
	#pragma unroll
	for (unsigned int wm = 0; wm < WPT_M; wm++)
	{
		unsigned int c_dim1 = (globalRow + wm * RTS_M) * ldc;
		#pragma unroll
		for (unsigned int wn = 0; wn < WPT_N; wn++)
		{
			unsigned int c_coord = globalCol + wn * RTS_N + c_dim1;
			C[c_coord] = alpha * Cres[wm][wn] + beta * C[c_coord];
		}
	}
}
