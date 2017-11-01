#ifndef _MY_SGEMM_CM_KERNELS_H_
#define _MY_SGEMM_CM_KERNELS_H_

// Column-Major format kernels

#define work_per_thread 8
#define RTS 4
#define tile_size 64

__global__ 
void sgemm_6_rm(
	float *A, const unsigned int lda, 
	float *B, const unsigned int ldb, 
	float *C, const unsigned int ldc, 
	const unsigned int common_dim,
	const unsigned int c_height, 
	const unsigned int c_width
)
{	
	// Thread identifiers
	const int col = threadIdx.x; // Local col ID (max: tile_size)
	const int row = threadIdx.y; // Local row ID (max: tile_size/work_per_thread == RTS)
	const int globalRow = tile_size * blockIdx.x + row; // Row ID of C (0..c_height)
	const int globalCol = tile_size * blockIdx.y + col; // Col ID of C (0..c_width)

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
		float *read_A_topleft = A + tile_size * blockIdx.x * lda + t * tile_size;
		float *read_B_topleft = B + tile_size * t * ldb + tile_size * blockIdx.y;
		#pragma unroll
		for (unsigned int w = 0; w < work_per_thread; w++)
		{
			As[row + w * RTS][col] = read_A_topleft[(w * RTS + row) * lda + col];
			Bs[row + w * RTS][col] = read_B_topleft[(w * RTS + row) * lda + col];
		}

		__syncthreads();

		// Perform the computation for a single tile
		#pragma unroll
		for (int k = 0; k < tile_size; k++) 
		{
			#pragma unroll
			for (int w = 0; w < work_per_thread; w++)
			{
				acc[w] += As[row + w * RTS][k] * Bs[k][col];
			}
		}
		__syncthreads();
	}
	
	// Store the final results in C
	#pragma unroll
	for (unsigned int w = 0; w < work_per_thread; w++) 
		C[(globalRow  + w * RTS) * ldc + globalCol] = acc[w];
}

__global__ 
void sgemm_6_cm(
	float *A, const unsigned int lda, 
	float *B, const unsigned int ldb, 
	float *C, const unsigned int ldc, 
	const unsigned int common_dim,
	const unsigned int c_height, 
	const unsigned int c_width
)
{	
	// Thread identifiers
	const int row = threadIdx.x; // Local row ID (max: tile_size)
	const int col = threadIdx.y; // Local col ID (max: tile_size/work_per_thread == RTS)
	const int globalRow = tile_size * blockIdx.x + row; // Row ID of C (0..c_height)
	const int globalCol = tile_size * blockIdx.y + col; // Col ID of C (0..c_width)

	// Local memory to fit a tile of TS*TS elements of A and B
	__shared__ float As[tile_size][tile_size];
	__shared__ float Bs[tile_size][tile_size];

	// Initialise the accumulation registers
	float acc[work_per_thread];
	#pragma unroll
	for (unsigned int w = 0; w < work_per_thread; w++) acc[w] = 0;
	
	// Loop over all tiles
	const unsigned int numTiles = common_dim / tile_size;
	for (unsigned int t = 0; t < numTiles; t++) 
	{
		// Load one tile of A and B into local memory
		const unsigned int tiledRow = tile_size * t + row;
		const unsigned int tiledCol = tile_size * t + col;
		#pragma unroll
		for (int w = 0; w < work_per_thread; w++) 
		{
			As[col + w * RTS][row] = A[(tiledCol + w * RTS) * lda + globalRow];
			Bs[col + w * RTS][row] = B[(globalCol + w * RTS) * ldb + tiledRow];
		}
		__syncthreads();

		// Perform the computation for a single tile
		#pragma unroll // K40c, M=K=N=2^11, 760 Gflops -> 810 Gflops
		for (unsigned int k = 0; k < tile_size; k++) 
		{
			#pragma unroll
			for (unsigned int w = 0; w < work_per_thread; w++) 
			{
				acc[w] += As[k][row] * Bs[col + w * RTS][k];
			}
		}
		__syncthreads();
	}

	// Store the final results in C
	#pragma unroll
	for (unsigned int w = 0; w < work_per_thread; w++) 
		C[(globalCol + w * RTS) * ldc + globalRow] = acc[w];
}

#define WIDTH 4	
#define TSM   128  // The tile-size in dimension M
#define TSN   128  // The tile-size in dimension N
#define TSK   32   // The tile-size in dimension K
#define WPTM  16   // The amount of work-per-thread in dimension M
#define WPTN  8    // The amount of work-per-thread in dimension N
#define RTSM  8    // The reduced tile-size in dimension M (== TSM/WPTM == number of threads)
#define RTSN  16   // The reduced tile-size in dimension N (== TSN/WPTN == number of threads)
#define LPTA  32   // The amount of loads-per-thread for A (== (TSK*WPTM*WPTN)/(TSN) )
#define LPTB  32   // The amount of loads-per-thread for B (== (TSK*WPTM*WPTN)/(TSM) )

#define TRANSPOSEX 16
#define TRANSPOSEY 16

// Simple transpose kernel for a P rows * Q columns matrix, Column-Major style
__global__ 
void transpose(
	const unsigned int P, const unsigned int Q,
	const float *input, float* output
) 
{	
	// Thread identifiers
	const unsigned int tx  = threadIdx.x;
	const unsigned int ty  = threadIdx.y;
	const unsigned int row = blockIdx.x * TRANSPOSEX + tx; // 0..P
	const unsigned int col = blockIdx.y * TRANSPOSEY + ty; // 0..Q

	// Set-up the local memory for shuffling
	__shared__ float buffer[TRANSPOSEX][TRANSPOSEY];

	// Swap the x and y coordinates to perform the rotation (coalesced)
	if (row < P && col < Q) buffer[ty][tx] = input[col * P + row];

	// Synchronise all threads
	__syncthreads();

	// We don't have to swap the x and y thread indices here,
	// because that's already done in the local memory
	const int new_col = blockIdx.y * TRANSPOSEY + tx;
	const int new_row = blockIdx.x * TRANSPOSEX + ty;

	// Store the transposed result (coalesced)
	if (new_col < Q && new_row < P) output[new_row * Q + new_col] = buffer[tx][ty];
}

__global__ 
void sgemm_7_reg2D(
	float *A, const unsigned int lda, 
	float *B, const unsigned int ldb, 
	float *C, const unsigned int ldc, 
	const unsigned int common_dim,
	const unsigned int c_height, 
	const unsigned int c_width
)
{
	const unsigned int tidm = threadIdx.x; // Local row ID (max: TSM/WPTM == RTSM)
	const unsigned int tidn = threadIdx.y; // Local col ID (max: TSN/WPTN == RTSN)
	const unsigned int offsetM = TSM * blockIdx.x; // Work-group offset
	const unsigned int offsetN = TSN * blockIdx.y; // Work-group offset

	// Local memory to fit a tile of A and B
	__shared__ float As[TSK][TSM];
	__shared__ float Bs[TSN][TSK+2]; // +2 to avoid bank comflict

	// Register space
	float Breg[WPTN];
	float acc[WPTM][WPTN];

	// Initialise the accumulation registers
	#pragma unroll
	for (unsigned int wm = 0; wm < WPTM; wm++) 
	{
		#pragma unroll
		for (unsigned int wn = 0; wn < WPTN; wn++) 
		{
			acc[wm][wn] = 0.0f;
		}
	}
	
	// Loop over all tiles
	const unsigned int numTiles = common_dim / TSK;
	unsigned int t = 0;
	do {

		// Load one tile of A and B into local memory
		#pragma unroll
		for (unsigned int la = 0; la < LPTA; la++) 
		{
			unsigned int tid = tidn * RTSM + tidm;
			unsigned int tile_id  = la * RTSN * RTSM + tid;
			unsigned int row_in_tile = tile_id % TSM;
			unsigned int col_in_tile = tile_id / TSM;
			unsigned int global_col = TSK * t + col_in_tile;
			As[col_in_tile][row_in_tile] = A[global_col * lda + offsetM + row_in_tile];
			Bs[row_in_tile][col_in_tile] = B[global_col * ldb + offsetN + row_in_tile];
			// If B is not transposed, the following line is very slow:
			// Bs[row_in_tile][col_in_tile] = B[global_col + (offsetN + row_in_tile)  * ldb];
		}
		__syncthreads();

		// Loop over the values of a single tile
		for (unsigned int k = 0; k < TSK; k++) 
		{

			// Cache the values of Bs in registers
			#pragma unroll
			for (unsigned int wn = 0; wn < WPTN; wn++) 
			{
				unsigned int col = tidn + wn * RTSN;
				Breg[wn] = Bs[col][k];
			}

			// Perform the computation
			#pragma unroll
			for (unsigned int wm = 0; wm < WPTM; wm++) 
			{
				unsigned int row = tidm + wm * RTSM;
				float Areg = As[k][row];
				#pragma unroll
				for (int wn = 0; wn < WPTN; wn++)
				{
					acc[wm][wn] += Areg * Breg[wn];
				}
			}
		}
		__syncthreads();

		// Next tile
		t++;
	} while (t < numTiles);

	// Store the final results in C
	#pragma unroll
	for (unsigned int wm = 0; wm < WPTM; wm++) 
	{
		unsigned int globalRow = offsetM + tidm + wm * RTSM;
		#pragma unroll
		for (int wn = 0; wn < WPTN; wn++) 
		{
			int globalCol = offsetN + tidn + wn * RTSN;
			C[globalCol * ldc + globalRow] = acc[wm][wn];
		}
	}
}

__global__ 
void sgemm_8_reg2Dvec(
	float4 *A, const unsigned int lda, 
	float4 *B, const unsigned int ldb, 
	float *C, const unsigned int ldc, 
	const unsigned int common_dim,
	const unsigned int c_height, 
	const unsigned int c_width
)
{
	const unsigned int tidm = threadIdx.x; // Local row ID (max: TSM/WPTM == RTSM)
	const unsigned int tidn = threadIdx.y; // Local col ID (max: TSN/WPTN == RTSN)
	const unsigned int offsetM = TSM * blockIdx.x; // Work-group offset
	const unsigned int offsetN = TSN * blockIdx.y; // Work-group offset

	// Local memory to fit a tile of A and B
	__shared__ float As[TSK][TSM];
	__shared__ float Bs[TSK][TSN]; 

	// Register space
	float Breg[WPTN];
	float acc[WPTM][WPTN];

	// Initialise the accumulation registers
	#pragma unroll
	for (unsigned int wm = 0; wm < WPTM; wm++) 
	{
		#pragma unroll
		for (unsigned int wn = 0; wn < WPTN; wn++) 
		{
			acc[wm][wn] = 0.0f;
		}
	}
	
	// Loop over all tiles
	const unsigned int numTiles = common_dim / TSK;
	unsigned int t = 0;
	do {
		// Load one tile of A and B into local memory
		#pragma unroll
		for (unsigned int la = 0; la < LPTA / WIDTH; la++) 
		{
			unsigned int tid = tidn * RTSM + tidm;
			unsigned int tile_id = la * RTSN * RTSM + tid;
			unsigned int row_in_tile = tile_id % (TSM / WIDTH);
			unsigned int col_in_tile = tile_id / (TSM / WIDTH);
			unsigned int global_col = TSK * t + col_in_tile;
			unsigned int indexA = global_col * (lda / WIDTH) + (offsetM / WIDTH) + row_in_tile;
			unsigned int indexB = global_col * (ldb / WIDTH) + (offsetN / WIDTH) + row_in_tile;
			
			// float4 vecA = A[indexA];
			// float4 vecB = B[indexB];
			// The following could boost from 1485 GFlops -> 1610 GFlops
			float4 vecA = __ldg(&A[indexA]);
			float4 vecB = __ldg(&B[indexB]);
			
			As[col_in_tile][WIDTH * row_in_tile + 0] = vecA.x;
			As[col_in_tile][WIDTH * row_in_tile + 1] = vecA.y;
			As[col_in_tile][WIDTH * row_in_tile + 2] = vecA.z;
			As[col_in_tile][WIDTH * row_in_tile + 3] = vecA.w;
			
			Bs[col_in_tile][WIDTH * row_in_tile + 0] = vecB.x;
			Bs[col_in_tile][WIDTH * row_in_tile + 1] = vecB.y;
			Bs[col_in_tile][WIDTH * row_in_tile + 2] = vecB.z;
			Bs[col_in_tile][WIDTH * row_in_tile + 3] = vecB.w;
			
		}
		__syncthreads();

		// Loop over the values of a single tile
		for (unsigned int k = 0; k < TSK; k++) 
		{

			// Cache the values of Bs in registers
			#pragma unroll
			for (unsigned int wn = 0; wn < WPTN; wn++) 
			{
				unsigned int col = tidn + wn * RTSN;
				Breg[wn] = Bs[k][col];
			}
			
			
			// Perform the computation
			#pragma unroll
			for (unsigned int wm = 0; wm < WPTM; wm++) 
			{
				unsigned int row = tidm + wm * RTSM;
				float Areg = As[k][row];
				#pragma unroll
				for (int wn = 0; wn < WPTN; wn++)
				{
					acc[wm][wn] += Areg * Breg[wn];
				}
			}
		}
		__syncthreads();

		// Next tile
		t++;
	} while (t < numTiles);

	// Store the final results in C
	#pragma unroll
	for (unsigned int wm = 0; wm < WPTM; wm++) 
	{
		unsigned int globalRow = offsetM + tidm + wm * RTSM;
		#pragma unroll
		for (int wn = 0; wn < WPTN; wn++) 
		{
			unsigned int globalCol = offsetN + tidn + wn * RTSN;
			C[globalCol * ldc + globalRow] = acc[wm][wn];
		}
	}
}

#define CEIL_DIV(x,y) (((x) + (y) - 1) / (y))
#define PADDINGX 16
#define PADDINGY 16

// Pad the P rows Q columns matrix into P_pad rows * Q_pad cols matrix, Column-Major style operation
__global__
void padZeros(
	const unsigned int P, const unsigned int Q,
	const unsigned int P_pad, const unsigned int Q_pad,
	const float *input, float *output
)
{
	const unsigned int tx = blockIdx.x * PADDINGX + threadIdx.x;
	const unsigned int ty = blockIdx.y * PADDINGY + threadIdx.y;
	if (tx < P_pad && ty < Q_pad)
	{
		float val = 0.0f;
		if (tx < P && ty < Q) val = input[ty * P + tx];
		output[ty * P_pad + tx] = val;
	}
}

// Inverse operation of paddingZeros
__global__
void removePadZeros(
	const unsigned int P_pad, const unsigned int Q_pad,
	const unsigned int P, const unsigned int Q,
	const float *input, float *output
)
{
	const unsigned int tx = blockIdx.x * PADDINGX + threadIdx.x;
	const unsigned int ty = blockIdx.y * PADDINGY + threadIdx.y;
	if (tx < P && ty < Q)
	{
		output[ty * P + tx] = input[ty * P_pad + tx];
	}
}

#endif
