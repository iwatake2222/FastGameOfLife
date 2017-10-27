#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"
#include "stdlib.h"
#include <string.h>
#include "algorithmCudaNormal.h"
#include "algorithmCudaNormalInternal.h"


namespace AlgorithmCudaNormal
{
#if 0
}	// indent guard
#endif

__forceinline__ __device__ void updateCell(int* matDst, int* matSrc, int globalIndex, int cnt)
{
	if (matSrc[globalIndex] == 0) {
		if (cnt == 3) {
			// birth
			matDst[globalIndex] = 1;
		} else {
			// keep dead
			matDst[globalIndex] = 0;
		}
	} else {
		if (cnt <= 2 || cnt >= 5) {
			// die
			matDst[globalIndex] = 0;
		} else {
			// keep alive (age++)
			matDst[globalIndex] = matSrc[globalIndex] + 1;
		}
	}
}

/*
 * when block size is (32,32), calculation block is (30,30)
 * the 1st block calculates (1,1) - (30,30)  (using matrix[(0,0) - (31,31)])
 * the 2nd block calculates (31,1) - (60,30) (using matrix[(30,0) - (61,0)])
 * Note: matrix memory size is (1 + width + 1, 1 + height + 1) (the real matrix is [(1,1) - (memWidth - 2, memHeight - 2)]
*/
__global__ void loop_2_stream(int* matDst, int *matSrc, int width, int height, int memWidth, int memHeight, int offsetY)
{
	__shared__ int tile[BLOCK_SIZE_H][BLOCK_SIZE_W];
	/* this is position on memory */
	int globalX = blockIdx.x * (blockDim.x - 2 * MEMORY_MARGIN) + threadIdx.x;	// <- increase 30(block size - 2) per block
	int globalY = blockIdx.y * (blockDim.y - 2 * MEMORY_MARGIN) + threadIdx.y;	// <- increase 30(block size - 2) per block
	globalY += offsetY;
	int localX = threadIdx.x;
	int localY = threadIdx.y;

	if (globalX >= memWidth || globalY >= memHeight) return;

	/* copy data from global memory to shared memory [(0,0) - (31,31)] */
	int thisCell = tile[localY][localX] = matSrc[memWidth * globalY + globalX];
	__syncthreads();

	if (globalX >= memWidth - 1 || globalY >= memHeight - 1 || localX == 0 || localX == blockDim.x - 1 || localY == 0 || localY == blockDim.y - 1) return;
	/* calculate if [(1,1) - (30,30)] */

	int cnt;
	cnt = (tile[localY - 1][localX - 1] != 0) + (tile[localY - 1][localX - 0] != 0) + (tile[localY - 1][localX + 1] != 0)
		+ (tile[localY - 0][localX - 1] != 0) + (thisCell != 0) + (tile[localY - 0][localX + 1] != 0)
		+ (tile[localY + 1][localX - 1] != 0) + (tile[localY + 1][localX - 0] != 0) + (tile[localY + 1][localX + 1] != 0);

	updateCell(matDst, matSrc, memWidth * globalY + globalX, cnt);
}

/* The algorithm using alias area on 4 corners and edges so that main logic doen't need to consider border
 *  with shared memory
 *  with stream
 *  Note: matrix memory size is (1 + width + 1, 1 + height + 1) (the real matrix is [(1,1) - (memWidth - 2, memHeight - 2)]
*/
void process_2_stream(ALGORITHM_CUDA_NORMAL_PARAM *param, int width, int height)
{
	int memWidth = width + 2 * MEMORY_MARGIN;
	int memHeight = height + 2 * MEMORY_MARGIN;

	int heightStream = ceil((double)memHeight / NUM_STREAM);

	/* block size setting for main logic
	 * do copy per 32(BLOCK_SIZE)
	 * do calculation per 30(BLOCK_SIZE-2)
	 * the number of block is ceil(width / 30)
	*/
	dim3 block(BLOCK_SIZE_W, BLOCK_SIZE_H, 1);
	dim3 grid((int)ceil(width / (double)(BLOCK_SIZE_W - 2 * MEMORY_MARGIN)), (int)ceil(height / (double)(BLOCK_SIZE_H - 2 * MEMORY_MARGIN) / NUM_STREAM), 1);

	/* Create alias area in CPU at first */
	int *p = param->hostMatSrc;
	memcpy(p, p + (memHeight - 2) * memWidth, memWidth * sizeof(int));
	memcpy(p + (memHeight - 1) * memWidth, p + (1) * memWidth, memWidth * sizeof(int));
	for (int y = 1; y < memHeight - 1; y++) {
		p[memWidth * y + 0] = p[memWidth * y + memWidth - 2];
		p[memWidth * y + memWidth - 1] = p[memWidth * y + 1];
	}
	p[memWidth * 0 + 0] = p[memWidth * (memHeight - 2) + memWidth - 2];
	p[memWidth * 0 + memWidth - 1] = p[memWidth * (memHeight - 2) + 1];
	p[memWidth * (memHeight - 1) + 0] = p[memWidth * (1) + memWidth - 2];
	p[memWidth * (memHeight - 1) + memWidth - 1] = p[memWidth * (1) + 1];

	/* copy border line data at first to simplyfy logic of each stream (no need to consider border line) */
	for (int i = 0; i < NUM_STREAM; i++) {
		int offsetFirstLine = (i * heightStream) * memWidth;
		CHECK(cudaMemcpy(param->devMatSrc + offsetFirstLine, param->hostMatSrc + offsetFirstLine, memWidth * sizeof(int), cudaMemcpyHostToDevice));
		int offsetLastLine = ((i + 1) * heightStream - 1) * memWidth;
		if (offsetLastLine < (memHeight - 1) * memWidth ) {
			CHECK(cudaMemcpy(param->devMatSrc + offsetLastLine, param->hostMatSrc + offsetLastLine, memWidth * sizeof(int), cudaMemcpyHostToDevice));
		} else {
			CHECK(cudaMemcpy(param->devMatSrc + (memHeight - 1) * memWidth, param->hostMatSrc + (memHeight - 1) * memWidth, memWidth * sizeof(int), cudaMemcpyHostToDevice));
		}
	}

	/* create stream(copy(h2d), kernel, copy(d2h)) */
	for (int i = 0; i < NUM_STREAM; i++) {
		cudaStream_t* pStream = (cudaStream_t*)(param->pStream[i]);
		int offsetY = i * heightStream;
		int copyLine = heightStream;
		if (offsetY + heightStream > memHeight) copyLine = memHeight - offsetY;
		CHECK(cudaMemcpyAsync(param->devMatSrc + offsetY * memWidth, param->hostMatSrc + offsetY * memWidth, memWidth * copyLine * sizeof(int), cudaMemcpyHostToDevice, *pStream));
		loop_2_stream << < grid, block, 0, *pStream >> > (param->devMatDst, param->devMatSrc, width, height, memWidth, memHeight, offsetY);
		CHECK(cudaMemcpyAsync(param->hostMatDst + offsetY * memWidth, param->devMatDst + offsetY * memWidth, memWidth * copyLine * sizeof(int), cudaMemcpyDeviceToHost, *pStream));
	}

	for (int i = 0; i < NUM_STREAM; i++) {
		cudaStream_t* pStream = (cudaStream_t*)(param->pStream[i]);
		CHECK(cudaStreamSynchronize(*pStream));
	}

	swapMat(param);
	// hostMatSrc is ready to be displayed
}


}
