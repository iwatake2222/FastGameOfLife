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
__global__ void loop_3_stream(int* matDst, int *matSrc, int width, int height, int memWidth, int memHeight, int offsetY)
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

__global__ void loop_3_stream_makeAliasRow(int *mat, int width, int height, int memWidth, int memHeight)
{
	int *p = mat;
	int devMatX = blockIdx.x * blockDim.x + threadIdx.x + 1;
	p[0 + devMatX] = p[memWidth * (memHeight - 2) + devMatX];
	p[memWidth * (memHeight - 1) + devMatX] = p[memWidth * 1 + devMatX];
	
	if (devMatX == 1) {
		p[memWidth * 0 + 0] = p[memWidth * (memHeight - 2) + memWidth - 2];
		p[memWidth * 0 + memWidth - 1] = p[memWidth * (memHeight - 2) + 1];
		p[memWidth * (memHeight - 1) + 0] = p[memWidth * (1) + memWidth - 2];
		p[memWidth * (memHeight - 1) + memWidth - 1] = p[memWidth * (1) + 1];
	}
}

__global__ void loop_3_stream_makeAliasCol(int *mat, int width, int height, int memWidth, int memHeight)
{
	int *p = mat;
	int devMatY = blockIdx.x * blockDim.x + threadIdx.x + 1;

	p[memWidth * devMatY + 0] = p[memWidth * devMatY + memWidth - 2];
	p[memWidth * devMatY + memWidth - 1] = p[memWidth * devMatY + 1];
	
}

/* The algorithm using alias area on 4 corners and edges so that main logic doen't need to consider border
 *  with shared memory
 *  with stream
 *  do not copy matSrc from host to device all the time 
 *  Note: matrix memory size is (1 + width + 1, 1 + height + 1) (the real matrix is [(1,1) - (memWidth - 2, memHeight - 2)]
*/
void process_3_stream(ALGORITHM_CUDA_NORMAL_PARAM *param, int width, int height)
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

	if (param->isMatrixUpdated) {
		/* create alias area in host memory (copy from host to device is done by stream later) */
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
			if (offsetLastLine < (memHeight - 1) * memWidth) {
				CHECK(cudaMemcpy(param->devMatSrc + offsetLastLine, param->hostMatSrc + offsetLastLine, memWidth * sizeof(int), cudaMemcpyHostToDevice));
			} else {
				CHECK(cudaMemcpy(param->devMatSrc + (memHeight - 1) * memWidth, param->hostMatSrc + (memHeight - 1) * memWidth, memWidth * sizeof(int), cudaMemcpyHostToDevice));
			}
		}
	} else {
		/* create alias area in device memory */
		dim3 blockW(BLOCK_SIZE_W);
		dim3 gridW(width / BLOCK_SIZE_W);
		dim3 blockH(BLOCK_SIZE_H);
		dim3 gridH(height / BLOCK_SIZE_H);
		loop_3_stream_makeAliasRow <<<gridW, blockW >>> (param->devMatSrc, width, height, memWidth, memHeight);
		loop_3_stream_makeAliasCol << <gridH, blockH >> > (param->devMatSrc, width, height, memWidth, memHeight);
		CHECK(cudaDeviceSynchronize());
	}

	/* create stream(copy(h2d), kernel, copy(d2h)) */
	for (int i = 0; i < NUM_STREAM; i++) {
		cudaStream_t* pStream = (cudaStream_t*)(param->pStream[i]);
		int offsetY = i * heightStream;
		int copyLine = heightStream;
		if (offsetY + heightStream > memHeight) copyLine = memHeight - offsetY;
		if (param->isMatrixUpdated) {
			CHECK(cudaMemcpyAsync(param->devMatSrc + offsetY * memWidth, param->hostMatSrc + offsetY * memWidth, memWidth * copyLine * sizeof(int), cudaMemcpyHostToDevice, *pStream));
		}
		loop_3_stream << < grid, block, 0, *pStream >> > (param->devMatDst, param->devMatSrc, width, height, memWidth, memHeight, offsetY);
		CHECK(cudaMemcpyAsync(param->hostMatDst + offsetY * memWidth, param->devMatDst + offsetY * memWidth, memWidth * copyLine * sizeof(int), cudaMemcpyDeviceToHost, *pStream));
	}

	for (int i = 0; i < NUM_STREAM; i++) {
		cudaStream_t* pStream = (cudaStream_t*)(param->pStream[i]);
		CHECK(cudaStreamSynchronize(*pStream));
	}

	swapMat(param);
	param->isMatrixUpdated = 0;

	// hostMatSrc is ready to be displayed
}

/* The algorithm using alias area on 4 corners and edges so that main logic doen't need to consider border
*  with shared memory
*  1. copy from host to device
*  2. repeat main logic several times
*  3. copy from device to host
*  Note: matrix memory size is (1 + width + 1, 1 + height + 1) (the real matrix is [(1,1) - (memWidth - 2, memHeight - 2)]
*/
void process_3_repeat(ALGORITHM_CUDA_NORMAL_PARAM *param, int width, int height, int repeatNum)
{
	int memWidth = width + 2 * MEMORY_MARGIN;
	int memHeight = height + 2 * MEMORY_MARGIN;

	/* block size setting for main logic
	* do copy per 32(BLOCK_SIZE)
	* do calculation per 30(BLOCK_SIZE-2)
	* the number of block is ceil(width / 30)
	*/
	dim3 block(BLOCK_SIZE_W, BLOCK_SIZE_H, 1);
	dim3 grid((int)ceil(width / (double)(BLOCK_SIZE_W - 2 * MEMORY_MARGIN)), (int)ceil(height / (double)(BLOCK_SIZE_H - 2 * MEMORY_MARGIN)), 1);


	/* Copy matrix data from host to device */
	if (param->isMatrixUpdated) {
		CHECK(cudaMemcpy(param->devMatSrc, param->hostMatSrc, memWidth * memHeight * sizeof(int), cudaMemcpyHostToDevice));
	}

	for (int i = 0; i < repeatNum; i++) {
		/* create alias area in device memory */
		dim3 blockW(BLOCK_SIZE_W);
		dim3 gridW(width / BLOCK_SIZE_W);
		dim3 blockH(BLOCK_SIZE_H);
		dim3 gridH(height / BLOCK_SIZE_H);
		loop_3_stream_makeAliasRow << <gridW, blockW >> > (param->devMatSrc, width, height, memWidth, memHeight);
		loop_3_stream_makeAliasCol << <gridH, blockH >> > (param->devMatSrc, width, height, memWidth, memHeight);
		CHECK(cudaDeviceSynchronize());

		/*** operate logic without border check ***/
		loop_3_stream << < grid, block >> > (param->devMatDst, param->devMatSrc, width, height, memWidth, memHeight, 0);
		CHECK(cudaDeviceSynchronize());
		swapMat(param);
	}
	swapMat(param);
	CHECK(cudaMemcpy(param->hostMatDst + (memWidth * 1) + MEMORY_MARGIN, param->devMatDst + (memWidth * 1) + MEMORY_MARGIN, memWidth * height * sizeof(int), cudaMemcpyDeviceToHost));
	swapMat(param);
	param->isMatrixUpdated = 0;

	// hostMatSrc is ready to be displayed
}


}
