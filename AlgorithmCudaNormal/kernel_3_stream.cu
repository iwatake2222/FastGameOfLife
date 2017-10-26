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
when block size is (32,32), calculation block is (30,30)
the 1st block calculates (1,1) - (30,30)  (using matrix[(0,0) - (31,31)])
the 2nd block calculates (31,1) - (60,30) (using matrix[(30,0) - (61,0)])
*/
__global__ void loop_3_stream(int* matDst, int *matSrc, int width, int height, int devMatWidth, int devMatHeight, int offsetY)
{
	__shared__ int tile[BLOCK_SIZE_H][BLOCK_SIZE_W];
	/* this is position on memory */
	int globalX = blockIdx.x * (blockDim.x - 2 * MEMORY_MARGIN) + threadIdx.x;	// <- increase 30(block size - 2) per block
	int globalY = blockIdx.y * (blockDim.y - 2 * MEMORY_MARGIN) + threadIdx.y;	// <- increase 30(block size - 2) per block
	globalY += offsetY;
	int localX = threadIdx.x;
	int localY = threadIdx.y;

	if (globalX >= devMatWidth || globalY >= devMatHeight) return;

	/* copy data from global memory to shared memory */
	int thisCell = tile[localY][localX] = matSrc[devMatWidth * globalY + globalX];
	__syncthreads();

	if (globalX >= devMatWidth - 1 || globalY >= devMatHeight - 1 || localX == 0 || localX == blockDim.x - 1 || localY == 0 || localY == blockDim.y - 1) return;

	int cnt;
	cnt = (tile[localY - 1][localX - 1] != 0) + (tile[localY - 1][localX - 0] != 0) + (tile[localY - 1][localX + 1] != 0)
		+ (tile[localY - 0][localX - 1] != 0) + (thisCell != 0) + (tile[localY - 0][localX + 1] != 0)
		+ (tile[localY + 1][localX - 1] != 0) + (tile[localY + 1][localX - 0] != 0) + (tile[localY + 1][localX + 1] != 0);

	updateCell(matDst, matSrc, devMatWidth * globalY + globalX, cnt);
}

__global__ void loop_3_stream_makeAliasRow(int *mat, int width, int height, int devMatWidth, int devMatHeight)
{
	int *p = mat;
	int devMatX = blockIdx.x * blockDim.x + threadIdx.x + 1;
	p[0 + devMatX] = p[devMatWidth * (devMatHeight - 2) + devMatX];
	p[devMatWidth * (devMatHeight - 1) + devMatX] = p[devMatWidth * 1 + devMatX];
	
	if (devMatX == 1) {
		p[devMatWidth * 0 + 0] = p[devMatWidth * (devMatHeight - 2) + devMatWidth - 2];
		p[devMatWidth * 0 + devMatWidth - 1] = p[devMatWidth * (devMatHeight - 2) + 1];
		p[devMatWidth * (devMatHeight - 1) + 0] = p[devMatWidth * (1) + devMatWidth - 2];
		p[devMatWidth * (devMatHeight - 1) + devMatWidth - 1] = p[devMatWidth * (1) + 1];
	}
}

__global__ void loop_3_stream_makeAliasCol(int *mat, int width, int height, int devMatWidth, int devMatHeight)
{
	int *p = mat;
	int devMatY = blockIdx.x * blockDim.x + threadIdx.x + 1;

	p[devMatWidth * devMatY + 0] = p[devMatWidth * devMatY + devMatWidth - 2];
	p[devMatWidth * devMatY + devMatWidth - 1] = p[devMatWidth * devMatY + 1];
	
}

void process_3_stream(ALGORITHM_CUDA_NORMAL_PARAM *param, int width, int height)
{
	int devMatWidth = width + 2 * MEMORY_MARGIN;
	int devMatHeight = height + 2 * MEMORY_MARGIN;
	int blocksizeW = BLOCK_SIZE_W;
	int blocksizeH = BLOCK_SIZE_H;

	/* block size setting for main logic
	* do copy per 32(BLOCK_SIZE)
	* do calculation per 30(BLOCK_SIZE-2)
	*/
	dim3 block(blocksizeW, blocksizeH, 1);
	dim3 grid((int)ceil(width / (double)(blocksizeW - 2 * MEMORY_MARGIN)), (int)ceil(height / (double)(blocksizeH - 2 * MEMORY_MARGIN) / NUM_STREAM), 1);

	/* block size settings for alias copy kernel */
	dim3 blockH(blocksizeH);
	dim3 gridH(height / blocksizeH);
	dim3 blockW(blocksizeW);
	dim3 gridW(width / blocksizeW);

	if (param->isFirstOperation) {
		/* create alias area in CPU */
		int *p = param->hostMatSrc;
		memcpy(p, p + (devMatHeight - 2) * devMatWidth, devMatWidth * sizeof(int));
		memcpy(p + (devMatHeight - 1) * devMatWidth, p + (1) * devMatWidth, devMatWidth * sizeof(int));
		for (int y = 1; y < devMatHeight - 1; y++) {
			p[devMatWidth * y + 0] = p[devMatWidth * y + devMatWidth - 2];
			p[devMatWidth * y + devMatWidth - 1] = p[devMatWidth * y + 1];
		}
		p[devMatWidth * 0 + 0] = p[devMatWidth * (devMatHeight - 2) + devMatWidth - 2];
		p[devMatWidth * 0 + devMatWidth - 1] = p[devMatWidth * (devMatHeight - 2) + 1];
		p[devMatWidth * (devMatHeight - 1) + 0] = p[devMatWidth * (1) + devMatWidth - 2];
		p[devMatWidth * (devMatHeight - 1) + devMatWidth - 1] = p[devMatWidth * (1) + 1];
	} else {
		/* create alias area in device memory */
		dim3 blockW(blocksizeW);
		dim3 gridW(width / blocksizeW);
		dim3 blockH(blocksizeH);
		dim3 gridH(height / blocksizeH);
		loop_3_stream_makeAliasRow <<<gridW, blockW >>> (param->devMatSrc, width, height, devMatWidth, devMatHeight);
		loop_3_stream_makeAliasCol << <gridH, blockH >> > (param->devMatSrc, width, height, devMatWidth, devMatHeight);
		CHECK(cudaDeviceSynchronize());
	}

	/* copy border line data at first */
	int heightStream = ceil((double)devMatHeight / NUM_STREAM);
	if (param->isFirstOperation) {
		for (int i = 0; i < NUM_STREAM; i++) {
			int offsetFirstLine = (i * heightStream) * devMatWidth;
			CHECK(cudaMemcpy(param->devMatSrc + offsetFirstLine, param->hostMatSrc + offsetFirstLine, devMatWidth * sizeof(int), cudaMemcpyHostToDevice));
			int offsetLastLine = ((i + 1) * heightStream - 1) * devMatWidth;
			if (offsetLastLine < (devMatHeight - 1) * devMatWidth) {
				CHECK(cudaMemcpy(param->devMatSrc + offsetLastLine, param->hostMatSrc + offsetLastLine, devMatWidth * sizeof(int), cudaMemcpyHostToDevice));
			} else {
				CHECK(cudaMemcpy(param->devMatSrc + (devMatHeight - 1) * devMatWidth, param->hostMatSrc + (devMatHeight - 1) * devMatWidth, devMatWidth * sizeof(int), cudaMemcpyHostToDevice));
			}
		}
	}

	/* create stream(copy(h2d), kernel, copy(d2h)) */
	for (int i = 0; i < NUM_STREAM; i++) {
		cudaStream_t* pStream = (cudaStream_t*)(param->pStream[i]);
		int offsetY = i * heightStream;
		int copyLine = heightStream;
		if (offsetY + heightStream > devMatHeight) copyLine = devMatHeight - offsetY;
		if (param->isFirstOperation) {
			CHECK(cudaMemcpyAsync(param->devMatSrc + offsetY * devMatWidth, param->hostMatSrc + offsetY * devMatWidth, devMatWidth * copyLine * sizeof(int), cudaMemcpyHostToDevice, *pStream));
		}
		loop_3_stream << < grid, block, 0, *pStream >> > (param->devMatDst, param->devMatSrc, width, height, devMatWidth, devMatHeight, offsetY);
		CHECK(cudaMemcpyAsync(param->hostMatDst + offsetY * devMatWidth, param->devMatDst + offsetY * devMatWidth, devMatWidth * copyLine * sizeof(int), cudaMemcpyDeviceToHost, *pStream));
	}

	for (int i = 0; i < NUM_STREAM; i++) {
		cudaStream_t* pStream = (cudaStream_t*)(param->pStream[i]);
		CHECK(cudaStreamSynchronize(*pStream));
	}

	swapMat(param);
	param->isFirstOperation = 0;
}


}
