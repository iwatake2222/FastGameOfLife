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
__global__ void loop_2(int* matDst, int *matSrc, int width, int height, int devMatWidth, int devMatHeight)
{
	__shared__ int tile[BLOCK_SIZE_H][BLOCK_SIZE_W];
	/* this is position on memory */
	int globalX = blockIdx.x * (blockDim.x - 2 * MEMORY_MARGIN) + threadIdx.x;	// <- increase 30(block size - 2) per block
	int globalY = blockIdx.y * (blockDim.y - 2 * MEMORY_MARGIN) + threadIdx.y;	// <- increase 30(block size - 2) per block
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


__global__ void copyAliasRow2(int* devMat, int devMatWidth, int devMatHeight)
{
	int devMatX = blockIdx.x * blockDim.x + threadIdx.x + 1;
	devMat[devMatWidth * 0 + devMatX] = devMat[devMatWidth * (devMatHeight - 2) + devMatX];
	devMat[devMatWidth * (devMatHeight - 1) + devMatX] = devMat[devMatWidth * 1 + devMatX];
}

__global__ void copyAliasCol2(int* devMat, int devMatWidth, int devMatHeight)
{
	int devMatY = blockIdx.x * blockDim.x + threadIdx.x + 1;
	devMat[devMatWidth * devMatY + 0] = devMat[devMatWidth * devMatY + (devMatWidth - 2)];
	devMat[devMatWidth * devMatY + devMatWidth - 1] = devMat[devMatWidth * devMatY + 1];
}

/* use global memory. always copy from host to device */
void process_2(ALGORITHM_CUDA_NORMAL_PARAM *param, int width, int height)
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
	dim3 grid((int)ceil(width / (double)(blocksizeW - 2 * MEMORY_MARGIN)), (int)ceil(height / (double)(blocksizeH - 2 * MEMORY_MARGIN)), 1);

	/*  create alias area in CPU at first*/
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

	/* then, copy all the area */
	CHECK(cudaMemcpy(param->devMatSrc, param->hostMatSrc, devMatWidth * devMatHeight * sizeof(int), cudaMemcpyHostToDevice));

	/*** operate logic without border check ***/
	loop_2 << < grid, block >> > (param->devMatDst, param->devMatSrc, width, height, devMatWidth, devMatHeight);
	CHECK(cudaDeviceSynchronize());

	CHECK(cudaMemcpy(param->hostMatDst + (devMatWidth * 1) + MEMORY_MARGIN, param->devMatDst + (devMatWidth * 1) + MEMORY_MARGIN, devMatWidth * height * sizeof(int), cudaMemcpyDeviceToHost));

	swapMat(param);
}


}
