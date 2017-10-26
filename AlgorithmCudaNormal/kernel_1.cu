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

__global__ void loop_1_withoutBorderCheck(int* matDst, int *matSrc, int width, int height, int devMatWidth, int devMatHeight)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int cnt = 0;
	for (int yy = y; yy <= y + 2; yy++) {
		int yIndex = devMatWidth * yy;
		for (int xx = x; xx <= x + 2; xx++) {
			if (matSrc[yIndex + xx] != 0) {
				cnt++;
			}
		}
	}

	updateCell(matDst, matSrc, devMatWidth * (y + 1) + (x + 1), cnt);
}


__global__ void copyAliasRow(int* devMat, int devMatWidth, int devMatHeight)
{
	int devMatX = blockIdx.x * blockDim.x + threadIdx.x + 1;
	devMat[devMatWidth * 0 + devMatX] = devMat[devMatWidth * (devMatHeight - 2) + devMatX];
	devMat[devMatWidth * (devMatHeight - 1) + devMatX] = devMat[devMatWidth * 1 + devMatX];
}

__global__ void copyAliasCol(int* devMat, int devMatWidth, int devMatHeight)
{
	int devMatY = blockIdx.x * blockDim.x + threadIdx.x + 1;
	devMat[devMatWidth * devMatY + 0] = devMat[devMatWidth * devMatY + (devMatWidth - 2)];
	devMat[devMatWidth * devMatY + devMatWidth - 1] = devMat[devMatWidth * devMatY + 1];
}

/* use global memory. always copy from host to device */
void process_1(ALGORITHM_CUDA_NORMAL_PARAM *param, int width, int height)
{
	int devMatWidth =  width + 2 * MEMORY_MARGIN;
	int devMatHeight =  height + 2 * MEMORY_MARGIN;
	int blocksizeW = BLOCK_SIZE_W;
	int blocksizeH = BLOCK_SIZE_H;
	dim3 block(blocksizeW, blocksizeH, 1);
	dim3 grid(width / blocksizeW, height / blocksizeH, 1);

	dim3 blockH(blocksizeH);
	dim3 gridH(height / blocksizeH);

	dim3 blockW(blocksizeW);
	dim3 gridW(width / blocksizeW);

	/*** Copy host memory to device global memory***/
#if 1
	/* 1. create alias area in CPU at first*/
	int *p = param->hostMatSrc;
	memcpy(p, p + (devMatHeight - 2) * devMatWidth, devMatWidth * sizeof(int));
	memcpy(p + (devMatHeight - 1) * devMatWidth, p + (1) * devMatWidth, devMatWidth * sizeof(int));
	for (int y = 1; y < devMatHeight - 1; y++) {
		p[devMatWidth * y + 0] = p[devMatWidth * y + devMatWidth - 2];
		p[devMatWidth * y + devMatWidth - 1] = p[devMatWidth * y + 1];
	}
	p[devMatWidth * 0                  + 0]               = p[devMatWidth * (devMatHeight - 2) + devMatWidth - 2];
	p[devMatWidth * 0                  + devMatWidth - 1] = p[devMatWidth * (devMatHeight - 2) + 1];
	p[devMatWidth * (devMatHeight - 1) + 0]               = p[devMatWidth * (1)                + devMatWidth - 2];
	p[devMatWidth * (devMatHeight - 1) + devMatWidth - 1] = p[devMatWidth * (1)                + 1];

	/* 2. then, copy all the area */
	CHECK(cudaMemcpy(param->devMatSrc, param->hostMatSrc, devMatWidth * devMatHeight * sizeof(int), cudaMemcpyHostToDevice));
#else
	/* [ (0, 0) - (width-1), (height-1) ] -> [ (1, 1) - (devMatWidth-2), (devMatHeight-2) ] */
	CHECK(cudaMemcpy(param->devMatSrc + (devMatWidth * 1) + MEMORY_MARGIN, param->hostMatSrc + (devMatWidth * 1) + MEMORY_MARGIN, devMatWidth * height * sizeof(int), cudaMemcpyHostToDevice));
	
	/*** Create alias area ***/
	/* create alias lines in device global memory
	 * [(1, devMatHeight-2) - (devMatWidth-2, devMatHeight-2)] -> [(1, 0) - (devMatWidth-2, 0)]
	 * [(1, 1) - (devMatWidth-2, 1)] -> [(1, devMatHeight-1) - (devMatWidth-2, devMatHeight-1)]
	 */
	copyAliasRow << < gridW, blockW >> > (param->devMatSrc, devMatWidth, devMatHeight);

	/* create alias columns in device global memory
	* [(devMatWidth-2, 1) - (devMatWidth-2, devMatHeight-2)] -> [(0, 1) - (0, devMatHeight-2)]
	* [(1, 1) - (1, devMatHeight-2)] -> [(devMatWidth-1, 1) - (devMatWidth-1, devMatHeight-2)]
	*/
	copyAliasCol << < gridH, blockH >> > (param->devMatSrc, devMatWidth, devMatHeight);
	//CHECK(cudaDeviceSynchronize());

	/* create alias dots for four corners in device global memory */
	CHECK(cudaMemcpy(param->devMatSrc + devMatWidth * (0) + 0, param->devMatSrc + devMatWidth * (devMatHeight - 2) + devMatWidth - 2, 1 * sizeof(int), cudaMemcpyDeviceToDevice));
	CHECK(cudaMemcpy(param->devMatSrc + devMatWidth * (0) + devMatWidth - 1, param->devMatSrc + devMatWidth * (devMatHeight - 2) + 1, 1 * sizeof(int), cudaMemcpyDeviceToDevice));
	CHECK(cudaMemcpy(param->devMatSrc + devMatWidth * (devMatHeight - 1) + 0, param->devMatSrc + devMatWidth * (1) + devMatWidth - 2, 1 * sizeof(int), cudaMemcpyDeviceToDevice));
	CHECK(cudaMemcpy(param->devMatSrc + devMatWidth * (devMatHeight - 1) + devMatWidth - 1, param->devMatSrc + devMatWidth * (1) + 1, 1 * sizeof(int), cudaMemcpyDeviceToDevice));
#endif

	/*** operate logic without border check ***/
	loop_1_withoutBorderCheck << < grid, block >> > (param->devMatDst, param->devMatSrc, width, height, devMatWidth, devMatHeight);
	CHECK(cudaDeviceSynchronize());
	
	CHECK(cudaMemcpy(param->hostMatDst + (devMatWidth * 1) + MEMORY_MARGIN, param->devMatDst + (devMatWidth * 1) + MEMORY_MARGIN, devMatWidth * height * sizeof(int), cudaMemcpyDeviceToHost));

	swapMat(param);
}


}
