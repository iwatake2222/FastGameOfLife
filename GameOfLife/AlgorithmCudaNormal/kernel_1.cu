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

__global__ void loop_1_withoutBorderCheck(int* matDst, int *matSrc, int width, int height, int memWidth, int memHeight)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int cnt = 0;
	for (int yy = y; yy <= y + 2; yy++) {
		int yIndex = memWidth * yy;
		for (int xx = x; xx <= x + 2; xx++) {
			if (matSrc[yIndex + xx] != 0) {
				cnt++;
			}
		}
	}

	updateCell(matDst, matSrc, memWidth * (y + 1) + (x + 1), cnt);
}


__global__ void copyAliasRow(int* devMat, int memWidth, int memHeight)
{
	int devMatX = blockIdx.x * blockDim.x + threadIdx.x + 1;
	devMat[memWidth * 0 + devMatX] = devMat[memWidth * (memHeight - 2) + devMatX];
	devMat[memWidth * (memHeight - 1) + devMatX] = devMat[memWidth * 1 + devMatX];
}

__global__ void copyAliasCol(int* devMat, int memWidth, int memHeight)
{
	int devMatY = blockIdx.x * blockDim.x + threadIdx.x + 1;
	devMat[memWidth * devMatY + 0] = devMat[memWidth * devMatY + (memWidth - 2)];
	devMat[memWidth * devMatY + memWidth - 1] = devMat[memWidth * devMatY + 1];
}

/* The algorithm using alias area on 4 corners and edges so that main logic doen't need to consider border
 *  Note: matrix memory size is (1 + width + 1, 1 + height + 1) (the real matrix is [(1,1) - (memWidth - 2, memHeight - 2)]
*/
void process_1(ALGORITHM_CUDA_NORMAL_PARAM *param, int width, int height)
{
	int memWidth =  width + 2 * MEMORY_MARGIN;
	int memHeight =  height + 2 * MEMORY_MARGIN;
	dim3 block(BLOCK_SIZE_W, BLOCK_SIZE_H, 1);
	dim3 grid(width / BLOCK_SIZE_W, height / BLOCK_SIZE_H, 1);

	dim3 blockH(BLOCK_SIZE_H);
	dim3 gridH(height / BLOCK_SIZE_H);

	dim3 blockW(BLOCK_SIZE_W);
	dim3 gridW(width / BLOCK_SIZE_W);

	/*** Copy host memory to device global memory, and create alia area ***/
#if 1
	/* Create alias area in CPU at first, then copy all the memory area from host to device */
	int *p = param->hostMatSrc;
	memcpy(p, p + (memHeight - 2) * memWidth, memWidth * sizeof(int));
	memcpy(p + (memHeight - 1) * memWidth, p + (1) * memWidth, memWidth * sizeof(int));
	for (int y = 1; y < memHeight - 1; y++) {
		p[memWidth * y + 0] = p[memWidth * y + memWidth - 2];
		p[memWidth * y + memWidth - 1] = p[memWidth * y + 1];
	}
	p[memWidth * 0                  + 0]               = p[memWidth * (memHeight - 2) + memWidth - 2];
	p[memWidth * 0                  + memWidth - 1] = p[memWidth * (memHeight - 2) + 1];
	p[memWidth * (memHeight - 1) + 0]               = p[memWidth * (1)                + memWidth - 2];
	p[memWidth * (memHeight - 1) + memWidth - 1] = p[memWidth * (1)                + 1];

#if !defined(USE_ZEROCOPY_MEMORY)
	CHECK(cudaMemcpy(param->devMatSrc, param->hostMatSrc, memWidth * memHeight * sizeof(int), cudaMemcpyHostToDevice));
#endif
#else
	/* Copy world area from host to device, then create alias area in device memory */
	/* [ (0, 0) - (width-1), (height-1) ] -> [ (1, 1) - (memWidth-2), (memHeight-2) ] */
	CHECK(cudaMemcpy(param->devMatSrc + (memWidth * 1) + MEMORY_MARGIN, param->hostMatSrc + (memWidth * 1) + MEMORY_MARGIN, memWidth * height * sizeof(int), cudaMemcpyHostToDevice));
	
	/* create alias lines in device global memory
	 * [(1, memHeight-2) - (memWidth-2, memHeight-2)] -> [(1, 0) - (memWidth-2, 0)]
	 * [(1, 1) - (memWidth-2, 1)] -> [(1, memHeight-1) - (memWidth-2, memHeight-1)]
	 */
	copyAliasRow << < gridW, blockW >> > (param->devMatSrc, memWidth, memHeight);

	/* create alias columns in device global memory
	* [(memWidth-2, 1) - (memWidth-2, memHeight-2)] -> [(0, 1) - (0, memHeight-2)]
	* [(1, 1) - (1, memHeight-2)] -> [(memWidth-1, 1) - (memWidth-1, memHeight-2)]
	*/
	copyAliasCol << < gridH, blockH >> > (param->devMatSrc, memWidth, memHeight);
	//CHECK(cudaDeviceSynchronize());

	/* create alias dots for four corners in device global memory */
	CHECK(cudaMemcpy(param->devMatSrc + memWidth * (0) + 0, param->devMatSrc + memWidth * (memHeight - 2) + memWidth - 2, 1 * sizeof(int), cudaMemcpyDeviceToDevice));
	CHECK(cudaMemcpy(param->devMatSrc + memWidth * (0) + memWidth - 1, param->devMatSrc + memWidth * (memHeight - 2) + 1, 1 * sizeof(int), cudaMemcpyDeviceToDevice));
	CHECK(cudaMemcpy(param->devMatSrc + memWidth * (memHeight - 1) + 0, param->devMatSrc + memWidth * (1) + memWidth - 2, 1 * sizeof(int), cudaMemcpyDeviceToDevice));
	CHECK(cudaMemcpy(param->devMatSrc + memWidth * (memHeight - 1) + memWidth - 1, param->devMatSrc + memWidth * (1) + 1, 1 * sizeof(int), cudaMemcpyDeviceToDevice));
#endif

	/*** operate logic without border check ***/
	loop_1_withoutBorderCheck << < grid, block >> > (param->devMatDst, param->devMatSrc, width, height, memWidth, memHeight);
	CHECK(cudaDeviceSynchronize());

#if !defined(USE_ZEROCOPY_MEMORY)
	CHECK(cudaMemcpy(param->hostMatDst + (memWidth * 1) + MEMORY_MARGIN, param->devMatDst + (memWidth * 1) + MEMORY_MARGIN, memWidth * height * sizeof(int), cudaMemcpyDeviceToHost));
#endif
	swapMat(param);
	// hostMatSrc is ready to be displayed
}


}
