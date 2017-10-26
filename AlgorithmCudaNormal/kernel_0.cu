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

__global__ void loop_0(int* matDst, int *matSrc, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	//if (x >= width || y >= height) {
	//	printf("%d %d\n", x, y);
	//	return;
	//}

	register int cnt = 0;
	for (int yy = y - 1; yy <= y + 1; yy++) {
		int roundY = yy;
		if (roundY >= height) roundY = 0;
		if (roundY < 0) roundY = height - 1;
		for (int xx = x - 1; xx <= x + 1; xx++) {
			int roundX = xx;
			if (roundX >= width) roundX = 0;
			if (roundX < 0) roundX = width - 1;
			if (matSrc[width * roundY + roundX] != 0) {
				cnt++;
			}
		}
	}
	updateCell(matDst, matSrc, y * width + x, cnt);
}

/* use global memory. always copy from host to device */
void process_0(ALGORITHM_CUDA_NORMAL_PARAM *param, int width, int height)
{
	int blocksizeW = BLOCK_SIZE_W;
	int blocksizeH = BLOCK_SIZE_H;
	dim3 block(blocksizeW, blocksizeH);
	dim3 grid(width / blocksizeW, height / blocksizeH);

	CHECK(cudaMemcpy(param->devMatSrc, param->hostMatSrc, width * height * sizeof(int), cudaMemcpyHostToDevice));

	loop_0 <<< grid, block >>> (param->devMatDst, param->devMatSrc, width, height);
	CHECK(cudaDeviceSynchronize());

	CHECK(cudaMemcpy(param->hostMatDst, param->devMatDst, width * height * sizeof(int), cudaMemcpyDeviceToHost));

	swapMat(param);
}

/* use global memory. copy from host to device only at the first time */
void process_0_nocopy(ALGORITHM_CUDA_NORMAL_PARAM *param, int* matDst, int* matSrc, int width, int height, int genTimes)
{
	//if (param->isFirstOperation != 0) {
	//	/* after the 2nd time, devMatSrc is copied from devMatDst */
	//	CHECK(cudaMemcpy(param->devMatSrc, matSrc, width * height * sizeof(int), cudaMemcpyHostToDevice));
	//	param->isFirstOperation = 0;
	//}

	//int blocksizeW = BLOCK_SIZE_W;
	//int blocksizeH = BLOCK_SIZE_H;
	//dim3 block(blocksizeW, blocksizeH);
	//dim3 grid(width / blocksizeW, height / blocksizeH);
	//
	//for (int gen = 0; gen < genTimes; gen++) {
	//	loop_0 <<< grid, block >>> (param->devMatDst, param->devMatSrc, width, height);
	//	CHECK(cudaDeviceSynchronize());
	//	int *temp = param->devMatSrc;
	//	param->devMatSrc = param->devMatDst;
	//	param->devMatDst = temp;
	//}

	//CHECK(cudaMemcpy(matDst, param->devMatSrc, width * height * sizeof(int), cudaMemcpyDeviceToHost));
}

}
