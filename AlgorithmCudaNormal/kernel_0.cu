#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"
#include "stdlib.h"
#include <string.h>
#include "algorithmCudaNormal.h"

#define CHECK(call)\
do {\
	const cudaError_t error = call;\
	if (error != cudaSuccess) {\
		printf("Error: %s:%d, ", __FILE__, __LINE__);\
		printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));\
		exit(1);\
	}\
} while(0)

namespace AlgorithmCudaNormal
{
#if 0
}	// indent guard
#endif

__global__ void loop_0(int* matDst, int *matSrc, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	//if (x >= width || y >= height) {
	//	printf("%d %d\n", x, y);
	//	return;
	//}

	int cnt = 0;
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

	int yLine = y * width;
	if (matSrc[yLine + x] == 0) {
		if (cnt == 3) {
			// birth
			matDst[yLine + x] = 1;
		} else {
			// keep dead
			matDst[yLine + x] = 0;
		}
	} else {
		if (cnt <= 2 || cnt >= 5) {
			// die
			matDst[yLine + x] = 0;
		} else {
			// keep alive (age++)
			matDst[yLine + x] = matSrc[yLine + x] + 1;
		}
	}
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
