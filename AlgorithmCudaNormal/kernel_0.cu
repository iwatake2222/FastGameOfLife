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

/* The most basic algorithm
 */
void process_0(ALGORITHM_CUDA_NORMAL_PARAM *param, int width, int height)
{
	dim3 block(BLOCK_SIZE_W, BLOCK_SIZE_H);
	dim3 grid(width / BLOCK_SIZE_W, height / BLOCK_SIZE_H);

	CHECK(cudaMemcpy(param->devMatSrc, param->hostMatSrc, width * height * sizeof(int), cudaMemcpyHostToDevice));

	loop_0 <<< grid, block >>> (param->devMatDst, param->devMatSrc, width, height);
	CHECK(cudaDeviceSynchronize());

	CHECK(cudaMemcpy(param->hostMatDst, param->devMatDst, width * height * sizeof(int), cudaMemcpyDeviceToHost));

	swapMat(param);

	// hostMatSrc is ready to be displayed
}


}
