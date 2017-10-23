#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"
#include "stdlib.h"
#include <string.h>
#include "algorithmCudaNormal.h"

namespace AlgorithmCudaNormal
{
#if 0
}	// indent guard
#endif


__global__ void loop(int* matDst, int *matSrc, int width, int height)
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

#if 0
/* always copy from host to device */
void logicForOneGeneration(ALGORITHM_CUDA_NORMAL_PARAM *param, int* matDst, int* matSrc, int width, int height)
{
	cudaMemcpy(param->devMatSrc, matSrc, width * height * sizeof(int), cudaMemcpyHostToDevice);

	int blocksizeW = 16;
	int blocksizeH = 16;
	dim3 block(blocksizeW, blocksizeH);
	dim3 grid(width / blocksizeW, height / blocksizeH);
	loop <<<grid, block >>> (param->devMatDst, param->devMatSrc, width, height);
	cudaDeviceSynchronize();

	cudaMemcpy(matDst, param->devMatDst, width * height * sizeof(int), cudaMemcpyDeviceToHost);
}

#else

/* copy from host to device only at the first time */
void logicForOneGeneration(ALGORITHM_CUDA_NORMAL_PARAM *param, int* matDst, int* matSrc, int width, int height)
{
	if (param->isFirstOperation != 0) {
		/* after the 2nd time, devMatSrc is copied from devMatDst */
		cudaMemcpy(param->devMatSrc, matSrc, width * height * sizeof(int), cudaMemcpyHostToDevice);
		param->isFirstOperation = 0;
	}

	int blocksizeW = 16;
	int blocksizeH = 16;
	dim3 block(blocksizeW, blocksizeH);
	dim3 grid(width / blocksizeW, height / blocksizeH);
	loop << <grid, block >> > (param->devMatDst, param->devMatSrc, width, height);
	cudaDeviceSynchronize();
	cudaMemcpy(param->devMatSrc, param->devMatDst, width * height * sizeof(int), cudaMemcpyDeviceToDevice);

	cudaMemcpy(matDst, param->devMatDst, width * height * sizeof(int), cudaMemcpyDeviceToHost);
}
#endif


void cudaInitialize(ALGORITHM_CUDA_NORMAL_PARAM *param, int width, int height)
{
	cudaMalloc((void**)&param->devMatSrc, width * height * sizeof(int));
	cudaMalloc((void**)&param->devMatDst, width * height * sizeof(int));
	param->isFirstOperation = 1;
}

void cudaFinalize(ALGORITHM_CUDA_NORMAL_PARAM *param)
{
	cudaFree(param->devMatSrc);
	cudaFree(param->devMatDst);
	cudaDeviceReset();
}


/*
 * Don't use cudaMallocManaged
 * Memory access exception occurs when I call logicForOneGeneration from several threads
 */
#if 0
void allocManaged(int **p, int size)
{
	cudaMallocManaged(p, size);
}

void freeManaged(int *p)
{
	cudaFree(p);
}

void cudaDeviceSynchronizeWrapper()
{
	cudaDeviceSynchronize();
}

void logicForOneGeneration(int* matDst, int* matSrc, int width, int height)
{
	int blocksizeW = 16;
	int blocksizeH = 16;
	dim3 block(blocksizeW, blocksizeH);
	dim3 grid(width / blocksizeW, height / blocksizeH);
	loop << <grid, block >> > (matDst, matSrc, width, height);
	cudaDeviceSynchronize();
}
#endif



}
