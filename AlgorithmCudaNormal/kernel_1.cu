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

__global__ void loop_1(int* matDst, int *matSrc, int width, int height, int offsetY)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	y += offsetY;

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


/* Divide area into several landscape area, and each stream processes each area */
void process_1(ALGORITHM_CUDA_NORMAL_PARAM *param, int width, int height)
{
	int blocksizeW = BLOCK_SIZE_W;
	int blocksizeH = BLOCK_SIZE_H;
	dim3 block(blocksizeW, blocksizeH);
	dim3 grid(width / blocksizeW, height / blocksizeH / NUM_STREAM);

	// copy border line data at first
	for (int i = 0; i < NUM_STREAM; i++) {
		int offsetFirstLine = (i * height / NUM_STREAM) * width;
		CHECK(cudaMemcpy(param->devMatSrc + offsetFirstLine, param->hostMatSrc + offsetFirstLine, width * sizeof(int), cudaMemcpyHostToDevice));
		int offsetLastLine = ((i + 1) * height / NUM_STREAM - 1) * width;
		CHECK(cudaMemcpy(param->devMatSrc + offsetLastLine, param->hostMatSrc + offsetLastLine, width * sizeof(int), cudaMemcpyHostToDevice));
	}

	for (int i = 0; i < NUM_STREAM; i++) {
		cudaStream_t* pStream = (cudaStream_t*)(param->pStream[i]);
		int offsetY = i * height / NUM_STREAM;
		int transferHeight = height / NUM_STREAM;
		CHECK(cudaMemcpyAsync(param->devMatSrc + offsetY * width, param->hostMatSrc + offsetY * width, width * transferHeight * sizeof(int), cudaMemcpyHostToDevice, *pStream));
		loop_1 << < grid, block, 0, *pStream >> > (param->devMatDst, param->devMatSrc, width, height, offsetY);
		CHECK(cudaMemcpyAsync(param->hostMatDst + offsetY * width, param->devMatDst + offsetY * width, width * transferHeight * sizeof(int), cudaMemcpyDeviceToHost, *pStream));
	}

	for (int i = 0; i < NUM_STREAM; i++) {
		cudaStream_t* pStream = (cudaStream_t*)(param->pStream[i]);
		CHECK(cudaStreamSynchronize(*pStream));
	}

	swapMat(param);
}


}
