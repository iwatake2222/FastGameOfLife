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

__global__ void loop_0_stream(int* matDst, int *matSrc, int width, int height, int offsetY)
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
	updateCell(matDst, matSrc, y * width + x, cnt);
}

/* The most basic algorithm
 *   with stream (divide area into several landscape area, and each stream processes each area)
 */
void process_0_stream(ALGORITHM_CUDA_NORMAL_PARAM *param, int width, int height)
{
	dim3 block(BLOCK_SIZE_W, BLOCK_SIZE_H);
	dim3 grid(width / BLOCK_SIZE_W, height / BLOCK_SIZE_H / NUM_STREAM);
	int heightStream = ceil((double)height / NUM_STREAM);

	/* copy border line data at first to simplyfy logic of each stream (no need to consider border line) */
	for (int i = 0; i < NUM_STREAM; i++) {
		int offsetFirstLine = (i * height / NUM_STREAM) * width;
		CHECK(cudaMemcpy(param->devMatSrc + offsetFirstLine, param->hostMatSrc + offsetFirstLine, width * sizeof(int), cudaMemcpyHostToDevice));
		int offsetLastLine = ((i + 1) * height / NUM_STREAM - 1) * width;
		CHECK(cudaMemcpy(param->devMatSrc + offsetLastLine, param->hostMatSrc + offsetLastLine, width * sizeof(int), cudaMemcpyHostToDevice));
	}

	/* create stream(copy(h2d), kernel, copy(d2h)) */
	for (int i = 0; i < NUM_STREAM; i++) {
		cudaStream_t* pStream = (cudaStream_t*)(param->pStream[i]);
		int offsetY = i * heightStream;
		CHECK(cudaMemcpyAsync(param->devMatSrc + offsetY * width, param->hostMatSrc + offsetY * width, width * heightStream * sizeof(int), cudaMemcpyHostToDevice, *pStream));
		loop_0_stream << < grid, block, 0, *pStream >> > (param->devMatDst, param->devMatSrc, width, height, offsetY);
		CHECK(cudaMemcpyAsync(param->hostMatDst + offsetY * width, param->devMatDst + offsetY * width, width * heightStream * sizeof(int), cudaMemcpyDeviceToHost, *pStream));
	}

	for (int i = 0; i < NUM_STREAM; i++) {
		cudaStream_t* pStream = (cudaStream_t*)(param->pStream[i]);
		CHECK(cudaStreamSynchronize(*pStream));
	}

	swapMat(param);
	// hostMatSrc is ready to be displayed
}


}
