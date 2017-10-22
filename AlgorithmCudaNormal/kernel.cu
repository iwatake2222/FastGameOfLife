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

__global__ void loop(int* matDst, int *matSrc, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	//printf("%d %d\n", x, y);

	if (x >= width || y >= height) {
		printf("%d %d\n", x, y);
		return;
	}

	int yLine = y * width;

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

void logicForOneGeneration(int* matDst, int* matSrc, int width, int height)
{
	//cudaStream_t stream;
	//cudaStreamCreate(&stream);
	//cudaStreamAttachMemAsync(stream, matSrc, 0, cudaMemAttachHost);
	//cudaStreamAttachMemAsync(stream, matDst, 0, cudaMemAttachSingle);
	////cudaDeviceSynchronize();
	//int blocksizeW = 16;
	//int blocksizeH = 16;
	//dim3 block(blocksizeW, blocksizeH);
	//dim3 grid(width / blocksizeW, height / blocksizeH);
	//loop <<<grid, block, 0, stream >>> (matDst, matSrc, width, height);
	//cudaStreamSynchronize(stream);
	//cudaStreamDestroy(stream);
	//cudaDeviceSynchronize();
	////cudaDeviceReset();

	int blocksizeW = 16;
	int blocksizeH = 16;
	dim3 block(blocksizeW, blocksizeH);
	dim3 grid(width / blocksizeW, height / blocksizeH);
	loop << <grid, block >> > (matDst, matSrc, width, height);

	cudaDeviceSynchronize();
	//cudaDeviceReset();

}




}
