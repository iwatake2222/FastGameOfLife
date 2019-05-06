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

#if 0
__global__ void loop_2(int* matDst, int *matSrc, int width, int height)
{
	__shared__ int tile[BLOCK_SIZE_H][BLOCK_SIZE_W];
	int globalX = blockIdx.x * blockDim.x + threadIdx.x;
	int globalY = blockIdx.y * blockDim.y + threadIdx.y;

	/*** copy data from global memory to shared memory ***/
	/* copy the data located in the same position */
	int thisCell = tile[threadIdx.y][threadIdx.x] = matSrc[width * globalY + globalX];

	__syncthreads();

	int cnt = 0;
	for (int yy = (int)threadIdx.y - 1; yy <= (int)threadIdx.y + 1; yy++) {
		for (int xx = (int)threadIdx.x - 1; xx <= (int)threadIdx.x + 1; xx++) {
			if (xx >= 0 && xx < blockDim.x && yy >= 0 && yy < blockDim.y) {
				if (tile[yy][xx] != 0) cnt++;
			} else {
				int roundY = blockIdx.y * blockDim.y + yy;
				if (roundY >= height) roundY = 0;
				if (roundY < 0) roundY = height - 1;
				int roundX = blockIdx.x * blockDim.x + xx;
				if (roundX >= width) roundX = 0;
				if (roundX < 0) roundX = width - 1;
				if (matSrc[width * roundY + roundX] != 0) {
					cnt++;
				}
			}
		}
	}


	int targetIndex = globalY * width + globalX;
	if (thisCell == 0) {
		if (cnt == 3) {
			// birth
			matDst[targetIndex] = 1;
		} else {
			// keep dead
			matDst[targetIndex] = 0;
		}
	} else {
		if (cnt <= 2 || cnt >= 5) {
			// die
			matDst[targetIndex] = 0;
		} else {
			// keep alive (age++)
			matDst[targetIndex] = thisCell + 1;
		}
	}
}
#endif

#if 0
__global__ void loop_2(int* matDst, int *matSrc, int width, int height)
{
	__shared__ int tile[1 + BLOCK_SIZE_H + 1][1 + BLOCK_SIZE_W + 1];
	int globalX = blockIdx.x * blockDim.x + threadIdx.x;
	int globalY = blockIdx.y * blockDim.y + threadIdx.y;
	int localX = threadIdx.x + 1;
	int localY = threadIdx.y + 1;
	int globalXLeft = 999;
	int globalXRight = 999;
	int globalYTop = 999;
	int globalYBottom = 999;

	/*** copy data from global memory to shared memory ***/
	/* copy the data located in the same position */
	int thisCell = tile[localY][localX] = matSrc[width * globalY + globalX];

	/* copy neighborhood  */
	if (threadIdx.x == 0) {
		if (globalX != 0) {
			globalXLeft = globalX - 1;
		} else {
			globalXLeft = width - 1;
		}
	}
	if (threadIdx.x == blockDim.x - 1) {
		if (globalX != width - 1) {
			globalXRight = globalX + 1;
		} else {
			globalXRight = 0;
		}
	}
	if (threadIdx.y == 0) {
		if (globalY != 0) {
			globalYTop = globalY - 1;
		} else {
			globalYTop = height - 1;
		}
	}
	if (threadIdx.y == blockDim.y - 1) {
		if (globalY != height - 1) {
			globalYBottom = globalY + 1;
		} else {
			globalYBottom = 0;
		}
	}

	if (globalXLeft != 999) {
		tile[localY][localX - 1] = matSrc[width * globalY + globalXLeft];
		if (globalYTop != 999) {
			tile[localY - 1][localX - 1] = matSrc[width * globalYTop + globalXLeft];
		} else if (globalYBottom != 999) {
			tile[localY + 1][localX - 1] = matSrc[width * globalYBottom + globalXLeft];
		}
	} else	if (globalXRight != 999) {
		tile[localY][localX + 1] = matSrc[width * globalY + globalXRight];
		if (globalYTop != 999) {
			tile[localY - 1][localX + 1] = matSrc[width * globalYTop + globalXRight];
		} else if (globalYBottom != 999) {
			tile[localY + 1][localX + 1] = matSrc[width * globalYBottom + globalXRight];
		}
	}
	if (globalYTop != 999) {
		tile[localY - 1][localX] = matSrc[width * globalYTop + globalX];
	} else if (globalYBottom != 999) {
		tile[localY + 1][localX] = matSrc[width * globalYBottom + globalX];
	}

	__syncthreads();

	//if (globalX == 22 && globalY == 15) {
	//	printf("cuda local:\n");
	//	printf("%d %d %d\n", tile[localY - 1][localX - 1], tile[localY - 1][localX - 0], tile[localY - 1][localX + 1]);
	//	printf("%d %d %d\n", tile[localY - 0][localX - 1], tile[localY - 0][localX - 0], tile[localY - 0][localX + 1]);
	//	printf("%d %d %d\n", tile[localY + 1][localX - 1], tile[localY + 1][localX - 0], tile[localY + 1][localX + 1]);
	//}

	register int cnt = 0;
	for (int yy = localY - 1; yy <= localY + 1; yy++) {
		for (int xx = localX - 1; xx <= localX + 1; xx++) {
			if (tile[yy][xx] != 0) cnt++;
		}
	}

	int targetIndex = globalY * width + globalX;
	if (thisCell == 0) {
		if (cnt == 3) {
			// birth
			matDst[targetIndex] = 1;
		} else {
			// keep dead
			matDst[targetIndex] = 0;
		}
	} else {
		if (cnt <= 2 || cnt >= 5) {
			// die
			matDst[targetIndex] = 0;
		} else {
			// keep alive (age++)
			matDst[targetIndex] = thisCell + 1;
		}
	}
}
#endif
#if 0
void process_2(ALGORITHM_CUDA_NORMAL_PARAM *param, int width, int height)
{
	int blocksizeW = BLOCK_SIZE_W;
	int blocksizeH = BLOCK_SIZE_H;
	dim3 block(blocksizeW, blocksizeH);
	dim3 grid(width / blocksizeW, height / blocksizeH);

	CHECK(cudaMemcpy(param->devMatSrc, param->hostMatSrc, width * height * sizeof(int), cudaMemcpyHostToDevice));

	loop_2 <<< grid, block >>> (param->devMatDst, param->devMatSrc, width, height);
	CHECK(cudaDeviceSynchronize());

	CHECK(cudaMemcpy(param->hostMatDst, param->devMatDst, width * height * sizeof(int), cudaMemcpyDeviceToHost));

	swapMat(param);
}
#endif

}
