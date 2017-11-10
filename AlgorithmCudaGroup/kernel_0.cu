#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"
#include "stdlib.h"
#include <string.h>
#include "algorithmCudaGroup.h"
#include "algorithmCudaGroupInternal.h"

namespace AlgorithmCudaGroup
{
#if 0
}	// indent guard
#endif

__forceinline__ __device__ void updateCell(DNA* matDst, DNA* matSrc, int globalIndex, int cnt, int group)
{
	if (matSrc[globalIndex].age == 0) {
		if (cnt == 3) {
			// birth
			matDst[globalIndex].age = 1;
			matDst[globalIndex].group = group / 3;
		} else {
			// keep dead
			matDst[globalIndex].age = 0;
		}
	} else {
		if (cnt <= 2 || cnt >= 5) {
			// die
			matDst[globalIndex].age = 0;
		} else {
			// keep alive (age++)
			matDst[globalIndex].age = matSrc[globalIndex].age + 1;
			matDst[globalIndex].group = matSrc[globalIndex].group;
		}
	}
}

__global__ void loop_0(DNA* matDst, DNA *matSrc, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	//if (x >= width || y >= height) {
	//	printf("%d %d\n", x, y);
	//	return;
	//}

	register int cnt = 0;
	register int group = 0;
	for (int yy = y - 1; yy <= y + 1; yy++) {
		int roundY = yy;
		if (roundY >= height) roundY = 0;
		if (roundY < 0) roundY = height - 1;
		for (int xx = x - 1; xx <= x + 1; xx++) {
			int roundX = xx;
			if (roundX >= width) roundX = 0;
			if (roundX < 0) roundX = width - 1;
			if (matSrc[width * roundY + roundX].age != 0) {
				cnt++;
				group += matSrc[width * roundY + roundX].group;
			}
		}
	}
	updateCell(matDst, matSrc, y * width + x, cnt, group);
}

/* The most basic algorithm
*/
void process_0(ALGORITHM_CUDA_GROUP_PARAM *param, int width, int height, int repeatNum = 1)
{
	dim3 block(BLOCK_SIZE_W, BLOCK_SIZE_H);
	dim3 grid(width / BLOCK_SIZE_W, height / BLOCK_SIZE_H);

	if (param->isMatrixUpdated) {
		CHECK(cudaMemcpy(param->devMatSrc, param->hostMatSrc, width * height * sizeof(DNA), cudaMemcpyHostToDevice));
	}

	for (int i = 0; i < repeatNum; i++) {
		loop_0 << < grid, block >> > (param->devMatDst, param->devMatSrc, width, height);
		CHECK(cudaDeviceSynchronize());
		swapMat(param);
	}

	CHECK(cudaMemcpy(param->hostMatSrc, param->devMatSrc, width * height * sizeof(DNA), cudaMemcpyDeviceToHost));

	param->isMatrixUpdated = 0;

	// hostMatSrc is ready to be displayed
}


}
