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

int NUM_STREAM = NUM_STREAM_MAX;

void cudaInitialize(ALGORITHM_CUDA_GROUP_PARAM *param, int width, int height)
{
	NUM_STREAM = height / BLOCK_SIZE_H;
	if (NUM_STREAM > NUM_STREAM_MAX) NUM_STREAM = NUM_STREAM_MAX;

	CHECK(cudaMalloc((void**)&param->devMatSrc, (width + 2 * MEMORY_MARGIN) * (height + 2 * MEMORY_MARGIN) * sizeof(DNA)));
	CHECK(cudaMalloc((void**)&param->devMatDst, (width + 2 * MEMORY_MARGIN) * (height + 2 * MEMORY_MARGIN) * sizeof(DNA)));

#if 1
	CHECK(cudaMallocHost((void**)&param->hostMatSrc, (width + 2 * MEMORY_MARGIN) * (height + 2 * MEMORY_MARGIN) * sizeof(DNA)));
	CHECK(cudaMallocHost((void**)&param->hostMatDst, (width + 2 * MEMORY_MARGIN) * (height + 2 * MEMORY_MARGIN) * sizeof(DNA)));
#else
	param->hostMatSrc = new DNA[(width + 2 * MEMORY_MARGIN) * (height + 2 * MEMORY_MARGIN)];
	param->hostMatDst = new DNA[(width + 2 * MEMORY_MARGIN) * (height + 2 * MEMORY_MARGIN)];
#endif

	param->isMatrixUpdated = 1;

	for (int i = 0; i < NUM_STREAM; i++) {
		cudaStream_t *stream;
		stream = new cudaStream_t;
		CHECK(cudaStreamCreate(stream));
		param->pStream[i] = (void*)stream;
	}
}

void cudaFinalize(ALGORITHM_CUDA_GROUP_PARAM *param)
{
	for (int i = 0; i < NUM_STREAM; i++) {
		cudaStream_t *stream = (cudaStream_t*)(param->pStream[i]);
		CHECK(cudaStreamDestroy(*stream));
		delete stream;
	}

	CHECK(cudaFree(param->devMatSrc));
	CHECK(cudaFree(param->devMatDst));

#if 1
	CHECK(cudaFreeHost(param->hostMatSrc));
	CHECK(cudaFreeHost(param->hostMatDst));
#else
	delete param->hostMatSrc;
	delete param->hostMatDst;
#endif

	/* todo: call this at the really end of the application */
	//CHECK(cudaDeviceReset());
}

void cudaProcess(ALGORITHM_CUDA_GROUP_PARAM *param, int width, int height, int repeatNum)
{
	for (int i = 0; i < repeatNum; i++) {
#if defined(ALGORITHM_CPU)
		extern void processWithBorderCheck(DNA *matDst, DNA *matSrc, int width, int height);
		processWithBorderCheck(param->hostMatDst, param->hostMatSrc, width, height);
		swapMat(param);
#elif defined(ALGORITHM_0)
		extern void process_0(ALGORITHM_CUDA_GROUP_PARAM *param, int width, int height, int repeatNum);
		process_0(param, width, height, repeatNum);
		break;
#endif
	}
}

void swapMat(ALGORITHM_CUDA_GROUP_PARAM *param)
{
	DNA *temp;
	temp = param->devMatDst;
	param->devMatDst = param->devMatSrc;
	param->devMatSrc = temp;

	temp = param->hostMatDst;
	param->hostMatDst = param->hostMatSrc;
	param->hostMatSrc = temp;
}


inline void updateCell(DNA* matDst, DNA *matSrc, int x, int yLine, int cnt, int group)
{
	/* Note: yLine is index of array (yLine = y*width) */
	if (matSrc[yLine + x].age == 0) {
		if (cnt == 3) {
			// birth
			matDst[yLine + x].age = 1;
			matDst[yLine + x].group = group / 3;
		} else {
			// keep dead
			matDst[yLine + x].age = 0;
		}
	} else {
		if (cnt <= 2 || cnt >= 5) {
			// die
			matDst[yLine + x].age = 0;
		} else {
			// keep alive (age++)
			matDst[yLine + x].age = matSrc[yLine + x].age + 1;
			matDst[yLine + x].group = matSrc[yLine + x].group;
		}
	}
}

void processWithBorderCheck(DNA *matDst, DNA *matSrc, int width, int height)
{
	for (int y = 0; y < height; y++) {
		int yLine = width * y;
		for (int x = 0; x < width; x++) {
			int cnt = 0;
			int group = 0;
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
			//if (x == 8 && y == 0) {
			//	printf("ref:\n");
			//	printf("cnt = %d\n", cnt);
			//	printf("%d %d %d\n", matSrc[(y - 1)*width + x - 1], matSrc[(y - 1)*width + x - 0], matSrc[(y - 1)*width + x + 1]);
			//	printf("%d %d %d\n", matSrc[(y - 0)*width + x - 1], matSrc[(y - 0)*width + x - 0], matSrc[(y - 0)*width + x + 1]);
			//	printf("%d %d %d\n", matSrc[(y + 1)*width + x - 1], matSrc[(y + 1)*width + x - 0], matSrc[(y + 1)*width + x + 1]);
			//}
			updateCell(matDst, matSrc, x, yLine, cnt, group);
		}
	}
}



}
