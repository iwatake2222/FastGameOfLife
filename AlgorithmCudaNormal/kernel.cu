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

void cudaInitialize(ALGORITHM_CUDA_NORMAL_PARAM *param, int width, int height)
{
	CHECK(cudaMalloc((void**)&param->devMatSrc, width * height * sizeof(int)));
	CHECK(cudaMalloc((void**)&param->devMatDst, width * height * sizeof(int)));
#if 1
	CHECK(cudaMallocHost((void**)&param->hostMatSrc, width * height * sizeof(int)));
	CHECK(cudaMallocHost((void**)&param->hostMatDst, width * height * sizeof(int)));
#else
	param->hostMatSrc = new int[width * height];
	param->hostMatDst = new int[width * height];
#endif
	param->isFirstOperation = 1;

	for (int i = 0; i < NUM_STREAM; i++) {
		cudaStream_t *stream;
		stream = new cudaStream_t;
		CHECK(cudaStreamCreate(stream));
		param->pStream[i] = (void*)stream;
	}
}

void cudaFinalize(ALGORITHM_CUDA_NORMAL_PARAM *param)
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
	CHECK(cudaDeviceReset());
}

void cudaProcess(ALGORITHM_CUDA_NORMAL_PARAM *param, int width, int height)
{
#if 0
	extern void process_0(ALGORITHM_CUDA_NORMAL_PARAM *param, int width, int height);
	process_0(param, width, height);
#endif
#if 1
	extern void process_1(ALGORITHM_CUDA_NORMAL_PARAM *param, int width, int height);
	process_1(param, width, height);
#endif
#if 0
	extern void process_0_nocopy(ALGORITHM_CUDA_NORMAL_PARAM *param, int* matDst, int* matSrc, int width, int height, int genTimes);
	process_0_nocopy(param, matDst, matSrc, width, height, genTimes);
#endif
}

void swapMat(ALGORITHM_CUDA_NORMAL_PARAM *param)
{
	int *temp;
	temp = param->devMatDst;
	param->devMatDst = param->devMatSrc;
	param->devMatSrc = temp;

	temp = param->hostMatDst;
	param->hostMatDst = param->hostMatSrc;
	param->hostMatSrc = temp;
}

/*
* Don't use cudaMallocManaged
* Memory access exception occurs when I call logicForOneGeneration from several threads
*/
void cudaAllocManaged(int **p, int size)
{
	cudaMallocManaged(p, size);
}

void cudaFreeManaged(int *p)
{
	cudaFree(p);
}

void cudaDeviceSynchronizeWrapper()
{
	cudaDeviceSynchronize();
}


}
