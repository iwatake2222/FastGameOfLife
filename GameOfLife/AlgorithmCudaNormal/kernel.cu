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

int NUM_STREAM = NUM_STREAM_MAX;

void cudaInitialize(ALGORITHM_CUDA_NORMAL_PARAM *param, int width, int height)
{
	NUM_STREAM = height / BLOCK_SIZE_H;
	if (NUM_STREAM > NUM_STREAM_MAX) NUM_STREAM = NUM_STREAM_MAX;


#if defined(USE_ZEROCOPY_MEMORY)
	#if defined(USE_PINNED_MEMORY)
	CHECK(cudaMallocHost((void**)&param->hostMatSrc, (width + 2 * MEMORY_MARGIN) * (height + 2 * MEMORY_MARGIN) * sizeof(int), cudaHostAllocMapped));
	CHECK(cudaMallocHost((void**)&param->hostMatDst, (width + 2 * MEMORY_MARGIN) * (height + 2 * MEMORY_MARGIN) * sizeof(int), cudaHostAllocMapped));
	#else
	CHECK(cudaHostAlloc((void**)&param->hostMatSrc, (width + 2 * MEMORY_MARGIN) * (height + 2 * MEMORY_MARGIN) * sizeof(int), cudaHostAllocMapped));
	CHECK(cudaHostAlloc((void**)&param->hostMatDst, (width + 2 * MEMORY_MARGIN) * (height + 2 * MEMORY_MARGIN) * sizeof(int), cudaHostAllocMapped));
	#endif
#elif defined(USE_PINNED_MEMORY)
	CHECK(cudaMallocHost((void**)&param->hostMatSrc, (width + 2 * MEMORY_MARGIN) * (height + 2 * MEMORY_MARGIN) * sizeof(int)));
	CHECK(cudaMallocHost((void**)&param->hostMatDst, (width + 2 * MEMORY_MARGIN) * (height + 2 * MEMORY_MARGIN) * sizeof(int)));
#else
	param->hostMatSrc = new int[(width + 2 * MEMORY_MARGIN) * (height + 2 * MEMORY_MARGIN)];
	param->hostMatDst = new int[(width + 2 * MEMORY_MARGIN) * (height + 2 * MEMORY_MARGIN)];
#endif

#ifdef USE_ZEROCOPY_MEMORY
	CHECK(cudaHostGetDevicePointer((void**)&param->devMatSrc, (void*)param->hostMatSrc, 0));
	CHECK(cudaHostGetDevicePointer((void**)&param->devMatDst, (void*)param->hostMatDst, 0));
#else
	CHECK(cudaMalloc((void**)&param->devMatSrc, (width + 2 * MEMORY_MARGIN) * (height + 2 * MEMORY_MARGIN) * sizeof(int)));
	CHECK(cudaMalloc((void**)&param->devMatDst, (width + 2 * MEMORY_MARGIN) * (height + 2 * MEMORY_MARGIN) * sizeof(int)));
#endif

	param->isMatrixUpdated = 1;

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

#if !defined(USE_ZEROCOPY_MEMORY)
	CHECK(cudaFree(param->devMatSrc));
	CHECK(cudaFree(param->devMatDst));
#endif

#if defined(USE_PINNED_MEMORY) || defined(USE_ZEROCOPY_MEMORY)
	CHECK(cudaFreeHost(param->hostMatSrc));
	CHECK(cudaFreeHost(param->hostMatDst));
#else
	delete param->hostMatSrc;
	delete param->hostMatDst;
#endif

	/* todo: call this at the really end of the application */
	//CHECK(cudaDeviceReset());
}

void cudaProcess(ALGORITHM_CUDA_NORMAL_PARAM *param, int width, int height, int repeatNum)
{
	for (int i = 0; i < repeatNum; i++) {
#if defined(ALGORITHM_0)
		extern void process_0(ALGORITHM_CUDA_NORMAL_PARAM *param, int width, int height);
		process_0(param, width, height);
#elif defined(ALGORITHM_0_STREAM)
		extern void process_0_stream(ALGORITHM_CUDA_NORMAL_PARAM *param, int width, int height);
		process_0_stream(param, width, height);
#elif defined(ALGORITHM_1)
		extern void process_1(ALGORITHM_CUDA_NORMAL_PARAM *param, int width, int height);
		process_1(param, width, height);
#elif defined(ALGORITHM_2)
		extern void process_2(ALGORITHM_CUDA_NORMAL_PARAM *param, int width, int height);
		process_2(param, width, height);
#elif defined(ALGORITHM_2_STREAM)
		extern void process_2_stream(ALGORITHM_CUDA_NORMAL_PARAM *param, int width, int height);
		process_2_stream(param, width, height);
#elif defined(ALGORITHM_3_STREAM)
		extern void process_3_stream(ALGORITHM_CUDA_NORMAL_PARAM *param, int width, int height);
		process_3_stream(param, width, height);
#elif defined(ALGORITHM_3_REPEAT)
		extern void process_3_repeat(ALGORITHM_CUDA_NORMAL_PARAM *param, int width, int height, int repeatNum);
		process_3_repeat(param, width, height, repeatNum);
		break;
#elif defined(ALGORITHM_3_AUTO)
		extern void process_3_stream(ALGORITHM_CUDA_NORMAL_PARAM *param, int width, int height);
		extern void process_3_repeat(ALGORITHM_CUDA_NORMAL_PARAM *param, int width, int height, int repeatNum);
		if (repeatNum == 1) {
			process_3_stream(param, width, height);
		} else {
			process_3_repeat(param, width, height, repeatNum);
		}
		break;
#endif
	}
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
