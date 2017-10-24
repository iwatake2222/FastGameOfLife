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

void cudaProcess(ALGORITHM_CUDA_NORMAL_PARAM *param, int* matDst, int* matSrc, int width, int height, int genTimes)
{
#if 1
	extern void process_0(ALGORITHM_CUDA_NORMAL_PARAM *param, int* matDst, int* matSrc, int width, int height, int genTimes);
	process_0(param, matDst, matSrc, width, height, genTimes);
#endif
#if 0
	extern void process_1(ALGORITHM_CUDA_NORMAL_PARAM *param, int* matDst, int* matSrc, int width, int height, int genTimes);
	process_1(param, matDst, matSrc, width, height, genTimes);
#endif
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
