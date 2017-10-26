#pragma once
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

void swapMat(ALGORITHM_CUDA_NORMAL_PARAM *param);
}

void printMatrix(int *mat, int width, int height);
