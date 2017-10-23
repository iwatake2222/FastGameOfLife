#pragma once

namespace AlgorithmCudaNormal
{
#if 0
}	// indent guard
#endif

typedef struct {
	int *devMatSrc;
	int *devMatDst;
	int isFirstOperation;
} ALGORITHM_CUDA_NORMAL_PARAM;

#ifdef DLL_EXPORT
//__declspec(dllexport) void allocManaged(int **p, int size);
//__declspec(dllexport) void freeManaged(int *p);
//__declspec(dllexport) void cudaDeviceSynchronizeWrapper();
__declspec(dllexport) void cudaInitialize(ALGORITHM_CUDA_NORMAL_PARAM *param, int width, int height);
__declspec(dllexport) void cudaFinalize(ALGORITHM_CUDA_NORMAL_PARAM *param);
__declspec(dllexport) void logicForOneGeneration(ALGORITHM_CUDA_NORMAL_PARAM *param, int* matDst, int* matSrc, int width, int height);
#else
//__declspec(dllimport) void allocManaged(int **p, int size);
//__declspec(dllimport) void freeManaged(int *p);
//__declspec(dllimport) void cudaDeviceSynchronizeWrapper();
__declspec(dllimport) void cudaInitialize(ALGORITHM_CUDA_NORMAL_PARAM *param, int width, int height);
__declspec(dllimport) void cudaFinalize(ALGORITHM_CUDA_NORMAL_PARAM *param);
__declspec(dllimport) void logicForOneGeneration(ALGORITHM_CUDA_NORMAL_PARAM *param, int* matDst, int* matSrc, int width, int height);
#endif

}