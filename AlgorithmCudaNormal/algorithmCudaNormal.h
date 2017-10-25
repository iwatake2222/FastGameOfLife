#pragma once

namespace AlgorithmCudaNormal
{
#if 0
}	// indent guard
#endif

const static int BLOCK_SIZE_W = 16;
const static int BLOCK_SIZE_H = 16;
const static int NUM_STREAM = 4;

typedef struct {
	int *devMatSrc;
	int *devMatDst;
	int *hostMatSrc;
	int *hostMatDst;
	void* pStream[NUM_STREAM];	// type is cudaStream_t.
	int isFirstOperation;
} ALGORITHM_CUDA_NORMAL_PARAM;

void swapMat(ALGORITHM_CUDA_NORMAL_PARAM *param);

#ifdef DLL_EXPORT
__declspec(dllexport) void cudaAllocManaged(int **p, int size);
__declspec(dllexport) void cudaFreeManaged(int *p);
__declspec(dllexport) void cudaDeviceSynchronizeWrapper();
__declspec(dllexport) void cudaInitialize(ALGORITHM_CUDA_NORMAL_PARAM *param, int width, int height);
__declspec(dllexport) void cudaFinalize(ALGORITHM_CUDA_NORMAL_PARAM *param);
__declspec(dllexport) void cudaProcess(ALGORITHM_CUDA_NORMAL_PARAM *param, int width, int height);
#else
__declspec(dllimport) void cudaAllocManaged(int **p, int size);
__declspec(dllimport) void cudaFreeManaged(int *p);
__declspec(dllimport) void cudaDeviceSynchronizeWrapper();
__declspec(dllimport) void cudaInitialize(ALGORITHM_CUDA_NORMAL_PARAM *param, int width, int height);
__declspec(dllimport) void cudaFinalize(ALGORITHM_CUDA_NORMAL_PARAM *param);
__declspec(dllimport) void cudaProcess(ALGORITHM_CUDA_NORMAL_PARAM *param, int width, int height);
#endif

}
