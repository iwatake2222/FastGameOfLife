#pragma once

namespace AlgorithmCudaNormal
{
#if 0
}	// indent guard
#endif

//#define ALGORITHM_0
//#define ALGORITHM_0_STREAM
//#define ALGORITHM_1
//#define ALGORITHM_2
//#define ALGORITHM_2_STREAM
//#define ALGORITHM_3_STREAM
#define ALGORITHM_3_REPEAT

const static int BLOCK_SIZE_W = 32;
const static int BLOCK_SIZE_H = 32;
const static int NUM_STREAM = 8;

#if defined(ALGORITHM_0) || defined(ALGORITHM_0_STREAM)
const static int MEMORY_MARGIN = 0;
#else
const static int MEMORY_MARGIN = 1;	// 1 pixel for each edge
#endif

typedef struct {
	int *devMatSrc;
	int *devMatDst;
	int *hostMatSrc;
	int *hostMatDst;
	void* pStream[NUM_STREAM];	// type is cudaStream_t.
	int isMatrixUpdated;
} ALGORITHM_CUDA_NORMAL_PARAM;

#ifdef DLL_EXPORT
__declspec(dllexport) void cudaAllocManaged(int **p, int size);
__declspec(dllexport) void cudaFreeManaged(int *p);
__declspec(dllexport) void cudaDeviceSynchronizeWrapper();
__declspec(dllexport) void cudaInitialize(ALGORITHM_CUDA_NORMAL_PARAM *param, int width, int height);
__declspec(dllexport) void cudaFinalize(ALGORITHM_CUDA_NORMAL_PARAM *param);
__declspec(dllexport) void cudaProcess(ALGORITHM_CUDA_NORMAL_PARAM *param, int width, int height, int repeatNum);
#else
__declspec(dllimport) void cudaAllocManaged(int **p, int size);
__declspec(dllimport) void cudaFreeManaged(int *p);
__declspec(dllimport) void cudaDeviceSynchronizeWrapper();
__declspec(dllimport) void cudaInitialize(ALGORITHM_CUDA_NORMAL_PARAM *param, int width, int height);
__declspec(dllimport) void cudaFinalize(ALGORITHM_CUDA_NORMAL_PARAM *param);
__declspec(dllimport) void cudaProcess(ALGORITHM_CUDA_NORMAL_PARAM *param, int width, int height, int repeatNum);
#endif

}
