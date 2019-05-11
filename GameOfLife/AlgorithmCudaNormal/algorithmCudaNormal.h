#pragma once

namespace AlgorithmCudaNormal
{
#if 0
}	// indent guard
#endif

/* Select the optimization method */
// #define ALGORITHM_0          // without padding (check border)
// #define ALGORITHM_0_STREAM   // without padding (check border), and stream
// #define ALGORITHM_1            // with padding()alias (don't check border)
// #define ALGORITHM_2          // with padding()alias (don't check border), and shared memory
// #define ALGORITHM_2_STREAM   // with padding()alias (don't check border), shared memory, and stream
// #define ALGORITHM_3_STREAM   // with padding()alias (don't check border), shared memory, and stream. do not copy from host->device if not updated by user
// #define ALGORITHM_3_REPEAT   // with padding()alias (don't check border), shared memory, and stream. do not copy from host->device nor from device->host when repeatNum is set
#define ALGORITHM_3_AUTO     // automatically switch ALGORITHM_3_STREAM and ALGORITHM_3_REPEAT

/* Select the memory allocation method */
// #define USE_ZEROCOPY_MEMORY
 #define USE_PINNED_MEMORY

// must be less than "Maximum number of threads per block"
const static int BLOCK_SIZE_W = 32;
const static int BLOCK_SIZE_H = 32;
const static int NUM_STREAM_MAX = 8;

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
	void* pStream[NUM_STREAM_MAX];	// type is cudaStream_t.
	int isMatrixUpdated;
} ALGORITHM_CUDA_NORMAL_PARAM;


void cudaAllocManaged(int **p, int size);
void cudaFreeManaged(int *p);
void cudaDeviceSynchronizeWrapper();
void cudaInitialize(ALGORITHM_CUDA_NORMAL_PARAM *param, int width, int height);
void cudaFinalize(ALGORITHM_CUDA_NORMAL_PARAM *param);
void cudaProcess(ALGORITHM_CUDA_NORMAL_PARAM *param, int width, int height, int repeatNum);


}
