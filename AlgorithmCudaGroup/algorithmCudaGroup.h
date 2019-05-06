#pragma once

namespace AlgorithmCudaGroup
{
#if 0
}	// indent guard
#endif

//#define ALGORITHM_CPU
#define ALGORITHM_0
//#define ALGORITHM_0_STREAM
//#define ALGORITHM_1
//#define ALGORITHM_2
//#define ALGORITHM_2_STREAM
//#define ALGORITHM_3_STREAM
//#define ALGORITHM_3_REPEAT


const static int BLOCK_SIZE_W = 32;
const static int BLOCK_SIZE_H = 32;
const static int NUM_STREAM_MAX = 8;

#if defined(ALGORITHM_CPU) || defined(ALGORITHM_0) || defined(ALGORITHM_0_STREAM)
const static int MEMORY_MARGIN = 0;
#else
const static int MEMORY_MARGIN = 1;	// 1 pixel for each edge
#endif

/* must be the same as LogicGroup class (padding, alignment)*/
const static int PURE_GROUP_A = 999 * 1;
const static int PURE_GROUP_B = 999 * 2;
const static int CELL_DEAD = 0;
const static int CELL_ALIVE = 1;
typedef struct {
	int group;	// e.g.: PURE GROUP_A = 999*1, PURE GROUP_B = 999*2. A child of them = 999*1.5
	int age;
} DNA;

typedef struct {
	DNA *devMatSrc;
	DNA *devMatDst;
	DNA *hostMatSrc;
	DNA *hostMatDst;
	void* pStream[NUM_STREAM_MAX];	// type is cudaStream_t.
	int isMatrixUpdated;
} ALGORITHM_CUDA_GROUP_PARAM;


void cudaInitialize(ALGORITHM_CUDA_GROUP_PARAM *param, int width, int height);
void cudaFinalize(ALGORITHM_CUDA_GROUP_PARAM *param);
void cudaProcess(ALGORITHM_CUDA_GROUP_PARAM *param, int width, int height, int repeatNum);


}
