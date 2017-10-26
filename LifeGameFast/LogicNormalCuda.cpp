#include "stdafx.h"
#include "LogicNormalCuda.h"
#include "algorithmCudaNormal.h"

using namespace AlgorithmCudaNormal;

LogicNormalCuda::LogicNormalCuda(int worldWidth, int worldHeight)
	: LogicNormal(worldWidth, worldHeight)
{
	cudaInitialize(&cudaParam, worldWidth, worldHeight);
}

LogicNormalCuda::~LogicNormalCuda()
{
	cudaFinalize(&cudaParam);
}


void LogicNormalCuda::gameLogic()
{
	m_info.generation++;

	if (m_isMatrixUpdated) {
		if (MEMORY_MARGIN == 0) {
			memcpy(cudaParam.hostMatSrc, m_matDisplay, sizeof(int) * WORLD_WIDTH * WORLD_HEIGHT);
		} else {
			for (int y = 0; y < WORLD_HEIGHT; y++) {
				memcpy(cudaParam.hostMatSrc + ((y + MEMORY_MARGIN) * (WORLD_WIDTH + 2 * MEMORY_MARGIN)) + MEMORY_MARGIN, m_matDisplay + WORLD_WIDTH * y, WORLD_WIDTH * sizeof(int));
			}
		}
	}

	cudaProcess(&cudaParam, WORLD_WIDTH, WORLD_HEIGHT);

	m_mutexMatDisplay.lock();	// wait if thread is copying matrix data 
	if (MEMORY_MARGIN == 0) {
		memcpy(m_matDisplay, cudaParam.hostMatSrc, sizeof(int) * WORLD_WIDTH * WORLD_HEIGHT);
	} else {
		for (int y = 0; y < WORLD_HEIGHT; y++) {
			memcpy(m_matDisplay + WORLD_WIDTH * y, cudaParam.hostMatSrc + ((y + MEMORY_MARGIN) * (WORLD_WIDTH + 2 * MEMORY_MARGIN)) + MEMORY_MARGIN, WORLD_WIDTH * sizeof(int));
		}
	}
	m_mutexMatDisplay.unlock();

}

