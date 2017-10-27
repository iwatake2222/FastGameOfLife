#include "stdafx.h"
#include "LogicNormalCuda.h"
#include "algorithmCudaNormal.h"

using namespace AlgorithmCudaNormal;

LogicNormalCuda::LogicNormalCuda(int worldWidth, int worldHeight)
	: LogicNormal(worldWidth, worldHeight)
{
	cudaInitialize(&m_cudaParam, worldWidth, worldHeight);
}

LogicNormalCuda::~LogicNormalCuda()
{
	cudaFinalize(&m_cudaParam);
}

int* LogicNormalCuda::getDisplayMat() {
	if (m_lastRetrievedGenration != m_info.generation) {
		/* update display matrix if generation proceeded */
		//m_mutexMatDisplay.lock();	// wait if thread is copying matrix data 
		if (MEMORY_MARGIN == 0) {
			memcpy(m_matDisplay, m_cudaParam.hostMatSrc, sizeof(int) * WORLD_WIDTH * WORLD_HEIGHT);
		} else {
			for (int y = 0; y < WORLD_HEIGHT; y++) {
				memcpy(m_matDisplay + WORLD_WIDTH * y, m_cudaParam.hostMatSrc + ((y + MEMORY_MARGIN) * (WORLD_WIDTH + 2 * MEMORY_MARGIN)) + MEMORY_MARGIN, WORLD_WIDTH * sizeof(int));
			}
		}
		m_lastRetrievedGenration = m_info.generation;
		//m_mutexMatDisplay.unlock();
	}
	return m_matDisplay;
}

void LogicNormalCuda::gameLogic()
{
	if (m_isMatrixUpdated) {
		m_cudaParam.isMatrixUpdated = 1;
		if (MEMORY_MARGIN == 0) {
			memcpy(m_cudaParam.hostMatSrc, m_matDisplay, sizeof(int) * WORLD_WIDTH * WORLD_HEIGHT);
		} else {
			for (int y = 0; y < WORLD_HEIGHT; y++) {
				memcpy(m_cudaParam.hostMatSrc + ((y + MEMORY_MARGIN) * (WORLD_WIDTH + 2 * MEMORY_MARGIN)) + MEMORY_MARGIN, m_matDisplay + WORLD_WIDTH * y, WORLD_WIDTH * sizeof(int));
			}
		}
		m_isMatrixUpdated = false;
	}

	//m_mutexMatDisplay.lock();	// wait if thread is copying matrix data 
	cudaProcess(&m_cudaParam, WORLD_WIDTH, WORLD_HEIGHT, 1);
	m_info.generation++;
	//m_mutexMatDisplay.unlock();

}

