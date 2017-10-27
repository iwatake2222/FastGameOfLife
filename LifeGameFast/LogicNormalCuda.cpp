#include "stdafx.h"
#include "LogicNormalCuda.h"
#include "algorithmCudaNormal.h"
#include "ControllerView.h"

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
	if (m_lastRetrievedGenration != m_info.generation && !m_isMatrixUpdated) {
		/* copy pixel data from m_matSrc -> m_matDisplay when updated (when generation proceeded) */
		/* if user updated m_matDisplay(m_isMatrixUpdated is true), use m_matDisplay directly */
		int tryCnt = 0;
		while (!m_mutexMatDisplay.try_lock()) {
			// wait if thread is copying matrix data. give up after several tries not to decrease draw performance
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			if (tryCnt++ > 10) return m_matDisplay;
		}
		if (MEMORY_MARGIN == 0) {
			memcpy(m_matDisplay, m_cudaParam.hostMatSrc, sizeof(int) * WORLD_WIDTH * WORLD_HEIGHT);
		} else {
			for (int y = 0; y < WORLD_HEIGHT; y++) {
				memcpy(m_matDisplay + WORLD_WIDTH * y, m_cudaParam.hostMatSrc + ((y + MEMORY_MARGIN) * (WORLD_WIDTH + 2 * MEMORY_MARGIN)) + MEMORY_MARGIN, WORLD_WIDTH * sizeof(int));
			}
		}
		m_lastRetrievedGenration = m_info.generation;
		m_mutexMatDisplay.unlock();
	}
	return m_matDisplay;
}

void LogicNormalCuda::gameLogic(int repeatNum)
{
	if (m_isMatrixUpdated) {
		/* copy pixel data from m_matDisplay -> m_matSrc when updated (when the first time, or when user put/clear cells) */
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

	m_mutexMatDisplay.lock();	// wait if view is copying matrix data 
	cudaProcess(&m_cudaParam, WORLD_WIDTH, WORLD_HEIGHT, repeatNum);
	m_info.generation += repeatNum;
	m_mutexMatDisplay.unlock();

}

