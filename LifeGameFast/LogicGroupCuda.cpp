#include "stdafx.h"
#include "LogicGroupCuda.h"
#include "algorithmCudaGroup.h"
#include "ControllerView.h"

/*
* Note: LogicGroupCuda doesn't use m_matSrc, m_matDst. uses m_cudaParam.hostMatSrc
*/

using namespace AlgorithmCudaGroup;

LogicGroupCuda::LogicGroupCuda(int worldWidth, int worldHeight)
	: LogicGroup(worldWidth, worldHeight)
{
	cudaInitialize(&m_cudaParam, worldWidth, worldHeight);
}

LogicGroupCuda::~LogicGroupCuda()
{
	cudaFinalize(&m_cudaParam);
}

int* LogicGroupCuda::getDisplayMat() {
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
			memcpy(m_matSrc, m_cudaParam.hostMatSrc, sizeof(DNA) * WORLD_WIDTH * WORLD_HEIGHT);// todo: may cause problem someday (memory allocation of struct might be different)
		} else {
			// todo
		}
		m_mutexMatDisplay.unlock();

		if (MEMORY_MARGIN == 0) {
			for (int i = 0; i < WORLD_WIDTH * WORLD_HEIGHT; i++) {
				//m_matDisplay[i] = convertCell2Display(&m_matSrc[i]);	
				m_matDisplay[i] = m_matSrc[i].age == 0 ? 0 :  m_matSrc[i].group;	// the same logic as convertCell2Display (for speeding up)
			}
		} else {
			// todo
		}
		m_lastRetrievedGenration = m_info.generation;
		
	}
	return m_matDisplay;
}

void LogicGroupCuda::gameLogic(int repeatNum)
{
	if (m_isMatrixUpdated) {
		/* copy pixel data from m_matDisplay -> m_matSrc when updated (when the first time, or when user put/clear cells) */
		m_cudaParam.isMatrixUpdated = 1;
		if (MEMORY_MARGIN == 0) {
			memcpy(m_cudaParam.hostMatSrc, m_matSrc, sizeof(DNA) * WORLD_WIDTH * WORLD_HEIGHT);
		} else {
			for (int y = 0; y < WORLD_HEIGHT; y++) {
				memcpy(m_cudaParam.hostMatSrc + ((y + MEMORY_MARGIN) * (WORLD_WIDTH + 2 * MEMORY_MARGIN)) + MEMORY_MARGIN, m_matSrc + WORLD_WIDTH * y, WORLD_WIDTH * sizeof(DNA));
			}
		}
		m_isMatrixUpdated = false;
	}

	m_mutexMatDisplay.lock();	// wait if view is copying matrix data 
	cudaProcess(&m_cudaParam, WORLD_WIDTH, WORLD_HEIGHT, repeatNum);
	m_info.generation += repeatNum;
	m_mutexMatDisplay.unlock();
}

