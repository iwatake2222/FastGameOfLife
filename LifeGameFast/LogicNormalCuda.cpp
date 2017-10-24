#include "stdafx.h"
#include "LogicNormalCuda.h"
#include "algorithmCudaNormal.h"

using namespace AlgorithmCudaNormal;

LogicNormalCuda::LogicNormalCuda(int worldWidth, int worldHeight)
	: LogicNormal(worldWidth, worldHeight)
{
	cudaInitialize(&cudaParam, worldWidth, worldHeight);

	/* save original m_matDisplay because it will be modified in this class*/
	m_originalMatDisplay = m_matDisplay;
}

LogicNormalCuda::~LogicNormalCuda()
{
	cudaFinalize(&cudaParam);

	/* restore m_matDisplay because it will be freeed in base class */
	m_matDisplay = m_originalMatDisplay;
}


void LogicNormalCuda::gameLogic()
{
	m_info.generation++;

	cudaProcess(&cudaParam, m_mat[m_matIdNew], m_mat[m_matIdOld], WORLD_WIDTH, WORLD_HEIGHT, 1);

	/* in this algorithm, it just show the same value as each cell */
#if 1
	/* directly use matrix for display data */
	m_matDisplay = m_mat[m_matIdNew];
#else
	m_mutexMatDisplay.lock();	// wait if thread is copying matrix data 
	memcpy(m_matDisplay, m_mat[m_matIdNew], sizeof(int) * WORLD_WIDTH * WORLD_HEIGHT);
	m_mutexMatDisplay.unlock();
#endif

	int tempId = m_matIdOld;
	m_matIdOld = m_matIdNew;
	m_matIdNew = tempId;
}

