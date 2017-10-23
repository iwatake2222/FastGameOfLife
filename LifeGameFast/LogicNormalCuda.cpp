#include "stdafx.h"
#include "LogicNormalCuda.h"
#include "algorithmCudaNormal.h"

using namespace AlgorithmCudaNormal;

LogicNormalCuda::LogicNormalCuda(int worldWidth, int worldHeight)
	: LogicNormal(worldWidth, worldHeight)
{
	/* todo: should make it better */
	/* in this class, m_mat must be shared with GPU */
	/* don't use memory allocated by super class*/
	//for (int i = 0; i < 2; i++) {
	//	delete m_mat[i];
	//	m_mat[i] = 0;
	//}

	//for (int i = 0; i < 2; i++) {
	//	allocManaged(&m_mat[i], WORLD_WIDTH * WORLD_HEIGHT * sizeof(int));
	//	memset(m_mat[i], 0x00, sizeof(int) * WORLD_WIDTH * WORLD_HEIGHT);
	//}


	cudaInitialize(&cudaParam, worldWidth, worldHeight);

	/* save original m_matDisplay because it will be modified in this class*/
	m_originalMatDisplay = m_matDisplay;
}

LogicNormalCuda::~LogicNormalCuda()
{
//	for (int i = 0; i < 2; i++) {
//		freeManaged(m_mat[i]);
//		m_mat[i] = 0;
//	}

	cudaFinalize(&cudaParam);

	/* restore m_matDisplay because it will be freeed in base class */
	m_matDisplay = m_originalMatDisplay;
}


void LogicNormalCuda::gameLogic()
{
	m_info.generation++;

	logicForOneGeneration(&cudaParam, m_mat[m_matIdNew], m_mat[m_matIdOld], WORLD_WIDTH, WORLD_HEIGHT);

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

