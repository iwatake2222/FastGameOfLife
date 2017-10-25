#include "stdafx.h"
#include "LogicNormalCuda.h"
#include "algorithmCudaNormal.h"

using namespace AlgorithmCudaNormal;

LogicNormalCuda::LogicNormalCuda(int worldWidth, int worldHeight)
	: LogicNormal(worldWidth, worldHeight)
{
	cudaInitialize(&cudaParam, worldWidth, worldHeight);

	/* save original m_mat because they will be modified in this class */
	/* todo: separate class */
	m_originalMatDisplay = m_matDisplay;
	m_originalMat0 = m_mat[0];
	m_originalMat1 = m_mat[1];
	
	m_mat[0] = cudaParam.hostMatSrc;
	m_mat[1] = cudaParam.hostMatDst;
}

LogicNormalCuda::~LogicNormalCuda()
{
	cudaFinalize(&cudaParam);

	/* restore original m_mat because they need to be rekeased in the base class */
	m_matDisplay = m_originalMatDisplay;
	m_mat[0] = m_originalMat0;
	m_mat[1] = m_originalMat1;
}


void LogicNormalCuda::gameLogic()
{
	m_info.generation++;

	cudaProcess(&cudaParam, WORLD_WIDTH, WORLD_HEIGHT);
	m_mat[0] = cudaParam.hostMatSrc;
	m_mat[1] = cudaParam.hostMatDst;

	/* in this algorithm, it just show the same value as each cell */
#if 1
	/* directly use matrix for display data */
	m_matDisplay = m_mat[0];
#else
	m_mutexMatDisplay.lock();	// wait if thread is copying matrix data 
	memcpy(m_matDisplay, m_mat[m_matIdNew], sizeof(int) * WORLD_WIDTH * WORLD_HEIGHT);
	m_mutexMatDisplay.unlock();
#endif

	m_matIdOld = 0;
	m_matIdNew = 1;
}

