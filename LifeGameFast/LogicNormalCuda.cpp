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
	for (int i = 0; i < 2; i++) {
		delete m_mat[i];
		m_mat[i] = 0;
	}

	for (int i = 0; i < 2; i++) {
		allocManaged(&m_mat[i], WORLD_WIDTH * WORLD_HEIGHT * sizeof(int));
		memset(m_mat[i], 0x00, sizeof(int) * WORLD_WIDTH * WORLD_HEIGHT);
	}
}

LogicNormalCuda::~LogicNormalCuda()
{
	for (int i = 0; i < 2; i++) {
		freeManaged(m_mat[i]);
		m_mat[i] = 0;
	}

	/* todo: should make it better */
	/* after this, base destructor tries to delete m_mat using delete (no effect) */
}


void LogicNormalCuda::gameLogic()
{
	m_info.generation++;

	logicForOneGeneration(m_mat[m_matIdNew], m_mat[m_matIdOld], WORLD_WIDTH, WORLD_HEIGHT);

	/* ššš@this takes time !!!*/
	/* in this algorithm, it just show the same value as each cell */
	m_mutexMatDisplay.lock();	// wait if thread is copying matrix data 
	memcpy(m_matDisplay, m_mat[m_matIdNew], sizeof(int) * WORLD_WIDTH * WORLD_HEIGHT);
	m_mutexMatDisplay.unlock();

	int tempId = m_matIdOld;
	m_matIdOld = m_matIdNew;
	m_matIdNew = tempId;
}

