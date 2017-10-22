#include "stdafx.h"
#include "LogicNormalCuda.h"
#include "algorithmCudaNormal.h"

using namespace AlgorithmCudaNormal;

LogicNormalCuda::LogicNormalCuda(int worldWidth, int worldHeight)
	: LogicBase(worldWidth, worldHeight)
{
}

LogicNormalCuda::~LogicNormalCuda()
{
}

void LogicNormalCuda::allocMemory(int **p, int size)
{
	allocManaged(p, size * sizeof(int));
}

void LogicNormalCuda::freeMemory(int *p)
{
	freeManaged(p);
}


void LogicNormalCuda::gameLogic()
{
	WORLD_INFORMATION info = { 0 };
	info.generation = m_info.generation + 1;
	info.status = m_info.status;
	info.calcTime = m_info.calcTime;

	logicForOneGeneration(m_mat[m_matIdNew], &(info.numAlive), &(info.numBirth), &(info.numDie), m_mat[m_matIdOld], WORLD_WIDTH, WORLD_HEIGHT);

	std::this_thread::sleep_for(std::chrono::milliseconds(10));
	memcpy(m_matDisplay, m_mat[m_matIdNew], sizeof(int) * WORLD_WIDTH*WORLD_HEIGHT);

	m_info = info;
}

void LogicNormalCuda::loopWithBorder(int x0, int x1, int y0, int y1, WORLD_INFORMATION* info)
{
	for (int y = y0; y < y1; y++) {
		int yLine = WORLD_WIDTH * y;
		for (int x = x0; x < x1; x++) {
			int cnt = 0;
			for (int yy = y - 1; yy <= y + 1; yy++) {
				int roundY = yy;
				if (roundY >= WORLD_HEIGHT) roundY = 0;
				if (roundY < 0) roundY = WORLD_HEIGHT - 1;
				for (int xx = x - 1; xx <= x + 1; xx++) {
					int roundX = xx;
					if (roundX >= WORLD_WIDTH) roundX = 0;
					if (roundX < 0) roundX = WORLD_WIDTH - 1;
					if (m_mat[m_matIdOld][WORLD_WIDTH * roundY + roundX] != 0) {
						cnt++;
					}
				}
			}
			if (m_mat[m_matIdOld][yLine + x] == 0) {
				if (cnt == 3) {
					// birth
					m_mat[m_matIdNew][yLine + x] = 1;
					info->numAlive++;
					info->numBirth++;
				} else {
					// keep dead
					m_mat[m_matIdNew][yLine + x] = 0;
				}
			} else {
				if (cnt <= 2 || cnt >= 5) {
					// die
					m_mat[m_matIdNew][yLine + x] = 0;
					info->numDie++;
				} else {
					// keep alive (age++)
					m_mat[m_matIdNew][yLine + x] = m_mat[m_matIdOld][yLine + x] + 1;
					info->numAlive++;
				}
			}
		}
	}
}

/* don't check border, but fast */
void LogicNormalCuda::loopWithoutBorder(int x0, int x1, int y0, int y1, WORLD_INFORMATION* info)
{
	for (int y = y0; y < y1; y++) {
		int yLine = WORLD_WIDTH * y;
		for (int x = x0; x < x1; x++) {
			int cnt = 0;
			for (int yy = y - 1; yy <= y + 1; yy++) {
				for (int xx = x - 1; xx <= x + 1; xx++) {
					if (m_mat[m_matIdOld][WORLD_WIDTH * yy + xx] != 0) {
						cnt++;
					}
				}
			}
			if (m_mat[m_matIdOld][yLine + x] == 0) {
				if (cnt == 3) {
					// birth
					m_mat[m_matIdNew][yLine + x] = 1;
					info->numAlive++;
					info->numBirth++;
				} else {
					// keep dead
					m_mat[m_matIdNew][yLine + x] = 0;
				}
			} else {
				if (cnt <= 2 || cnt >= 5) {
					// die
					m_mat[m_matIdNew][yLine + x] = 0;
					info->numDie++;
				} else {
					// keep alive (age++)
					m_mat[m_matIdNew][yLine + x] = m_mat[m_matIdOld][yLine + x] + 1;
					info->numAlive++;
				}
			}
		}
	}
}
