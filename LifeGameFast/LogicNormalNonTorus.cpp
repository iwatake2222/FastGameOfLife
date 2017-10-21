#include "stdafx.h"
#include "LogicNormalNonTorus.h"



LogicNormalNonTorus::LogicNormalNonTorus(int worldWidth, int worldHeight)
	: LogicBase(worldWidth, worldHeight)
{
}

LogicNormalNonTorus::~LogicNormalNonTorus()
{
}

void LogicNormalNonTorus::gameLogic()
{
	WORLD_INFORMATION info = { 0 };
	info.generation = m_info.generation + 1;
	info.status = m_info.status;
	info.calcTime = m_info.calcTime;

	/* four edges */
	loopWithBorder(0, WORLD_WIDTH, 0, 1, &info);
	loopWithBorder(0, WORLD_WIDTH, WORLD_HEIGHT - 1, WORLD_HEIGHT, &info);
	loopWithBorder(0, 1, 0, WORLD_HEIGHT, &info);
	loopWithBorder(WORLD_WIDTH - 1, WORLD_WIDTH, 0, WORLD_HEIGHT, &info);

	/* for most area */
	loopWithoutBorder(1, WORLD_WIDTH - 1, 1, WORLD_HEIGHT - 1, &info);

	m_info = info;
}

void LogicNormalNonTorus::loopWithBorder(int x0, int x1, int y0, int y1, WORLD_INFORMATION* info)
{
	for (int y = y0; y < y1; y++) {
		int yLine = WORLD_WIDTH * y;
		for (int x = x0; x < x1; x++) {
			int cnt = 0;
			for (int yy = y - 1; yy <= y + 1; yy++) {
				int roundY = yy;
				if (roundY >= WORLD_HEIGHT) continue;
				if (roundY < 0) continue;
				for (int xx = x - 1; xx <= x + 1; xx++) {
					int roundX = xx;
					if (roundX >= WORLD_WIDTH) continue;
					if (roundX < 0) continue;
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
void LogicNormalNonTorus::loopWithoutBorder(int x0, int x1, int y0, int y1, WORLD_INFORMATION* info)
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
