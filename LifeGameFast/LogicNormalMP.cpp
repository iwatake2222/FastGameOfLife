#include "stdafx.h"
#include "LogicNormalMP.h"
#include <omp.h>

LogicNormalMP::LogicNormalMP(int worldWidth, int worldHeight)
	: LogicNormal(worldWidth, worldHeight)
{
}

LogicNormalMP::~LogicNormalMP()
{
}

void LogicNormalMP::processWithBorderCheck(int x0, int x1, int y0, int y1)
{
#pragma omp parallel for
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
			updateCell(x, yLine, cnt);
		}
	}
}

/* don't check border, but fast */
void LogicNormalMP::processWithoutBorderCheck(int x0, int x1, int y0, int y1)
{
#pragma omp parallel for
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
			updateCell(x, yLine, cnt);
		}
	}
}

