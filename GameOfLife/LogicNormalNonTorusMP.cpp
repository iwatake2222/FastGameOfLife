#include <stdio.h>
#include "LogicNormalNonTorusMP.h"
#include <omp.h>

LogicNormalNonTorusMP::LogicNormalNonTorusMP(int worldWidth, int worldHeight)
	: LogicNormal(worldWidth, worldHeight)
{
}

LogicNormalNonTorusMP::~LogicNormalNonTorusMP()
{
}


void LogicNormalNonTorusMP::processWithBorderCheck(int x0, int x1, int y0, int y1)
{
#pragma omp parallel for
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
					if (m_matSrc[WORLD_WIDTH * roundY + roundX] != CELL_DEAD) {
						cnt++;
					}
				}
			}
			updateCell(x, yLine, cnt);
		}
	}
}

/* don't check border, but fast */
void LogicNormalNonTorusMP::processWithoutBorderCheck(int x0, int x1, int y0, int y1)
{
#pragma omp parallel for
	for (int y = y0; y < y1; y++) {
		int yLine = WORLD_WIDTH * y;
		for (int x = x0; x < x1; x++) {
			int cnt = 0;
			for (int yy = y - 1; yy <= y + 1; yy++) {
				for (int xx = x - 1; xx <= x + 1; xx++) {
					if (m_matSrc[WORLD_WIDTH * yy + xx] != CELL_DEAD) {
						cnt++;
					}
				}
			}
			updateCell(x, yLine, cnt);
		}
	}
}
