#include "stdafx.h"
#include "LogicNormalNonTorus.h"

LogicNormalNonTorus::LogicNormalNonTorus(int worldWidth, int worldHeight)
	: LogicNormal(worldWidth, worldHeight)
{
}

LogicNormalNonTorus::~LogicNormalNonTorus()
{
}


void LogicNormalNonTorus::processWithBorderCheck(int x0, int x1, int y0, int y1)
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
					if (m_matSrc[WORLD_WIDTH * roundY + roundX] != CELL_DEAD) {
						cnt++;
					}
				}
			}
			updateCell(x, yLine, cnt);
		}
	}
}

