#include "stdafx.h"
#include "LogicNormal.h"

LogicNormal::LogicNormal(int worldWidth, int worldHeight)  
	: LogicBase(worldWidth, worldHeight)
{
	m_matIdOld = 0;
	m_matIdNew = 1;

	for (int i = 0; i < 2; i++) {
		m_mat[i] = new int[WORLD_WIDTH * WORLD_HEIGHT];
		memset(m_mat[i], 0x00, sizeof(int) * WORLD_WIDTH * WORLD_HEIGHT);
		memset(m_matDisplay, 0x00, sizeof(int) * WORLD_WIDTH * WORLD_HEIGHT);
	}
}

LogicNormal::~LogicNormal()
{
	for (int i = 0; i < 2; i++) {
		delete m_mat[i];
		m_mat[i] = 0;
	}
}

bool LogicNormal::toggleCell(int worldX, int worldY, int prm1, int prm2, int prm3, int prm4)
{
	if(LogicBase::toggleCell(worldX, worldY, prm1, prm2, prm3, prm4)) {
		if (m_matDisplay[WORLD_WIDTH * worldY + worldX] == CELL_DEAD) {
			setCell(worldX, worldY, prm1, prm2, prm3, prm4);
		} else {
			clearCell(worldX, worldY);
		}
		return true;
	}
	return false;
}

bool LogicNormal::setCell(int worldX, int worldY, int prm1, int prm2, int prm3, int prm4)
{
	if (LogicBase::setCell(worldX, worldY, prm1, prm2, prm3, prm4)) {
		m_matDisplay[WORLD_WIDTH * worldY + worldX] = CELL_ALIVE;
		m_mat[m_matIdOld][WORLD_WIDTH * worldY + worldX] = CELL_ALIVE;
		return true;
	}
	return false;
}

bool LogicNormal::clearCell(int worldX, int worldY)
{
	if (LogicBase::clearCell(worldX, worldY)) {
		m_matDisplay[WORLD_WIDTH * worldY + worldX] = CELL_DEAD;
		m_mat[m_matIdOld][WORLD_WIDTH * worldY + worldX] = CELL_DEAD;
		return true;
	}
	return false;
}

void LogicNormal::gameLogic() 
{
	m_info.generation++;

	/* four edges */
	processWithBorderCheck(0, WORLD_WIDTH, 0, 1);
	processWithBorderCheck(0, WORLD_WIDTH, WORLD_HEIGHT-1, WORLD_HEIGHT);
	processWithBorderCheck(0, 1, 0, WORLD_HEIGHT);
	processWithBorderCheck(WORLD_WIDTH-1, WORLD_WIDTH, 0, WORLD_HEIGHT);

	/* for most area */
	processWithoutBorderCheck(1, WORLD_WIDTH-1, 1, WORLD_HEIGHT-1);

	/* in this algorithm, it just show the same value as each cell */
	m_mutexMatDisplay.lock();	// wait if thread is copying matrix data 
	memcpy(m_matDisplay, m_mat[m_matIdNew], sizeof(int) * WORLD_WIDTH * WORLD_HEIGHT);
	m_mutexMatDisplay.unlock();

	int tempId = m_matIdOld;
	m_matIdOld = m_matIdNew;
	m_matIdNew = tempId;
}

void LogicNormal::processWithBorderCheck(int x0, int x1, int y0, int y1)
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
			updateCell(x, yLine, cnt);
		}
	}
}

/* don't check border, but fast */
void LogicNormal::processWithoutBorderCheck(int x0, int x1, int y0, int y1)
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
			updateCell(x, yLine, cnt);
		}
	}
}

inline void LogicNormal::updateCell(int x, int yLine, int cnt)
{
	/* Note: yLine is index of array (yLine = y*width) */
	if (m_mat[m_matIdOld][yLine + x] == 0) {
		if (cnt == 3) {
			// birth
			m_mat[m_matIdNew][yLine + x] = 1;
		} else {
			// keep dead
			m_mat[m_matIdNew][yLine + x] = 0;
		}
	} else {
		if (cnt <= 2 || cnt >= 5) {
			// die
			m_mat[m_matIdNew][yLine + x] = 0;
		} else {
			// keep alive (age++)
			m_mat[m_matIdNew][yLine + x] = m_mat[m_matIdOld][yLine + x] + 1;
		}
	}
}
