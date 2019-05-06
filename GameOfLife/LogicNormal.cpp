#include <stdio.h>
#include "LogicNormal.h"
#include "Values.h"

LogicNormal::LogicNormal(int worldWidth, int worldHeight)  
	: LogicBase(worldWidth, worldHeight)
{
	m_matSrc = new int[WORLD_WIDTH * WORLD_HEIGHT];
	m_matDst = new int[WORLD_WIDTH * WORLD_HEIGHT];
	memset(m_matSrc, 0x00, sizeof(int) * WORLD_WIDTH * WORLD_HEIGHT);
	memset(m_matDst, 0x00, sizeof(int) * WORLD_WIDTH * WORLD_HEIGHT);
	
}

LogicNormal::~LogicNormal()
{
	delete m_matSrc;
	delete m_matDst;
	m_matSrc = 0;
	m_matDst = 0;
}

inline int LogicNormal::convertCell2Display(int cell)
{
	return cell;
}

inline void LogicNormal::convertDisplay2Color(int displayedCell, double color[3])
{
	const double* colorTemp;
	int generation = m_info.generation;
	if (displayedCell == 0) {
		colorTemp = COLOR_3D_DEAD;
	} else if (generation == 0 || (displayedCell * 100) / generation < 20) {
		colorTemp = COLOR_3D_ALIVE0;
	} else if ((displayedCell * 100) / generation < 40) {
		colorTemp = COLOR_3D_ALIVE1;
	} else if ((displayedCell * 100) / generation < 60) {
		colorTemp = COLOR_3D_ALIVE2;
	} else if ((displayedCell * 100) / generation < 80) {
		colorTemp = COLOR_3D_ALIVE3;
	} else {
		colorTemp = COLOR_3D_ALIVE4;
	}
	memcpy(color, colorTemp, sizeof(double) * 3);
}

int* LogicNormal::getDisplayMat() {
	if (m_lastRetrievedGenration != m_info.generation) {
		/* copy pixel data from m_matSrc -> m_matDisplay when updated (when generation proceeded) */
		int tryCnt = 0;
		while (!m_mutexMatDisplay.try_lock()) {
			// wait if thread is copying matrix data. give up after several tries not to decrease draw performance
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			if (tryCnt++ > 10) return m_matDisplay;
		}
#if 1
		memcpy(m_matDisplay, m_matSrc, sizeof(int) * WORLD_WIDTH * WORLD_HEIGHT);
#else
		for (int i = 0; i < WORLD_WIDTH * WORLD_HEIGHT; i++) {
			m_matDisplay[i] = convertCell2Display(m_matSrc[i]);
		}
#endif
		m_lastRetrievedGenration = m_info.generation;
		m_mutexMatDisplay.unlock();
	}
	return m_matDisplay;
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
		m_matSrc[WORLD_WIDTH * worldY + worldX] = CELL_ALIVE;
		m_matDisplay[WORLD_WIDTH * worldY + worldX] = convertCell2Display(m_matSrc[WORLD_WIDTH * worldY + worldX]);
		return true;
	}
	return false;
}

bool LogicNormal::clearCell(int worldX, int worldY)
{
	if (LogicBase::clearCell(worldX, worldY)) {
		m_matSrc[WORLD_WIDTH * worldY + worldX] = CELL_DEAD;
		m_matDisplay[WORLD_WIDTH * worldY + worldX] = convertCell2Display(m_matSrc[WORLD_WIDTH * worldY + worldX]);
		return true;
	}
	return false;
}

void LogicNormal::gameLogic(int repeatNum)
{
	//if (m_isMatrixUpdated) {
	//	/* copy pixel data from m_matDisplay -> m_matSrc when updated (when the first time, or when user put/clear cells) */
	//	memcpy(m_matSrc, m_matDisplay, sizeof(int) * WORLD_WIDTH * WORLD_HEIGHT);
	//	m_isMatrixUpdated = 0;
	//}

	m_mutexMatDisplay.lock();	// wait if view is copying matrix data 
	for (int i = 0; i < repeatNum; i++) {
		/* four edges */
		processWithBorderCheck(0, WORLD_WIDTH, 0, 1);
		processWithBorderCheck(0, WORLD_WIDTH, WORLD_HEIGHT - 1, WORLD_HEIGHT);
		processWithBorderCheck(0, 1, 0, WORLD_HEIGHT);
		processWithBorderCheck(WORLD_WIDTH - 1, WORLD_WIDTH, 0, WORLD_HEIGHT);

		/* for most area */
		processWithoutBorderCheck(1, WORLD_WIDTH - 1, 1, WORLD_HEIGHT - 1);

		int *temp = m_matSrc;
		m_matSrc = m_matDst;
		m_matDst = temp;
	}
	m_info.generation += repeatNum;
	m_mutexMatDisplay.unlock();

	// m_matSrc is ready to display
	
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
void LogicNormal::processWithoutBorderCheck(int x0, int x1, int y0, int y1)
{
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

inline void LogicNormal::updateCell(int x, int yLine, int cnt)
{
	/* Note: yLine is index of array (yLine = y*width) */
	if (m_matSrc[yLine + x] == CELL_DEAD) {
		if (cnt == 3) {
			// birth
			m_matDst[yLine + x] = CELL_ALIVE;
		} else {
			// keep dead
			m_matDst[yLine + x] = CELL_DEAD;
		}
	} else {
		if (cnt <= 2 || cnt >= 5) {
			// die
			m_matDst[yLine + x] = CELL_DEAD;
		} else {
			// keep alive (age++)
			m_matDst[yLine + x] = m_matSrc[yLine + x] + 1;
		}
	}
}
