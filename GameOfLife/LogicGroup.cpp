#include <stdio.h>
#include "LogicGroup.h"
#include "Values.h"

LogicGroup::LogicGroup(int worldWidth, int worldHeight)
	: LogicBase(worldWidth, worldHeight)
{
	m_matSrc = new DNA[WORLD_WIDTH * WORLD_HEIGHT];
	m_matDst = new DNA[WORLD_WIDTH * WORLD_HEIGHT];
	memset(m_matSrc, 0x00, sizeof(DNA) * WORLD_WIDTH * WORLD_HEIGHT);
	memset(m_matDst, 0x00, sizeof(DNA) * WORLD_WIDTH * WORLD_HEIGHT);
}

LogicGroup::~LogicGroup()
{
	delete m_matSrc;
	delete m_matDst;
	m_matSrc = 0;
	m_matDst = 0;
}

inline int LogicGroup::convertCell2Display(DNA *cell)
{
	if (cell->age == CELL_DEAD) {
		return 0;
	} else {
		return cell->group;	// 999*1 ~ 999*2
	}
}

inline void LogicGroup::convertDisplay2Color(int displayedCell, double color[3])
{
	if (displayedCell == 0) {
		color[0] = color[1] = color[2] = 0.0;
	} else {
		double normalizedGroup = (displayedCell - PURE_GROUP_A) / (double)PURE_GROUP_A;	// 0.0 ~ 1.0
		color[0] = COLOR_3D_ALIVE_GROUP_A[0] * (1 - normalizedGroup) + COLOR_3D_ALIVE_GROUP_B[0] * normalizedGroup;
		color[1] = COLOR_3D_ALIVE_GROUP_A[1] * (1 - normalizedGroup) + COLOR_3D_ALIVE_GROUP_B[1] * normalizedGroup;
		color[2] = COLOR_3D_ALIVE_GROUP_A[2] * (1 - normalizedGroup) + COLOR_3D_ALIVE_GROUP_B[2] * normalizedGroup;
	}
}

int* LogicGroup::getDisplayMat() {
	if (m_lastRetrievedGenration != m_info.generation ) {
		/* copy pixel data from m_matSrc -> m_matDisplay when updated (when generation proceeded) */
		int tryCnt = 0;
		while (!m_mutexMatDisplay.try_lock()) {
			// wait if thread is copying matrix data. give up after several tries not to decrease draw performance
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			if (tryCnt++ > 10) return m_matDisplay;
		}
		for (int i = 0; i < WORLD_WIDTH * WORLD_HEIGHT; i++) {
			m_matDisplay[i] = convertCell2Display(&m_matSrc[i]);
		}
		m_lastRetrievedGenration = m_info.generation;
		m_mutexMatDisplay.unlock();
	}
	return m_matDisplay;
}

bool LogicGroup::toggleCell(int worldX, int worldY, int prm1, int prm2, int prm3, int prm4)
{
	if (LogicBase::toggleCell(worldX, worldY, prm1, prm2, prm3, prm4)) {
		if (m_matDisplay[WORLD_WIDTH * worldY + worldX] == 0) {
			setCell(worldX, worldY, prm1, prm2, prm3, prm4);
		} else {
			clearCell(worldX, worldY);
		}
		return true;
	}
	return false;
}


bool LogicGroup::setCell(int worldX, int worldY, int prm1, int prm2, int prm3, int prm4)
{
	if (LogicBase::setCell(worldX, worldY, prm1, prm2, prm3, prm4)) {
		m_matSrc[WORLD_WIDTH * worldY + worldX].age = CELL_ALIVE;
		if (prm1 == 0) {
			m_matSrc[WORLD_WIDTH * worldY + worldX].group = PURE_GROUP_A;
		} else {
			m_matSrc[WORLD_WIDTH * worldY + worldX].group = PURE_GROUP_B;
		}
		m_matDisplay[WORLD_WIDTH * worldY + worldX] = convertCell2Display(&m_matSrc[WORLD_WIDTH * worldY + worldX]);
		return true;
	}
	return false;
}

bool LogicGroup::clearCell(int worldX, int worldY)
{
	if (LogicBase::clearCell(worldX, worldY)) {
		m_matDisplay[WORLD_WIDTH * worldY + worldX] = 0;
		m_matSrc[WORLD_WIDTH * worldY + worldX].age = CELL_DEAD;
		return true;
	}
	return false;
}

void LogicGroup::gameLogic(int repeatNum)
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

		DNA *temp = m_matSrc;
		m_matSrc = m_matDst;
		m_matDst = temp;
	}
	m_info.generation += repeatNum;
	m_mutexMatDisplay.unlock();

	// m_matSrc is ready to display

}

void LogicGroup::processWithBorderCheck(int x0, int x1, int y0, int y1)
{
#pragma omp parallel for
	for (int y = y0; y < y1; y++) {
		int yLine = WORLD_WIDTH * y;
		for (int x = x0; x < x1; x++) {
			int cnt = 0;
			int group = 0;
			for (int yy = y - 1; yy <= y + 1; yy++) {
				int roundY = yy;
				if (roundY >= WORLD_HEIGHT) roundY = 0;
				if (roundY < 0) roundY = WORLD_HEIGHT - 1;
				for (int xx = x - 1; xx <= x + 1; xx++) {
					int roundX = xx;
					if (roundX >= WORLD_WIDTH) roundX = 0;
					if (roundX < 0) roundX = WORLD_WIDTH - 1;
					if (m_matSrc[WORLD_WIDTH * roundY + roundX].age != CELL_DEAD) {
						cnt++;
						group += m_matSrc[WORLD_WIDTH * roundY + roundX].group;
					}
				}
			}
			updateCell(x, yLine, cnt, group);
		}
	}
}

/* don't check border, but fast */
void LogicGroup::processWithoutBorderCheck(int x0, int x1, int y0, int y1)
{
#pragma omp parallel for
	for (int y = y0; y < y1; y++) {
		int yLine = WORLD_WIDTH * y;
		for (int x = x0; x < x1; x++) {
			int cnt = 0;
			int group = 0;
			for (int yy = y - 1; yy <= y + 1; yy++) {
				for (int xx = x - 1; xx <= x + 1; xx++) {
					if (m_matSrc[WORLD_WIDTH * yy + xx].age != CELL_DEAD) {
						cnt++;
						group += m_matSrc[WORLD_WIDTH * yy + xx].group;
					}
				}
			}
			updateCell(x, yLine, cnt, group);
		}
	}
}

inline void LogicGroup::updateCell(int x, int yLine, int cnt, int group)
{
	/* Note: yLine is index of array (yLine = y*width) */
	if (m_matSrc[yLine + x].age == 0) {
		if (cnt == 3) {
			// birth
			m_matDst[yLine + x].age = CELL_ALIVE;
			m_matDst[yLine + x].group = group / 3;
			//printf("%d\n", m_matDst[yLine + x].group);
		} else {
			// keep dead
			m_matDst[yLine + x].age = CELL_DEAD;
		}
	} else {
		if (cnt <= 2 || cnt >= 5) {
			// die
			m_matDst[yLine + x].age = CELL_DEAD;
		} else {
			// keep alive (age++)
			m_matDst[yLine + x].age = m_matSrc[yLine + x].age + 1;
			m_matDst[yLine + x].group = m_matSrc[yLine + x].group;
		}
	}
}