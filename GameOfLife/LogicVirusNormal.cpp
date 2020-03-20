#include <stdio.h>
#include <string.h>
#include "LogicVirusNormal.h"
#include "Values.h"


LogicVirusNormal::LogicVirusNormal(int worldWidth, int worldHeight)
	: LogicBase(worldWidth, worldHeight)
{
	m_matSrc = new int[WORLD_WIDTH * WORLD_HEIGHT];
	memset(m_matSrc, 0x00, sizeof(int) * WORLD_WIDTH * WORLD_HEIGHT);


}

LogicVirusNormal::~LogicVirusNormal()
{
	delete m_matSrc;
	m_matSrc = 0;
	for (int i = 0; i < m_personList.size(); i++) {
		delete m_personList[i];
	}
}

int LogicVirusNormal::getCellStatus(Person *p)
{
	int ret = HEALTHY;
	if (p->m_dayInfected > 0) ret = INFECTED;
	if (p->m_isOnset) ret = ONSET;
	if (p->m_isContagious) ret = CONTAGIUS;
	if (p->m_isImmunized) ret = IMMUNIZED;
	return ret;
}

void LogicVirusNormal::allocCells(int x0, int x1, int y0, int y1, int density, int prm1, int prm2, int prm3, int prm4)
{
	if (m_isCalculating) stopRun();
	if (x0 < 0) x0 = 0;
	if (x1 >= WORLD_WIDTH) x1 = WORLD_WIDTH;
	if (y0 < 0) y0 = 0;
	if (y1 >= WORLD_HEIGHT) y1 = WORLD_HEIGHT;
	if (density < 0) density = 0;
	if (density > 100) density = 100;

	m_info.generation++;

	int numPerson = (density * (x1 - x0) * (y1 - y0)) / 100;
	for (int i = 0; i < numPerson; i++) {
		m_personList.push_back(new Person(x0 / (double)WORLD_WIDTH, x1 / (double)WORLD_WIDTH, y0 / (double)WORLD_HEIGHT, y1 / (double)WORLD_HEIGHT));
	}

	for (int p = 0; p < m_personList.size(); p++) {
		int x, y;
		m_personList[p]->getPosition(&x, &y, WORLD_WIDTH, WORLD_HEIGHT);
		m_matSrc[WORLD_WIDTH * y + x] = getCellStatus(m_personList[p]);
		memcpy(m_matDisplay, m_matSrc, sizeof(int) * WORLD_WIDTH * WORLD_HEIGHT);
	}
}

inline int LogicVirusNormal::convertCell2Display(int cell)
{
	return cell;
}

inline void LogicVirusNormal::convertDisplay2Color(int displayedCell, double color[3])
{
	const double* colorTemp;
	int generation = m_info.generation;
	if (displayedCell == NONE) {
		colorTemp = COLOR_3D_VIRUS_NONE;
	} else if (displayedCell == HEALTHY) {
		colorTemp = COLOR_3D_VIRUS_HEALTHY;
	} else if (displayedCell == INFECTED) {
		colorTemp = COLOR_3D_VIRUS_INFECTED;
	} else if (displayedCell == ONSET) {
		colorTemp = COLOR_3D_VIRUS_ONSET;
	} else if (displayedCell == CONTAGIUS) {
		colorTemp = COLOR_3D_VIRUS_CONTAGIUS;
	} else if (displayedCell == IMMUNIZED) {
		colorTemp = COLOR_3D_VIRUS_IMMUNIZED;
	}
	memcpy(color, colorTemp, sizeof(double) * 3);

}

int* LogicVirusNormal::getDisplayMat() {
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

bool LogicVirusNormal::toggleCell(int worldX, int worldY, int prm1, int prm2, int prm3, int prm4)
{
	//if(LogicBase::toggleCell(worldX, worldY, prm1, prm2, prm3, prm4)) {
	//	if (m_matDisplay[WORLD_WIDTH * worldY + worldX] == CELL_DEAD) {
	//		setCell(worldX, worldY, prm1, prm2, prm3, prm4);
	//	} else {
	//		clearCell(worldX, worldY);
	//	}
	//	return true;
	//}
	return false;
}

bool LogicVirusNormal::setCell(int worldX, int worldY, int prm1, int prm2, int prm3, int prm4)
{
	//if (LogicBase::setCell(worldX, worldY, prm1, prm2, prm3, prm4)) {
	//	m_matSrc[WORLD_WIDTH * worldY + worldX] = 1;
	//	m_matDisplay[WORLD_WIDTH * worldY + worldX] = convertCell2Display(m_matSrc[WORLD_WIDTH * worldY + worldX]);
	//	return true;
	//}
	return false;
}

bool LogicVirusNormal::clearCell(int worldX, int worldY)
{
	//if (LogicBase::clearCell(worldX, worldY)) {
	//	m_matSrc[WORLD_WIDTH * worldY + worldX] = 0;
	//	m_matDisplay[WORLD_WIDTH * worldY + worldX] = convertCell2Display(m_matSrc[WORLD_WIDTH * worldY + worldX]);
	//	return true;
	//}
	return false;
}

void LogicVirusNormal::gameLogic(int repeatNum)
{
	//if (m_isMatrixUpdated) {
	//	/* copy pixel data from m_matDisplay -> m_matSrc when updated (when the first time, or when user put/clear cells) */
	//	memcpy(m_matSrc, m_matDisplay, sizeof(int) * WORLD_WIDTH * WORLD_HEIGHT);
	//	m_isMatrixUpdated = 0;
	//}

	for (int i = 0; i < repeatNum; i++) {
#pragma omp parallel for
		for (int p = 0; p < m_personList.size(); p++) {
			m_personList[p]->updatePos();
		}
#pragma omp parallel for
		for (int p = 0; p < m_personList.size(); p++) {
			m_personList[p]->updateSymptom(m_personList);
		}
	}

	m_mutexMatDisplay.lock();	// wait if view is copying matrix data 
	memset(m_matSrc, 0x00, sizeof(int) * WORLD_WIDTH * WORLD_HEIGHT);
	for (int p = 0; p < m_personList.size(); p++) {
		int x, y;
		m_personList[p]->getPosition(&x, &y, WORLD_WIDTH, WORLD_HEIGHT);
		m_matSrc[WORLD_WIDTH * y + x] = getCellStatus(m_personList[p]);
	}

	m_info.generation += repeatNum;
	m_mutexMatDisplay.unlock();

	// m_matSrc is ready to display
	
}
