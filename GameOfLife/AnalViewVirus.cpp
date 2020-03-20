#include "Common.h"
#include <stdio.h>
#include <GL/glut.h>
#include <GL/freeglut.h>
#include "AnalViewVirus.h"
#include "LogicVirusNormal.h"
#include "WindowManager.h"

AnalViewVirus::AnalViewVirus(WorldContext* pContext)
	: AnalView(pContext)
{
}


AnalViewVirus::~AnalViewVirus()
{
}

void AnalViewVirus::updateAnalInfo()
{
	ANAL_INFORMATION newInfo = { 0 };
	ILogic::WORLD_INFORMATION worldInfo;
	m_pContext->m_pLogic->getInformation(&worldInfo);

	if (worldInfo.status != 0 || worldInfo.generation != m_lastAnalyzedGeneration) {
		m_lastAnalyzedGeneration = worldInfo.generation;
		/*** Analyse informaiton of the current generation ***/
		analyseInformation(&newInfo);

		/*** Update analysis informaiton history ***/
		analInfoHistory.push_back(newInfo);
		if (analInfoHistory.size() > m_NumHistory) {
			//analInfoHistory.pop_front();
			m_NumHistory++;
		}
	} else {
		/* logic is not running. use the previous generation's information*/
		if (analInfoHistory.size() > 0) {
			newInfo = analInfoHistory.back();
		}
	}

	/*** Display analysis informaiton graph ***/
	if (m_maxGraphY < newInfo.numTotal) m_maxGraphY = newInfo.numTotal;
	displayInformationGraph();

	/*** Display analysis informaiton graph ***/
	displayInformationText(&newInfo);
}

void AnalViewVirus::analyseInformation(ANAL_INFORMATION *info)
{
	int worldWidth = m_pContext->WORLD_WIDTH;
	int worldHeight = m_pContext->WORLD_HEIGHT;
	int *mat = m_pContext->m_pLogic->getDisplayMat();

	ILogic::WORLD_INFORMATION worldInfo;
	m_pContext->m_pLogic->getInformation(&worldInfo);
	int generation = worldInfo.generation;

	for (int y = 0; y < worldHeight; y++) {
		int yIndex = worldWidth * y;
		for (int x = 0; x < worldWidth; x++) {
			if (mat[yIndex + x] == 0) {
			} else if (mat[yIndex + x] == LogicVirusNormal::HEALTHY) {
				info->numTotal++;
				info->numHealthy++;
			} else if (mat[yIndex + x] == LogicVirusNormal::INFECTED) {
				info->numTotal++;
				info->numInfected++;
			} else if (mat[yIndex + x] == LogicVirusNormal::ONSET) {
				info->numTotal++;
				info->numInfected++;
				info->numOnset++;
			} else if (mat[yIndex + x] == LogicVirusNormal::CONTAGIUS) {
				info->numTotal++;
				info->numInfected++;
				info->numContagius++;
			} else if (mat[yIndex + x] == LogicVirusNormal::IMMUNIZED) {
				info->numTotal++;
				info->numHealthy++;
				info->numImmunized++;
			} else {
				printf("coding error : AnalViewVirus\n");
			}
		}
	}
}

void AnalViewVirus::displayInformationGraph()
{
	int index = 0;
	int XonGraph = 0;
	int YonGraph = 0;
	glBegin(GL_LINE_STRIP);
	glColor3dv(COLOR_3D_VIRUS_HEALTHY);
	for (std::deque<ANAL_INFORMATION>::iterator it = analInfoHistory.begin(); it != analInfoHistory.end(); it++) {
		XonGraph = (index * m_graphWidth) / m_NumHistory;
		YonGraph = ((*it).numHealthy * (m_windowHeight - 2 * MARGIN)) / (m_maxGraphY * 1.2);
		glVertex2d(XonGraph + MARGIN, YonGraph + MARGIN);
		index++;
	}
	glEnd();

	index = 0;
	glBegin(GL_LINE_STRIP);
	glColor3dv(COLOR_3D_VIRUS_INFECTED);
	for (std::deque<ANAL_INFORMATION>::iterator it = analInfoHistory.begin(); it != analInfoHistory.end(); it++) {
		XonGraph = (index * m_graphWidth) / m_NumHistory;
		YonGraph = ((*it).numInfected * (m_windowHeight - 2 * MARGIN)) / (m_maxGraphY * 1.2);
		glVertex2d(XonGraph + MARGIN, YonGraph + MARGIN);
		index++;
	}
	glEnd();

	index = 0;
	glBegin(GL_LINE_STRIP);
	glColor3dv(COLOR_3D_VIRUS_ONSET);
	for (std::deque<ANAL_INFORMATION>::iterator it = analInfoHistory.begin(); it != analInfoHistory.end(); it++) {
		XonGraph = (index * m_graphWidth) / m_NumHistory;
		YonGraph = ((*it).numOnset * (m_windowHeight - 2 * MARGIN)) / (m_maxGraphY * 1.2);
		glVertex2d(XonGraph + MARGIN, YonGraph + MARGIN);
		index++;
	}
	glEnd();

	index = 0;
	glBegin(GL_LINE_STRIP);
	glColor3dv(COLOR_3D_VIRUS_CONTAGIUS);
	for (std::deque<ANAL_INFORMATION>::iterator it = analInfoHistory.begin(); it != analInfoHistory.end(); it++) {
		XonGraph = (index * m_graphWidth) / m_NumHistory;
		YonGraph = ((*it).numContagius * (m_windowHeight - 2 * MARGIN)) / (m_maxGraphY * 1.2);
		glVertex2d(XonGraph + MARGIN, YonGraph + MARGIN);
		index++;
	}
	glEnd();

	index = 0;
	glBegin(GL_LINE_STRIP);
	glColor3dv(COLOR_3D_VIRUS_IMMUNIZED);
	for (std::deque<ANAL_INFORMATION>::iterator it = analInfoHistory.begin(); it != analInfoHistory.end(); it++) {
		XonGraph = (index * m_graphWidth) / m_NumHistory;
		YonGraph = ((*it).numImmunized * (m_windowHeight - 2 * MARGIN)) / (m_maxGraphY * 1.2);
		glVertex2d(XonGraph + MARGIN, YonGraph + MARGIN);
		index++;
	}
	glEnd();


}

void AnalViewVirus::displayInformationText(ANAL_INFORMATION* info)
{
	char str[32];

	ILogic::WORLD_INFORMATION worldInfo;
	m_pContext->m_pLogic->getInformation(&worldInfo);

	glColor3dv(COLOR_3D_NORMAL);
	sprintf_s(str, "Days = %d", worldInfo.generation);
	writeTextArea(0, str);

	glColor3dv(COLOR_3D_NORMAL);
	sprintf_s(str, "numTotal = %d", info->numTotal);
	writeTextArea(1, str);

	glColor3dv(COLOR_3D_VIRUS_HEALTHY);
	sprintf_s(str, "numHealthy = %d", info->numHealthy);
	writeTextArea(2, str);

	glColor3dv(COLOR_3D_VIRUS_INFECTED);
	sprintf_s(str, "numInfected = %d", info->numInfected);
	writeTextArea(3, str);

	glColor3dv(COLOR_3D_VIRUS_ONSET);
	sprintf_s(str, "numOnset = %d", info->numOnset);
	writeTextArea(4, str);

	glColor3dv(COLOR_3D_VIRUS_CONTAGIUS);
	sprintf_s(str, "numContagius = %d", info->numContagius);
	writeTextArea(5, str);

	glColor3dv(COLOR_3D_VIRUS_IMMUNIZED);
	sprintf_s(str, "numImmunized = %d", info->numImmunized);
	writeTextArea(6, str);

	glColor3dv(COLOR_3D_NORMAL);
	sprintf_s(str, "SIZE = %d x %d", m_pContext->WORLD_WIDTH, m_pContext->WORLD_HEIGHT);
	writeTextArea(8, str);

	glColor3dv(COLOR_3D_NORMAL);
	sprintf_s(str, "DayPerSec = %3.2lf (%d [msec]", 1000.0 / worldInfo.calcTime, worldInfo.calcTime);
	writeTextArea(9, str);

	glColor3dv(COLOR_3D_NORMAL);
	sprintf_s(str, "FPS = %3.2lf (%d [msec]", 1000.0 / WindowManager::getInstance()->getDrawIntervalMS(), WindowManager::getInstance()->getDrawIntervalMS());
	writeTextArea(10, str);
}
