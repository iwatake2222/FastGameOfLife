#include "Common.h"
#include <stdio.h>
#include <GL/glut.h>
#include <gl/freeglut.h>
#include "AnalViewAge.h"
#include "WindowManager.h"

AnalViewAge::AnalViewAge(WorldContext* pContext)
	: AnalView(pContext)
{
}


AnalViewAge::~AnalViewAge()
{
}

void AnalViewAge:: updateAnalInfo()
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
			analInfoHistory.pop_front();
		}
	} else {
		/* logic is not running. use the previous generation's information*/
		if (analInfoHistory.size() > 0) {
			newInfo = analInfoHistory.back();
		}
	}

	/*** Display analysis informaiton graph ***/
	if (m_maxGraphY < newInfo.numAlive) m_maxGraphY = newInfo.numAlive;
	displayInformationGraph();

	/*** Display analysis informaiton graph ***/
	displayInformationText(&newInfo);
}

void AnalViewAge::analyseInformation(ANAL_INFORMATION *info)
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
			} else if (generation == 0 || (mat[yIndex + x] * 100) / generation < 20) {
				info->numAlive++;
				info->numAlive0++;
			} else if ((mat[yIndex + x] * 100) / generation < 40) {
				info->numAlive++;
				info->numAlive1++;
			} else if ((mat[yIndex + x] * 100) / generation < 60) {
				info->numAlive++;
				info->numAlive2++;
			} else if ((mat[yIndex + x] * 100) / generation < 80) {
				info->numAlive++;
				info->numAlive3++;
			} else {
				info->numAlive++;
				info->numAlive4++;
			}
		}
	}
}

void AnalViewAge::displayInformationGraph()
{
	int index = 0;
	int XonGraph = 0;
	int YonGraph = 0;
	glBegin(GL_LINE_STRIP);
	glColor3dv(COLOR_3D_ALIVE);
	for (std::deque<ANAL_INFORMATION>::iterator it = analInfoHistory.begin(); it != analInfoHistory.end(); it++) {
		XonGraph = (index * m_graphWidth) / m_NumHistory;
		YonGraph = ((*it).numAlive * (m_windowHeight - 2 * MARGIN)) / (m_maxGraphY * 1.2);
		glVertex2d(XonGraph + MARGIN, YonGraph + MARGIN);
		index++;
	}
	glEnd();

	index = 0;
	glBegin(GL_LINE_STRIP);
	glColor3dv(COLOR_3D_ALIVE0);
	for (std::deque<ANAL_INFORMATION>::iterator it = analInfoHistory.begin(); it != analInfoHistory.end(); it++) {
		XonGraph = (index * m_graphWidth) / m_NumHistory;
		YonGraph = ((*it).numAlive0 * (m_windowHeight - 2 * MARGIN)) / (m_maxGraphY * 1.2);
		glVertex2d(XonGraph + MARGIN, YonGraph + MARGIN);
		index++;
	}
	glEnd();

	index = 0;
	glBegin(GL_LINE_STRIP);
	glColor3dv(COLOR_3D_ALIVE1);
	for (std::deque<ANAL_INFORMATION>::iterator it = analInfoHistory.begin(); it != analInfoHistory.end(); it++) {
		XonGraph = (index * m_graphWidth) / m_NumHistory;
		YonGraph = ((*it).numAlive1 * (m_windowHeight - 2 * MARGIN)) / (m_maxGraphY * 1.2);
		glVertex2d(XonGraph + MARGIN, YonGraph + MARGIN);
		index++;
	}
	glEnd();

	index = 0;
	glBegin(GL_LINE_STRIP);
	glColor3dv(COLOR_3D_ALIVE2);
	for (std::deque<ANAL_INFORMATION>::iterator it = analInfoHistory.begin(); it != analInfoHistory.end(); it++) {
		XonGraph = (index * m_graphWidth) / m_NumHistory;
		YonGraph = ((*it).numAlive2 * (m_windowHeight - 2 * MARGIN)) / (m_maxGraphY * 1.2);
		glVertex2d(XonGraph + MARGIN, YonGraph + MARGIN);
		index++;
	}
	glEnd();

	index = 0;
	glBegin(GL_LINE_STRIP);
	glColor3dv(COLOR_3D_ALIVE3);
	for (std::deque<ANAL_INFORMATION>::iterator it = analInfoHistory.begin(); it != analInfoHistory.end(); it++) {
		XonGraph = (index * m_graphWidth) / m_NumHistory;
		YonGraph = ((*it).numAlive3 * (m_windowHeight - 2 * MARGIN)) / (m_maxGraphY * 1.2);
		glVertex2d(XonGraph + MARGIN, YonGraph + MARGIN);
		index++;
	}
	glEnd();

	index = 0;
	glBegin(GL_LINE_STRIP);
	glColor3dv(COLOR_3D_ALIVE4);
	for (std::deque<ANAL_INFORMATION>::iterator it = analInfoHistory.begin(); it != analInfoHistory.end(); it++) {
		XonGraph = (index * m_graphWidth) / m_NumHistory;
		YonGraph = ((*it).numAlive4 * (m_windowHeight - 2 * MARGIN)) / (m_maxGraphY * 1.2);
		glVertex2d(XonGraph + MARGIN, YonGraph + MARGIN);
		index++;
	}
	glEnd();
}

void AnalViewAge::displayInformationText(ANAL_INFORMATION* info)
{
	char str[32];

	ILogic::WORLD_INFORMATION worldInfo;
	m_pContext->m_pLogic->getInformation(&worldInfo);

	glColor3dv(COLOR_3D_NORMAL);
	sprintf_s(str, "GENERATION = %d", worldInfo.generation);
	writeTextArea(0, str);

	glColor3dv(COLOR_3D_ALIVE);
	sprintf_s(str, "TOTAL = %d", info->numAlive);
	writeTextArea(1, str);

	glColor3dv(COLOR_3D_ALIVE0);
	sprintf_s(str, " 0 - 20 = %d", info->numAlive0);
	writeTextArea(2, str);

	glColor3dv(COLOR_3D_ALIVE1);
	sprintf_s(str, "20 - 40 = %d", info->numAlive1);
	writeTextArea(3, str);

	glColor3dv(COLOR_3D_ALIVE2);
	sprintf_s(str, "40 - 60 = %d", info->numAlive2);
	writeTextArea(4, str);

	glColor3dv(COLOR_3D_ALIVE3);
	sprintf_s(str, "60 - 80 = %d", info->numAlive3);
	writeTextArea(5, str);

	glColor3dv(COLOR_3D_ALIVE4);
	sprintf_s(str, "80 -100 = %d", info->numAlive4);
	writeTextArea(6, str);

	glColor3dv(COLOR_3D_NORMAL);
	sprintf_s(str, "SIZE = %d x %d", m_pContext->WORLD_WIDTH, m_pContext->WORLD_HEIGHT);
	writeTextArea(8, str);

	glColor3dv(COLOR_3D_NORMAL);
	sprintf_s(str, "GPS = %3.2lf (%d [msec]", 1000.0 / worldInfo.calcTime, worldInfo.calcTime);
	writeTextArea(9, str);

	glColor3dv(COLOR_3D_NORMAL);
	sprintf_s(str, "FPS = %3.2lf (%d [msec]", 1000.0 / WindowManager::getInstance()->getDrawIntervalMS(), WindowManager::getInstance()->getDrawIntervalMS());
	writeTextArea(10, str);
}
