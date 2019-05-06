#include "Common.h"
#include <stdio.h>
#include <GL/glut.h>
#include <GL/freeglut.h>
#include "AnalViewGroup.h"
#include "WindowManager.h"
#include "LogicGroup.h"

AnalViewGroup::AnalViewGroup(WorldContext* pContext)
	: AnalView(pContext)
{
}


AnalViewGroup::~AnalViewGroup()
{
}

void AnalViewGroup::updateAnalInfo()
{
	ANAL_INFORMATION newInfo = { 0 };
	ILogic::WORLD_INFORMATION worldInfo;
	m_pContext->m_pLogic->getInformation(&worldInfo);

	if (worldInfo.status != 0 || worldInfo.generation != m_lastAnalyzedGeneration) {
		m_lastAnalyzedGeneration = worldInfo.generation;
		/*** Analyse informaiton of the current generation ***/
		analyseInformation(&newInfo);

		/*** Update analysis informaiton history ***/
		m_analInfo = newInfo;
	} else {
		/* logic is not running. use the previous generation's information*/
		newInfo = m_analInfo;
	}

	/*** Display analysis informaiton graph ***/
	if (m_maxGraphY < newInfo.numAlive) m_maxGraphY = newInfo.numAlive;
	displayInformationGraph(&newInfo);

	/*** Display analysis informaiton graph ***/
	displayInformationText(&newInfo);
}

void AnalViewGroup::analyseInformation(ANAL_INFORMATION *info)
{
	int worldWidth = m_pContext->WORLD_WIDTH;
	int worldHeight = m_pContext->WORLD_HEIGHT;
	int *mat = m_pContext->m_pLogic->getDisplayMat();

	for (int y = 0; y < worldHeight; y++) {
		int yIndex = worldWidth * y;
		for (int x = 0; x < worldWidth; x++) {
			if (mat[yIndex + x] == 0) {
			} else if (mat[yIndex + x] <= LogicGroup::PURE_GROUP_A*1.05) {	// 95% <= A <= 100%
				info->numAlive++;
				info->numAliveGroupA_Pure++;
			} else if (mat[yIndex + x] <= LogicGroup::PURE_GROUP_A * 1.4) {	// 60% <= A < 95%
				info->numAlive++;
				info->numAliveGroupA_Quater++;
			} else if (mat[yIndex + x] < LogicGroup::PURE_GROUP_A * 1.6) {	// 40% < A, B < 60%
				info->numAlive++;
				info->numAliveGroup_Half++;
			} else if (mat[yIndex + x] < LogicGroup::PURE_GROUP_B*0.95) {	// 60% <= B < 95%
				info->numAlive++;
				info->numAliveGroupB_Quater++;
			} else if (mat[yIndex + x] <= LogicGroup::PURE_GROUP_B) {	// 95% <= B <= 100%
				info->numAlive++;
				info->numAliveGroupB_Pure++;
			} else {
				printf("coding error : AnalViewGroup\n");
			}
		}
	}
}

void AnalViewGroup::displayInformationGraph(ANAL_INFORMATION *info)
{
	if (info->numAlive == 0) return;

	int xMin = MARGIN;
	int xMax = m_graphWidth;
	int yMin = MARGIN * 1 + MARGIN;
	int yMax = m_graphHeight - MARGIN * 1;

	int xA100 = (m_graphWidth * info->numAliveGroupA_Pure) / info->numAlive + xMin;
	int xA75 = (m_graphWidth * info->numAliveGroupA_Quater) / info->numAlive + xA100;
	int xHalf = (m_graphWidth * info->numAliveGroup_Half) / info->numAlive + xA75;
	int xB75 = (m_graphWidth * info->numAliveGroupB_Quater) / info->numAlive + xHalf;
	int xB100 = (m_graphWidth * info->numAliveGroupB_Pure) / info->numAlive + xB75;

	glBegin(GL_QUADS);

	glColor3dv(COLOR_3D_ALIVE_GROUP_A);
	glVertex2d(xMin, yMin);
	glVertex2d(xA100, yMin);
	glVertex2d(xA100, yMax);
	glVertex2d(xMin, yMax);

	glColor3dv(COLOR_3D_ALIVE_GROUP_A75);
	glVertex2d(xA100, yMin);
	glVertex2d(xA75, yMin);
	glVertex2d(xA75, yMax);
	glVertex2d(xA100, yMax);

	glColor3dv(COLOR_3D_ALIVE_GROUP_HALF);
	glVertex2d(xA75, yMin);
	glVertex2d(xHalf, yMin);
	glVertex2d(xHalf, yMax);
	glVertex2d(xA75, yMax);

	glColor3dv(COLOR_3D_ALIVE_GROUP_B75);
	glVertex2d(xHalf, yMin);
	glVertex2d(xB75, yMin);
	glVertex2d(xB75, yMax);
	glVertex2d(xHalf, yMax);

	glColor3dv(COLOR_3D_ALIVE_GROUP_B);
	glVertex2d(xB75, yMin);
	glVertex2d(xB100, yMin);
	glVertex2d(xB100, yMax);
	glVertex2d(xB75, yMax);

	glEnd();
}

void AnalViewGroup::displayInformationText(ANAL_INFORMATION* info)
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

	glColor3dv(COLOR_3D_ALIVE_GROUP_A);
	sprintf_s(str, "Pure GroupA = %d", info->numAliveGroupA_Pure);
	writeTextArea(2, str);

	glColor3dv(COLOR_3D_ALIVE_GROUP_A75);
	sprintf_s(str, "75%% GroupA = %d", info->numAliveGroupA_Quater);
	writeTextArea(3, str);

	glColor3dv(COLOR_3D_ALIVE_GROUP_HALF);
	sprintf_s(str, "HALF = %d", info->numAliveGroup_Half);
	writeTextArea(4, str);

	glColor3dv(COLOR_3D_ALIVE_GROUP_B75);
	sprintf_s(str, "75%% GroupB = %d", info->numAliveGroupB_Quater);
	writeTextArea(5, str);

	glColor3dv(COLOR_3D_ALIVE_GROUP_B);
	sprintf_s(str, "Pure GroupB = %d", info->numAliveGroupB_Pure);
	writeTextArea(6, str);

	glColor3dv(COLOR_3D_NORMAL);
	sprintf_s(str, "SIZE = %d x %d", m_pContext->WORLD_WIDTH, m_pContext->WORLD_HEIGHT);
	writeTextArea(7, str);

	glColor3dv(COLOR_3D_NORMAL);
	sprintf_s(str, "GPS = %3.2lf (%d [msec]", 1000.0 / worldInfo.calcTime, worldInfo.calcTime);
	writeTextArea(8, str);

	glColor3dv(COLOR_3D_NORMAL);
	sprintf_s(str, "FPS = %3.2lf (%d [msec]", 1000.0 / WindowManager::getInstance()->getDrawIntervalMS(), WindowManager::getInstance()->getDrawIntervalMS());
	writeTextArea(9, str);
}
