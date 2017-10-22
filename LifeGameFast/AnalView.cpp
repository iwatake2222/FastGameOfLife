#include "stdafx.h"
#include "AnalView.h"
#include "WindowManager.h"
#include "Values.h"
#include "ControllerView.h"

AnalView::AnalView(WorldContext* pContext)
{
	m_pContext = pContext;
	m_pContext->m_pAnalView = this;
	m_NumHistory = 400;
	initView();
}

AnalView::~AnalView()
{
	glutDestroyWindow(m_windowId);
	WindowManager::getInstance()->unregisterWindow(m_windowId);
	m_pContext->m_pAnalView = NULL;
}

void AnalView::initView()
{
	/* when this constructor is called, parent world view must be selected */
	int parentX = glutGet(GLUT_WINDOW_X);
	int parentY = glutGet(GLUT_WINDOW_Y);
	int parentW = glutGet(GLUT_WINDOW_WIDTH);
	int parentH = glutGet(GLUT_WINDOW_HEIGHT);
	m_windowWidth = parentW;
	m_windowHeight = WINDOW_HEIGHT;
	glutInitWindowPosition(parentX, parentY + parentH);
	glutInitWindowSize(m_windowWidth, m_windowHeight);
	glutInitDisplayMode(GLUT_RGBA);
	m_windowId = glutCreateWindow("Analysis information");
	WindowManager::getInstance()->registerWindow(m_windowId, this);

	m_maxGraphY = 0;
	m_graphWidth = m_windowWidth - TEXT_AREA_WIDTH > 4 * MARGIN ? m_windowWidth - TEXT_AREA_WIDTH - 3 * MARGIN : 0;
}

void AnalView::writeTextArea(int line, char* str)
{
	drawString(MARGIN + m_graphWidth + MARGIN, m_windowHeight - MARGIN - (line + 1) * 18, str);
}

void AnalView::drawString(double x, double y, char* str)
{
	glRasterPos2f(x, y);
	while(*str != '\0') {
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *str);
		str++;
	}
}

void AnalView::drawBackGround(void)
{
	glColor3dv(COLOR_3D_BK);
	glBegin(GL_QUADS);
	/* BG for graph area */
	glVertex2d(MARGIN, MARGIN);
	glVertex2d(MARGIN + m_graphWidth, MARGIN);
	glVertex2d(MARGIN + m_graphWidth, m_windowHeight - MARGIN);
	glVertex2d(MARGIN, m_windowHeight - MARGIN);

	/* BG for text area */
	glVertex2d(MARGIN + m_graphWidth + MARGIN, MARGIN);
	glVertex2d(MARGIN + m_graphWidth + MARGIN + TEXT_AREA_WIDTH, MARGIN);
	glVertex2d(MARGIN + m_graphWidth + MARGIN + TEXT_AREA_WIDTH, m_windowHeight - MARGIN);
	glVertex2d(MARGIN + m_graphWidth + MARGIN, m_windowHeight - MARGIN);
	glEnd();
}

void AnalView::analyseInformation(ANAL_INFORMATION *info)
{
	int worldWidth = m_pContext->WORLD_WIDTH;
	int worldHeight = m_pContext->WORLD_HEIGHT;
//#define DRAW_CELLS_SAFE
#ifdef DRAW_CELLS_SAFE
	int *mat = new int[worldWidth * worldHeight];
	m_pContext->m_pLogic->copyDisplayMat(mat);
#else
	/* without exclusive control for high speed performance */
	int *mat = m_pContext->m_pLogic->getDisplayMat();
#endif

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
#ifdef DRAW_CELLS_SAFE
	delete mat;
#endif
}

void AnalView::displayInformationGraph()
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

void AnalView::displayInformationText(ANAL_INFORMATION* info)
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
	sprintf_s(str, "FPS = %3.2lf", 1000.0 / worldInfo.calcTime);
	writeTextArea(9, str);
}

void AnalView::onUpdate(void)
{
	if (m_intervalCnt++ % ControllerView::getInstance()->m_viewInterval != 0) return;

	glClearColor(0.1f, 0.0f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	glLoadIdentity();
	/* window area is the same as device area [px] */
	glOrtho(0, m_windowWidth, 0, m_windowHeight, -1.0, 1.0);

	drawBackGround();

	ANAL_INFORMATION newInfo = { 0 };
	ILogic::WORLD_INFORMATION worldInfo;
	m_pContext->m_pLogic->getInformation(&worldInfo);
	if (worldInfo.status != 0) {
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

	glFlush();
}


void AnalView::onResize(int w, int h)
{
	glViewport(0, 0, w, h);
	m_windowWidth = w;
	m_windowHeight = h;
	m_graphWidth = m_windowWidth - TEXT_AREA_WIDTH > 4 * MARGIN ? m_windowWidth - TEXT_AREA_WIDTH - 3 * MARGIN : 0;
}

void AnalView::onClick(int button, int state, int x, int y)
{
	if ((GLUT_MIDDLE_BUTTON == button) && (GLUT_DOWN == state)) {
		m_maxGraphY = 0;
	}
}

void AnalView::onDrag(int x, int y)
{

}

void AnalView::onWheel(int wheel_number, int direction, int x, int y)
{
	if (direction == -1) {
		if (m_NumHistory > 20) {
			m_NumHistory-=10;
			for(int i = 0; i < 10 && analInfoHistory.size() != 0; i++) analInfoHistory.pop_front();
		}
	} else {
		if (m_NumHistory < 1000) m_NumHistory++;
	}
}

void AnalView::onKeyboard(unsigned char key, int x, int y)
{
	switch (key) {
	case 'q':
	case 'i':
		delete this;
		break;
	default:
		break;
	}
}
