#include "stdafx.h"
#include "AnalView.h"
#include "WindowManager.h"
#include "Values.h"
#include "ControllerView.h"
#include "AnalViewAge.h"
#include "AnalViewGroup.h"

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
	m_graphHeight = m_windowHeight - MARGIN * 2;
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



void AnalView::onUpdate(void)
{
	if (m_intervalCnt++ % ControllerView::getInstance()->m_drawInterval != 0) return;

	glClearColor(0.1f, 0.0f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	glLoadIdentity();
	/* window area is the same as device area [px] */
	glOrtho(0, m_windowWidth, 0, m_windowHeight, -1.0, 1.0);

	drawBackGround();

	UpdateAnalInfo();

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
	//if (direction == -1) {
	//	if (m_NumHistory > 20) {
	//		m_NumHistory-=10;
	//		for(int i = 0; i < 10 && analInfoHistory.size() != 0; i++) analInfoHistory.pop_front();
	//	}
	//} else {
	//	if (m_NumHistory < 1000) m_NumHistory++;
	//}
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

IView* AnalView::createAppropreateAnalView(WorldContext* pContext)
{
	switch (pContext->m_algorithm) {
	case ALGORITHM_NORMAL:
	case ALGORITHM_NORMAL_MP:
	case ALGORITHM_NORMAL_CUDA:
	case ALGORITHM_NORMAL_NON_TORUS:
	case ALGORITHM_NORMAL_NON_TORUS_MP:
	default:
		return new AnalViewAge(pContext);
	case ALGORITHM_GROUP:
		return new AnalViewGroup(pContext);
	}
}
