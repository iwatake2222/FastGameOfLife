#include <stdio.h>
#include <future>
#include <GL/glut.h>
#include <gl/freeglut.h>
#include "WorldView.h"
#include "WindowManager.h"
#include "ControllerView.h"
#include "AnalViewAge.h"
#include "Values.h"
#include "FileAccessor.h"

WorldView::WorldView(WorldContext* pContext, int windowX, int windowY, int windowWidth, int windowHeight)
{
	/* fixed during lifespan */
	WORLD_WIDTH = pContext->WORLD_WIDTH;
	WORLD_HEIGHT = pContext->WORLD_HEIGHT;
	WORLD_WIDTH_MARGIN = WORLD_WIDTH  * 0.05;
	WORLD_HEIGHT_MARGIN = WORLD_HEIGHT * 0.05;
	m_pContext = pContext;

	/* initial window size */
	m_windowWidth = windowWidth;
	m_windowHeight = windowHeight;

	m_worldVisibleX0 = -WORLD_WIDTH_MARGIN;
	m_worldVisibleX1 = WORLD_WIDTH + WORLD_WIDTH_MARGIN;
	m_worldVisibleY0 = -WORLD_HEIGHT;
	m_worldVisibleY1 = WORLD_HEIGHT + WORLD_HEIGHT_MARGIN;

	m_previousLDragXWorld = INVALID_NUM;
	m_previousLDragYWorld = INVALID_NUM;
	m_previousRDragX = INVALID_NUM;
	m_previousRDragY = INVALID_NUM;

	glutInitWindowPosition(windowX, windowY);
	glutInitWindowSize(m_windowWidth, m_windowHeight);
	initOpenGL();
	initView();
}

WorldView::~WorldView()
{
	glutDestroyWindow(m_windowId);
	WindowManager::getInstance()->unregisterWindow(m_windowId);
}


void WorldView::initOpenGL()
{
	glutInitDisplayMode(GLUT_RGBA);
	m_windowId = glutCreateWindow("The Game of Life");
	WindowManager::getInstance()->registerWindow(m_windowId, this);
}

void WorldView::initView()
{
	glClearColor(0.1f, 0.0f, 0.3f, 1.0f);

	/* diaplay the entire world area */
	/* the aspect of worldVisible must be the same as that of window */
	double worldVisibleWidth;
	double worldVisibleHeight;
	if (m_windowWidth > m_windowHeight) {
		worldVisibleHeight = WORLD_HEIGHT + WORLD_HEIGHT_MARGIN * 2;
		worldVisibleWidth = (worldVisibleHeight * m_windowWidth) / m_windowHeight;
	} else {
		worldVisibleWidth = WORLD_WIDTH + WORLD_WIDTH_MARGIN * 2;
		worldVisibleHeight = (worldVisibleWidth * m_windowHeight) / m_windowWidth;
	}
	m_worldVisibleX0 = WORLD_WIDTH / 2 - worldVisibleWidth / 2;
	m_worldVisibleX1 = WORLD_WIDTH / 2 + worldVisibleWidth / 2;
	m_worldVisibleY0 = WORLD_HEIGHT / 2 - worldVisibleHeight / 2;
	m_worldVisibleY1 = WORLD_HEIGHT / 2 + worldVisibleHeight / 2;

	glLoadIdentity();
	glOrtho(m_worldVisibleX0, m_worldVisibleX1, m_worldVisibleY0, m_worldVisibleY1, -1.0, 1.0);
	glFlush();
}

void WorldView::drawDeadCellsAll()
{
	glColor3dv(COLOR_3D_DEAD);
	glBegin(GL_QUADS);
	glVertex2d(0, 0);
	glVertex2d(WORLD_WIDTH, 0);
	glVertex2d(WORLD_WIDTH, WORLD_HEIGHT);
	glVertex2d(0, WORLD_HEIGHT);
	glEnd();
}

void WorldView::drawGrid()
{
	/* calculate the best interval for thinning grid line according to current view size */
	const static double MAX_LINE_DENSITY = 0.1;		// e.g. when window width = 640px, the number of grid number is at most 64
	double intervalX = 1;
	double intervalY = 1;
	double interval;
	double currentVisibleWidth = m_worldVisibleX1 - m_worldVisibleX0;
	double currentVisibleHeight = m_worldVisibleY1 - m_worldVisibleY0;
	if (currentVisibleWidth < m_windowWidth * MAX_LINE_DENSITY) {
		intervalX = 1;
	} else if (currentVisibleWidth < m_windowWidth) {
		intervalX = 10;
	} else if (currentVisibleWidth < m_windowWidth * 10) {
		intervalX = 100;
	} else {
		return;	/* no grid*/
	}
	if (currentVisibleHeight < m_windowHeight * MAX_LINE_DENSITY) {
		intervalY = 1;
	} else if (currentVisibleHeight * m_windowHeight) {
		intervalY = 10;
	} else if (currentVisibleHeight * m_windowHeight * 10) {
		intervalY = 100;
	} else {
		return;	/* no grid*/
	}
	interval = fmax(intervalX, intervalY);

	glColor3dv(COLOR_3D_GRID);
	glBegin(GL_LINES);
	/* draw vertical lines*/
	for (double x = 0; x <= WORLD_WIDTH; x += interval) {
		glVertex2d(x, -WORLD_HEIGHT_MARGIN);
		glVertex2d(x, WORLD_HEIGHT + WORLD_HEIGHT_MARGIN);
	}
	/* draw horizontal lines*/
	for (double y = 0; y <= WORLD_HEIGHT; y += interval) {
		glVertex2d(-WORLD_WIDTH_MARGIN, y);
		glVertex2d(WORLD_WIDTH + WORLD_WIDTH_MARGIN, y);
	}
	glEnd();
}

void WorldView::drawCells()
{
	/*** for lighter draw ***/
	/* don't draw 1> cells in 1 pixel */
	/* also thinned out by user setting */
	int intervalX = ceil((m_worldVisibleX1 - m_worldVisibleX0) / m_windowWidth);
	int intervalY = ceil((m_worldVisibleY1 - m_worldVisibleY0) / m_windowHeight);
	if (intervalX < 1)intervalX = 1;
	if (intervalY < 1)intervalY = 1;
	intervalX *= ControllerView::getInstance()->m_drawInterval;
	intervalY *= ControllerView::getInstance()->m_drawInterval;
	if (intervalX % 2 == 0)intervalX += 1;	// avoid even count because some specific pattern may not be drawn
	if (intervalY % 2 == 0)intervalY += 1;	// avoid even count because some specific pattern may not be drawn

	int *mat = m_pContext->m_pLogic->getDisplayMat();

	glBegin(GL_QUADS);
	/* draw only visible area */
	int ymin = (int)fmax(m_worldVisibleY0, 0);
	int ymax = (int)fmin(m_worldVisibleY1, WORLD_HEIGHT - 1);
	int xmin = (int)fmax(m_worldVisibleX0, 0);
	int xmax = (int)fmin(m_worldVisibleX1, WORLD_WIDTH - 1);
	for (int y = ymin; y <= ymax; y += intervalY) {
		int yIndex = WORLD_WIDTH * y;
		for (int x = xmin; x <= xmax; x += intervalX) {
			if (mat[yIndex + x] != 0) {	// draw alive cells only
				double color3d[3];
				m_pContext->m_pLogic->convertDisplay2Color(mat[yIndex + x], color3d);
				glColor3dv(color3d);
				glVertex2d(x + 0, y + 0);
				glVertex2d(x + intervalX, y + 0);
				glVertex2d(x + intervalX, y + intervalY);
				glVertex2d(x + 0, y + intervalY);
			}
		}
	}
	glEnd();
}

void WorldView::onUpdate(void)
{
	glClear(GL_COLOR_BUFFER_BIT);

	glLoadIdentity();
	glOrtho(m_worldVisibleX0, m_worldVisibleX1, m_worldVisibleY0, m_worldVisibleY1, -1.0, 1.0);

	drawDeadCellsAll();
	drawCells();
	drawGrid();

	glFlush();

	std::this_thread::sleep_for(std::chrono::milliseconds(1));
#if 0
	static std::chrono::system_clock::time_point  timePrevious;
	std::chrono::system_clock::time_point  timeNow;;
	timeNow = std::chrono::system_clock::now();
	int timeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(timeNow - timePrevious).count();
	//printf("fpsDraw = %lf\n", 1000.0 / timeElapsed);
	timePrevious = timeNow;
#endif
}

void WorldView::onResize(int w, int h)
{
	glViewport(0, 0, w, h);

	double resizeRatioWidth = (double)w / m_windowWidth;
	double resizeRatioHeight = (double)h / m_windowHeight;

	/* the aspect of worldVisible must be the same as that of window */
	/* stretch visible area in the same portion as window */
	double visibleWidth = m_worldVisibleX1 - m_worldVisibleX0;
	double visibleHeight = m_worldVisibleY1 - m_worldVisibleY0;
	visibleWidth *= resizeRatioWidth;
	visibleHeight *= resizeRatioHeight;
	m_worldVisibleX1 = m_worldVisibleX0 + visibleWidth;
	m_worldVisibleY1 = m_worldVisibleY0 + visibleHeight;

	m_windowWidth = w;
	m_windowHeight = h;

	ControllerView::getInstance()->setCurrentWorldContext(m_pContext);
}

void WorldView::convertPosWindow2World(int x, int y, int* worldX, int* worldY)
{
	double windowNormX = (double)x / m_windowWidth;
	double windowNormY = (double)(m_windowHeight - y) / m_windowHeight;
	*worldX = (int)floor(m_worldVisibleX0 + (m_worldVisibleX1 - m_worldVisibleX0) * windowNormX);
	*worldY = (int)floor(m_worldVisibleY0 + (m_worldVisibleY1 - m_worldVisibleY0) * windowNormY);
}

void WorldView::onClick(int button, int state, int x, int y)
{
	if ((GLUT_LEFT_BUTTON == button) && (GLUT_DOWN == state)) {
		/* the beginnig of drag */
		int worldX = 0, worldY = 0;
		convertPosWindow2World(x, y, &m_previousLDragXWorld, &m_previousLDragYWorld);
	} else 	if ((GLUT_LEFT_BUTTON == button) && (GLUT_UP == state)) {
		/* the end of drag or just clicked */
		int worldX = 0, worldY = 0;
		convertPosWindow2World(x, y, &worldX, &worldY);
		if ((m_previousLDragXWorld == worldX) && (m_previousLDragYWorld == worldY)) {
			/* on clicked, toggle the clicked cell */
			m_pContext->m_pLogic->toggleCell(worldX, worldY, ControllerView::getInstance()->m_prm1);
			glutPostRedisplay();
		}
		m_previousLDragXWorld = INVALID_NUM;
		m_previousLDragYWorld = INVALID_NUM;
	}

	if ((GLUT_RIGHT_BUTTON == button) && (GLUT_DOWN == state)) {
		/* the beginnig of drag */
		m_previousRDragX = x;
		m_previousRDragY = y;
	} else 	if ((GLUT_RIGHT_BUTTON == button) && (GLUT_UP == state)) {
		/* the end of drag */
		m_previousRDragX = INVALID_NUM;
		m_previousRDragY = INVALID_NUM;
	}

	if ((GLUT_MIDDLE_BUTTON == button) && (GLUT_DOWN == state)) {
		initView();
	}

	ControllerView::getInstance()->setCurrentWorldContext(m_pContext);
}

void WorldView::onDrag(int x, int y)
{
	if (m_previousLDragXWorld != INVALID_NUM) {
		/* left button drag to draw cells */
		int worldX = 0, worldY = 0;
		convertPosWindow2World(x, y, &worldX, &worldY);
		m_pContext->m_pLogic->setCell(worldX, worldY, ControllerView::getInstance()->m_prm1);
		glutPostRedisplay();
	}

	if (m_previousRDragX != INVALID_NUM) {
		/* right button drag to change view position */
		double currentVisibleWidth = m_worldVisibleX1 - m_worldVisibleX0;
		double currentVisibleHeight = m_worldVisibleY1 - m_worldVisibleY0;
		double worldDeltaX = -(x - m_previousRDragX) * currentVisibleWidth / m_windowWidth;
		double worldDeltaY = (y - m_previousRDragY) * currentVisibleHeight / m_windowHeight;
		//if ((worldVisibleX0 + worldDeltaX >= -WORLD_WIDTH_MARGIN) &&
		//	(worldVisibleX1 + worldDeltaX <= WORLD_WIDTH + WORLD_WIDTH_MARGIN)) {
		m_worldVisibleX0 += worldDeltaX;
		m_worldVisibleX1 += worldDeltaX;
		//}
		//if ((worldVisibleY0 + worldDeltaY >= -WORLD_HEIGHT_MARGIN) &&
		//	(worldVisibleY1 + worldDeltaY <= WORLD_HEIGHT + WORLD_HEIGHT_MARGIN)) {
		m_worldVisibleY0 += worldDeltaY;
		m_worldVisibleY1 += worldDeltaY;
		//}
		glutPostRedisplay();
		m_previousRDragX = x;
		m_previousRDragY = y;
	}
}

void WorldView::onWheel(int wheel_number, int direction, int x, int y)
{
	/* calculate the best zoom speed according to current zoom position */
	/*  - 1. zoom for X axis */
	/*  - 2. adjust Y axis to keep the original aspect */
	/* ToDo: should use bigger or smaller one */
	double currentVisibleWidth = m_worldVisibleX1 - m_worldVisibleX0;
	double currentVisibleHeight = m_worldVisibleY1 - m_worldVisibleY0;
	double zoomSpeed = 1;
	if (currentVisibleWidth < 100) {
		zoomSpeed = 2;
	} else if (currentVisibleWidth < 200) {
		zoomSpeed = 4;
	} else if (currentVisibleWidth < 400) {
		zoomSpeed = 8;
	} else {
		zoomSpeed = WORLD_WIDTH / 50.0;
	}

	double worldVisibleX0Candidate;
	double worldVisibleX1Candidate;
	double worldVisibleY0Candidate;
	double worldVisibleY1Candidate;

	if (direction == -1) {
		/* zoom out */
		worldVisibleX0Candidate = m_worldVisibleX0 - zoomSpeed;
		worldVisibleX1Candidate = m_worldVisibleX1 + zoomSpeed;
	} else {
		/* zoom in */
		worldVisibleX0Candidate = m_worldVisibleX0 + zoomSpeed;
		worldVisibleX1Candidate = m_worldVisibleX1 - zoomSpeed;
	}
	double newVisibleHeight = (worldVisibleX1Candidate - worldVisibleX0Candidate) * currentVisibleHeight / currentVisibleWidth;

	worldVisibleY0Candidate = (m_worldVisibleY1 + m_worldVisibleY0) / 2 - newVisibleHeight / 2;
	worldVisibleY1Candidate = (m_worldVisibleY1 + m_worldVisibleY0) / 2 + newVisibleHeight / 2;

	if (direction == -1) {
		/* zoom out */
		//if ((worldVisibleX1Candidate - worldVisibleX0Candidate <= WORLD_WIDTH + WORLD_WIDTH_MARGIN * 2) ||
		//	(newVisibleHeight <= WORLD_HEIGHT + WORLD_HEIGHT_MARGIN * 2)) {
		m_worldVisibleX0 = worldVisibleX0Candidate;
		m_worldVisibleX1 = worldVisibleX1Candidate;
		m_worldVisibleY0 = worldVisibleY0Candidate;
		m_worldVisibleY1 = worldVisibleY1Candidate;
		glutPostRedisplay();
		//} else {
		//	//printf("too zoom out\n");
		//}
	} else {
		/* zoom in */
		const static int MINIMUM_ZOOM = 10;
		if ((worldVisibleX1Candidate - worldVisibleX0Candidate >= MINIMUM_ZOOM) &&
			(newVisibleHeight >= MINIMUM_ZOOM)) {
			m_worldVisibleX0 = worldVisibleX0Candidate;
			m_worldVisibleX1 = worldVisibleX1Candidate;
			m_worldVisibleY0 = worldVisibleY0Candidate;
			m_worldVisibleY1 = worldVisibleY1Candidate;
			glutPostRedisplay();
		} else {
			//printf("too zoom out\n");
		}
	}
}

void WorldView::onKeyboard(unsigned char key, int x, int y)
{
	int density;
	switch (key) {
	case 'a':
		density = ControllerView::getInstance()->m_density;
		m_pContext->m_pLogic->allocCells((int)m_worldVisibleX0, (int)m_worldVisibleX1, (int)m_worldVisibleY0, (int)m_worldVisibleY1, density, ControllerView::getInstance()->m_prm1);
		glutPostRedisplay();
		break;
	case 'c':
		m_pContext->m_pLogic->allocCells((int)m_worldVisibleX0, (int)m_worldVisibleX1, (int)m_worldVisibleY0, (int)m_worldVisibleY1, 0);
		glutPostRedisplay();
		break;
	case 'A':
		density = ControllerView::getInstance()->m_density;
		m_pContext->m_pLogic->allocCells(0, WORLD_WIDTH, 0, WORLD_HEIGHT, density, ControllerView::getInstance()->m_prm1);
		m_pContext->m_pLogic->resetInformation();
		glutPostRedisplay();
		break;
	case 'C':
		m_pContext->m_pLogic->allocCells(0, WORLD_WIDTH, 0, WORLD_HEIGHT, 0);
		m_pContext->m_pLogic->resetInformation();
		glutPostRedisplay();
		break;
	case 'p':
		m_pContext->m_pLogic->toggleRun();
		break;
	case '1':
		m_pContext->m_pLogic->stepRun();
		break;
	case 'i':
		if (m_pContext->m_pAnalView == NULL) {
			AnalView::createAppropreateAnalView(m_pContext);
		} else {
			delete m_pContext->m_pAnalView;
		}
		break;
	case 'g':
		new WorldContext();
		break;
	case 'q':
		ControllerView::getInstance()->setCurrentWorldContext(NULL);
		delete m_pContext;
		break;
	case 'l':
	{
		/* load pettern file */
		char path[255];
		int patternWidth, patternHeight;	// pattern size
		int patternOffsetX, patternOffsetY;	// position in pattern
		int prm;
		FileAccessor::getFilepath(path);
		if (FileAccessor::startReadingPattern(path, &patternWidth, &patternHeight)) {
			printf("Pattern size(%d x %d)\n", patternWidth, patternHeight);
			while (FileAccessor::readPattern(&patternOffsetX, &patternOffsetY, &prm)) {
				int x = (m_worldVisibleX0 + m_worldVisibleX1) / 2 + patternOffsetX - patternWidth/2;
				int y = (m_worldVisibleY0 + m_worldVisibleY1) / 2 - (patternOffsetY - patternHeight / 2) - 1;
				if (prm != 0) {
					m_pContext->m_pLogic->setCell(x, y, ControllerView::getInstance()->m_prm1);
				} else {
					m_pContext->m_pLogic->clearCell(x, y);
				}
			}
			FileAccessor::stop();
		}
		((AnalView*)m_pContext->m_pAnalView)->refresh();	// todo: without cast
		break;
	}
	case 's':
	{
		/* save pettern file */
		char path[255];
		FileAccessor::getFilepath(path);
		if (FileAccessor::startWritingPattern(path)) {
			int x0 = m_worldVisibleX0 > 0 ? m_worldVisibleX0 : 0;
			int x1 = m_worldVisibleX1 < WORLD_WIDTH ? m_worldVisibleX1 : WORLD_WIDTH - 1;
			int y0 = m_worldVisibleY0 > 0 ? m_worldVisibleY0 : 0;
			int y1 = m_worldVisibleY1 < WORLD_HEIGHT ? m_worldVisibleY1 : WORLD_HEIGHT - 1;

			bool isNewline = false;
			int *mat = m_pContext->m_pLogic->getDisplayMat();
			for (int y = y1; y >= y0; y--) {
				int yIndex = WORLD_WIDTH * y;
				for (int x = x0; x <= x1; x++) {
					int prm = mat[yIndex + x];
					FileAccessor::writePattern(prm, isNewline);
					isNewline = false;
				}
				isNewline = true;
			}
			FileAccessor::stop();
		}
		break;
	}
	default:
		break;
	}
}

