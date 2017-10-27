#include "stdafx.h"
#include "WindowManager.h"

int WindowManager::m_drawIntervalMS;

WindowManager::WindowManager()
{
	WindowManager::m_drawIntervalMS = 0;
}

WindowManager::~WindowManager()
{
}

WindowManager* WindowManager::getInstance()
{
	static WindowManager s_windowManager;
	return &s_windowManager;
}

void WindowManager::init()
{
	int argc = 0;
	glutInit(&argc, NULL);
	glutIdleFunc(idle);
}

void WindowManager::startLoop()
{
	glutMainLoop();
}

void WindowManager::registerWindow(int windowId, IView* pWorldView)
{
	m_viewMap[windowId] = pWorldView;
	// need to register callback functions for each window(display)
	glutDisplayFunc(onUpdateWrapper);
	glutReshapeFunc(onResizeWrapper);
	glutMouseFunc(onClickWrapper);
	glutMotionFunc(onDragWrapper);
	glutMouseWheelFunc(onWheelWrapper);
	glutKeyboardFunc(onKeyboardWrapper);
}

void WindowManager::unregisterWindow(int windowId)
{
	//glutSetWindow(windowId);
	//glutDisplayFunc(NULL);
	//glutReshapeFunc(NULL);
	//glutMouseFunc(NULL);
	//glutMotionFunc(NULL);
	//glutMouseWheelFunc(NULL);
	//glutKeyboardFunc(NULL);
	m_viewMap.erase(windowId);
}

int WindowManager::getDrawIntervalMS()
{
	return WindowManager::m_drawIntervalMS;
}

void WindowManager::idle(void)
{
	for (std::map<int, IView *>::iterator it = getInstance()->m_viewMap.begin(); it != getInstance()->m_viewMap.end(); ++it) {
		glutSetWindow(it->first);
		glutPostRedisplay();
	}
}

void WindowManager::onUpdateWrapper()
{
	int windowId = glutGetWindow();
	getInstance()->m_viewMap[windowId]->onUpdate();

	static std::chrono::system_clock::time_point  timePrevious;
	std::chrono::system_clock::time_point  timeNow;;
	timeNow = std::chrono::system_clock::now();
	WindowManager::m_drawIntervalMS = std::chrono::duration_cast<std::chrono::milliseconds>(timeNow - timePrevious).count();
	//printf("fpsDraw = %lf\n", 1000.0 / m_drawIntervalMS);
	timePrevious = timeNow;
}
void WindowManager::onResizeWrapper(int w, int h)
{
	int windowId = glutGetWindow();
	getInstance()->m_viewMap[windowId]->onResize(w, h);
}
void WindowManager::onClickWrapper(int button, int state, int x, int y)
{
	int windowId = glutGetWindow();
	getInstance()->m_viewMap[windowId]->onClick(button, state, x, y);
}
void WindowManager::onDragWrapper(int x, int y)
{
	int windowId = glutGetWindow();
	getInstance()->m_viewMap[windowId]->onDrag(x, y);
}
void WindowManager::onWheelWrapper(int wheel_number, int direction, int x, int y)
{
	int windowId = glutGetWindow();
	getInstance()->m_viewMap[windowId]->onWheel(wheel_number, direction, x, y);
}
void WindowManager::onKeyboardWrapper(unsigned char key, int x, int y)
{
	int windowId = glutGetWindow();
	getInstance()->m_viewMap[windowId]->onKeyboard(key, x, y);
}
