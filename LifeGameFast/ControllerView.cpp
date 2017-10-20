#include "stdafx.h"
#include "ControllerView.h"
#include "WorldContextManager.h"

ControllerView::ControllerView()
{
	m_pSelectedWorldView = NULL;

	m_worldWidth = 512;
	m_worldHeight = 512;
	m_worldAlgorithm = ALGORITHM::NORMAL;
	m_density = 20;
	m_prm1 = 0;
	m_prm2 = 0;
	m_prm3 = 0;
	m_prm4 = 0;

	initLibrary();
	initUI();
}

ControllerView::~ControllerView()
{
	TwTerminate();
	glutDestroyWindow(m_windowId);
}

ControllerView* ControllerView::getInstance()
{
	static ControllerView s_controllerView;
	return &s_controllerView;
}

void ControllerView::setCurrentWorldView(IView* context)
{
	m_pSelectedWorldView = context;
}

void ControllerView::onUpdate(void)
{
	TwSetCurrentWindow(glutGetWindow());

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	TwDraw();
	glutPostRedisplay();
}

void ControllerView::onResize(int w, int h)
{
	glViewport(0, 0, w, h);
	TwSetCurrentWindow(glutGetWindow());
	TwWindowSize(w, h);
	//char str[10];
	//sprintf_s(str, 10, "%d %d", width, height);
	//TwSetParam(m_pBar, NULL, "size", TW_PARAM_CSTRING, 1, str);
}

int ControllerView::MouseButtonCB(int glutButton, int glutState, int mouseX, int mouseY)
{
	TwSetCurrentWindow(glutGetWindow());
	return TwEventMouseButtonGLUT(glutButton, glutState, mouseX, mouseY);
}

int ControllerView::MouseMotionCB(int mouseX, int mouseY)
{
	TwSetCurrentWindow(glutGetWindow());
	return TwEventMouseMotionGLUT(mouseX, mouseY);
}


int ControllerView::KeyboardCB(unsigned char glutKey, int mouseX, int mouseY)
{
	if (getInstance()->m_pSelectedWorldView)
		getInstance()->m_pSelectedWorldView->onKeyboard(glutKey, mouseX, mouseY);
	TwSetCurrentWindow(glutGetWindow());
	return TwEventKeyboardGLUT(glutKey, mouseX, mouseY);
}

int ControllerView::SpecialKeyCB(int glutKey, int mouseX, int mouseY)
{
	TwSetCurrentWindow(glutGetWindow());
	return TwEventSpecialGLUT(glutKey, mouseX, mouseY);
}


void ControllerView::initLibrary()
{
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	glutInitDisplayMode(GLUT_RGBA);
	glClearColor(0, 0, 0, 1);
	m_windowId = glutCreateWindow("Controller");
	glutCreateMenu(NULL);
	glutDisplayFunc(ControllerView::onUpdate);
	glutReshapeFunc(ControllerView::onResize);
	static int s_init = 0;
	if (s_init == 0) {
		TwInit(TW_OPENGL, NULL);
		s_init = 1;
	}
	
	glutMouseFunc((GLUTmousebuttonfun)MouseButtonCB);
	glutMotionFunc((GLUTmousemotionfun)MouseMotionCB);
	glutPassiveMotionFunc((GLUTmousemotionfun)MouseMotionCB);
	glutKeyboardFunc((GLUTkeyboardfun)KeyboardCB);
	glutSpecialFunc((GLUTspecialfun)SpecialKeyCB);
	TwGLUTModifiersFunc(glutGetModifiers);
}


void TW_CALL ControllerView::onClickBtnStartStop(void * clientData)
{
	ControllerView* pControllerView = (ControllerView*)clientData;
	if(pControllerView->m_pSelectedWorldView)
		pControllerView->m_pSelectedWorldView->onKeyboard('p', 0, 0);
}

void TW_CALL ControllerView::onClickBtnStep(void * clientData)
{
	ControllerView* pControllerView = (ControllerView*)clientData;
	if (pControllerView->m_pSelectedWorldView)
		pControllerView->m_pSelectedWorldView->onKeyboard('s', 0, 0);
}

void TW_CALL ControllerView::onClickBtnAlloc(void * clientData)
{
	ControllerView* pControllerView = (ControllerView*)clientData;
	if (pControllerView->m_pSelectedWorldView)
		pControllerView->m_pSelectedWorldView->onKeyboard('a', 0, 0);
}

void TW_CALL ControllerView::onClickBtnClear(void * clientData)
{
	ControllerView* pControllerView = (ControllerView*)clientData;
	if (pControllerView->m_pSelectedWorldView)
		pControllerView->m_pSelectedWorldView->onKeyboard('c', 0, 0);
}

void TW_CALL ControllerView::onClickBtnAllocAll(void * clientData)
{
	ControllerView* pControllerView = (ControllerView*)clientData;
	if (pControllerView->m_pSelectedWorldView)
		pControllerView->m_pSelectedWorldView->onKeyboard('A', 0, 0);
}

void TW_CALL ControllerView::onClickBtnClearAll(void * clientData)
{
	ControllerView* pControllerView = (ControllerView*)clientData;
	if (pControllerView->m_pSelectedWorldView)
		pControllerView->m_pSelectedWorldView->onKeyboard('C', 0, 0);
}

void TW_CALL ControllerView::onClickBtnWorldGenerate(void * clientData)
{
	WorldContextManager::getInstance()->generateWorld();
}

void TW_CALL ControllerView::onClickBtnWorldDelete(void * clientData)
{
	WorldContextManager::getInstance()->deleteWorld(getInstance()->m_pSelectedWorldView);
}

void ControllerView::initUI()
{
	char str[64];
	TwSetCurrentWindow(glutGetWindow());
	m_pBar = TwNewBar("Controller");

	TwDefine(" Controller position = '0 0' movable=false resizable=false ");
	int colWidth = WINDOW_WIDTH / 2;
	TwSetParam(m_pBar, NULL, "valueswidth", TW_PARAM_INT32, 1, &colWidth);
	sprintf_s(str, 64, "%d %d", WINDOW_WIDTH, WINDOW_HEIGHT);
	TwSetParam(m_pBar, NULL, "size", TW_PARAM_CSTRING, 1, str);
	TwDefine(" GLOBAL fontsize=3 fontstyle=fixed ");

	TwAddVarRW(m_pBar, "width", TW_TYPE_INT32, &m_worldWidth, " min=16 max=4096 step=16 label='Width' group='World Parameters' ");
	TwAddVarRW(m_pBar, "height", TW_TYPE_INT32, &m_worldHeight, " min=16 max=4096 step=16 label='Height' group='World Parameters' ");
	TwEnumVal algorithmEV[] = { { NORMAL, "Normal" },{ NEW1, "New1" },{ NEW2, "New2" } };
	TwType algorithmType = TwDefineEnum("AlgorithmType", algorithmEV, ALGORITHM_NUM);
	TwAddVarRW(m_pBar, "Algorithm", algorithmType, &m_worldAlgorithm, " keyIncr='<' keyDecr='>' help='Change algorithm.' group='World Parameters' ");
	TwAddSeparator(m_pBar, NULL, NULL);
	TwAddButton(m_pBar, "btnWorldGenerate", ControllerView::onClickBtnWorldGenerate, this, " label='Generate' group='World Parameters' ");
	TwAddButton(m_pBar, "btnWorldDelete", ControllerView::onClickBtnWorldDelete, this, " label='Delete' group='World Parameters' ");

	TwAddButton(m_pBar, "btnStartStop", ControllerView::onClickBtnStartStop, this, " label='Play [p]' group='Operations' ");
	TwAddButton(m_pBar, "btnStep", ControllerView::onClickBtnStep, this, " label='Step [s]' group='Operations' ");
	TwAddButton(m_pBar, "btnAlloc", ControllerView::onClickBtnAlloc, this, " label='Alloc [a]' group='Operations' ");
	TwAddButton(m_pBar, "btnClear", ControllerView::onClickBtnClear, this, " label='Clear [c]' group='Operations' ");
	TwAddButton(m_pBar, "btnAllocAll", ControllerView::onClickBtnAllocAll, this, " label='Alloc all [A]' group='Operations' ");
	TwAddButton(m_pBar, "btnClearAll", ControllerView::onClickBtnClearAll, this, " label='Clear all [C]' group='Operations' ");
	TwAddSeparator(m_pBar, NULL, NULL);
	TwAddVarRW(m_pBar, "density", TW_TYPE_INT32, &m_density, "min=0 max=100 step=1 label='Density' group='Operations' ");
	TwAddVarRW(m_pBar, "prm1", TW_TYPE_INT32, &m_prm1, "min=0 max=10 step=1 label='Group' group='Operations' ");
	TwAddVarRW(m_pBar, "prm2", TW_TYPE_INT32, &m_prm2, "min=0 max=100 step=1 label='prm2' group='Operations' ");
	TwAddVarRW(m_pBar, "prm3", TW_TYPE_INT32, &m_prm3, "min=0 max=100 step=1 label='prm3' group='Operations' ");
	TwAddVarRW(m_pBar, "prm4", TW_TYPE_INT32, &m_prm4, "min=0 max=100 step=1 label='prm4' group='Operations' ");
}

