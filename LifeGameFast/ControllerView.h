#pragma once
#include "WorldContext.h"
class ControllerView 
{
private:
	typedef enum { NORMAL = 0, NEW1, NEW2, ALGORITHM_NUM } ALGORITHM;
	static const int WINDOW_WIDTH  = 400;
	static const int WINDOW_HEIGHT = 400;
	int m_windowId;
	TwBar *m_pBar;
	WorldContext* m_pCurrentWorldContext;


public:
	int m_worldWidth;
	int m_worldHeight;
	ALGORITHM m_worldAlgorithm;
	int m_density;
	int m_prm1;
	int m_prm2;
	int m_prm3;
	int m_prm4;

	int m_viewInterval;


private:
	void initLibrary();
	void initUI();
	static void onUpdate(void);
	static void onResize(int w, int h);

	static int ControllerView::MouseButtonCB(int glutButton, int glutState, int mouseX, int mouseY);
	static int ControllerView::MouseMotionCB(int mouseX, int mouseY);
	static int ControllerView::KeyboardCB(unsigned char glutKey, int mouseX, int mouseY);
	static int ControllerView::SpecialKeyCB(int glutKey, int mouseX, int mouseY);
	
	static void TW_CALL onClickBtnStartStop(void * clientData);
	static void TW_CALL onClickBtnStep(void * clientData);
	static void TW_CALL onClickBtnAlloc(void * clientData);
	static void TW_CALL onClickBtnClear(void * clientData);
	static void TW_CALL onClickBtnAllocAll(void * clientData);
	static void TW_CALL onClickBtnClearAll(void * clientData);
	static void TW_CALL onClickBtnInformation(void * clientData);
	static void TW_CALL onClickBtnWorldGenerate(void * clientData);
	static void TW_CALL onClickBtnWorldQuit(void * clientData);

	ControllerView();
	~ControllerView();

public:
	static ControllerView* getInstance();
	void setCurrentWorldContext(WorldContext* context);

};

