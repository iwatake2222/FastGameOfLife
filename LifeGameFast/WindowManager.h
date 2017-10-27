#pragma once

#include <map>
#include "IView.h"

class WindowManager
{
private:
	std::map<int, IView *> m_viewMap;
	static int m_drawIntervalMS;

private:
	WindowManager();
	~WindowManager();
	
	static void idle(void);
	static void onUpdateWrapper();
	static void onResizeWrapper(int w, int h);
	static void onClickWrapper(int button, int state, int x, int y);
	static void onDragWrapper(int x, int y);
	static void onWheelWrapper(int wheel_number, int direction, int x, int y);
	static void onKeyboardWrapper(unsigned char key, int x, int y);

public:
	static WindowManager* getInstance();
	void init();
	void startLoop();
	void registerWindow(int windowId, IView* pWorldView);
	void unregisterWindow(int windowId);
	int getDrawIntervalMS();

};

