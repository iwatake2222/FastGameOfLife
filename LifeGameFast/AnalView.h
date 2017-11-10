#pragma once
#include "WorldContext.h"
#include "IView.h"
class AnalView : public IView
{
protected:
	static const int MARGIN = 10;
	static const int WINDOW_HEIGHT = 240;
	static const int TEXT_AREA_WIDTH = 200;

protected:
	/* Window Area (Device Area) is in [pixel]
	* x = 0 ~ WINDOW_WIDTH - 1, y = 0 ~ WINDOW_HEIGHT - 1
	* top left is(0, 0), bottom right is (WINDOW_WIDTH - 1, WINDOW_HEIGHT - 1)
	*/
	int m_windowWidth;
	int m_windowHeight;

	/* Graph Area (World Area). in this view, World Area = Device Area(pixel) */
	int m_graphWidth;
	int m_graphHeight;

	int m_windowId;

	int m_intervalCnt;

	/* pointer to the parent context which contains me */
	WorldContext *m_pContext;

	int m_NumHistory;
	
	unsigned int m_maxGraphY;

	int m_lastAnalyzedGeneration;

protected:
	void initView();

	void drawString(double x, double y, char* str);
	void writeTextArea(int line, char* str);

	void drawBackGround(void);
	virtual void updateAnalInfo() = 0;

	AnalView(WorldContext* pContext);	// created by createAppropreateAnalView
	virtual ~AnalView();

public:
	static IView* createAppropreateAnalView(WorldContext* pContext);

	void refresh();
	void onUpdate(void);
	void onResize(int w, int h);
	void onClick(int button, int state, int x, int y);
	void onDrag(int x, int y);
	void onWheel(int wheel_number, int direction, int x, int y);
	void onKeyboard(unsigned char key, int x, int y);
};

