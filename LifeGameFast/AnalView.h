#pragma once
#include "WorldContext.h"
#include "IView.h"
class AnalView : public IView
{
private:
	static const int MARGIN = 10;
	static const int WINDOW_HEIGHT = 240;
	static const int TEXT_AREA_WIDTH = 200;

	typedef struct {
		unsigned int numAlive;	// the number of total alive cells
		unsigned int numAlive0;	// the number of total alive cells whose age is age0( 0-20) normalized by 100
		unsigned int numAlive1;	// the number of total alive cells whose age is age0(20-40) normalized by 100
		unsigned int numAlive2;	// the number of total alive cells whose age is age0(40-60) normalized by 100
		unsigned int numAlive3;	// the number of total alive cells whose age is age0(60-80) normalized by 100
		unsigned int numAlive4;	// the number of total alive cells whose age is age0(80-100) normalized by 100
	} ANAL_INFORMATION;


private:
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
	std::deque<ANAL_INFORMATION> analInfoHistory;
	unsigned int m_maxGraphY;

private:
	void initView();

	void drawString(double x, double y, char* str);
	void writeTextArea(int line, char* str);

	void drawBackGround(void);
	void analyseInformation(ANAL_INFORMATION *info);
	void displayInformationGraph();
	void displayInformationText(ANAL_INFORMATION* info);
public:
	AnalView(WorldContext* pContext);
	~AnalView();

	void onUpdate(void);
	void onResize(int w, int h);
	void onClick(int button, int state, int x, int y);
	void onDrag(int x, int y);
	void onWheel(int wheel_number, int direction, int x, int y);
	void onKeyboard(unsigned char key, int x, int y);
};

