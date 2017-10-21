#pragma once
#include "WorldContext.h"

class WorldView : public IView
{
private:
	/* World Area (LifeGame world) is
	* x = 0 ~ WORLD_WIDTH - 1, y = 0 ~ WORLD_HEIGHT - 1
	* bottm left is (0,0), top right is (WORLD_WIDTH - 1, WORLD_HEIGHT - 1)
	*/
	int WORLD_WIDTH;
	int WORLD_HEIGHT;

	/* display area and grid line margine */
	double WORLD_WIDTH_MARGIN;
	double WORLD_HEIGHT_MARGIN;

	/* pointer to the parent context which contains me */
	WorldContext *m_pContext;

	/* current visible world area
	* changed when mouse drag/wheel, resize
	* the aspect of worldVisible must be the same as that of window
	*/
	double m_worldVisibleX0;
	double m_worldVisibleX1;
	double m_worldVisibleY0;
	double m_worldVisibleY1;

	/* Window Area (Device Area) is in [pixel]
	* x = 0 ~ WINDOW_WIDTH - 1, y = 0 ~ WINDOW_HEIGHT - 1
	* top left is(0, 0), bottom right is (WINDOW_WIDTH - 1, WINDOW_HEIGHT - 1)
	*/
	int m_windowWidth;
	int m_windowHeight;
	
	/* valid on left button dragging. value is in world area */
	int m_previousLDragXWorld;
	int m_previousLDragYWorld;

	/* valid on right button dragging. value is in window area[px] */
	int m_previousRDragX;
	int m_previousRDragY;

	int m_windowId;
	
	int m_intervalCnt;

private:
	void initOpenGL();
	void initView();
	void drawGrid();
	void drawCells();
	void convertPosWindow2World(int x, int y, int* worldX, int* worldY);

public:
	WorldView(WorldContext* pContext, int windowX = DEFAULT_WINDOW_X, int windowY = DEFAULT_WINDOW_Y, int windowWidth = DEFAULT_WINDOW_WIDTH, int windowHeight = DEFAULT_WINDOW_HEIGHT);
	~WorldView();

	void onUpdate(void);
	void onResize(int w, int h);
	void onClick(int button, int state, int x, int y);
	void onDrag(int x, int y);
	void onWheel(int wheel_number, int direction, int x, int y);
	void onKeyboard(unsigned char key, int x, int y);
};

