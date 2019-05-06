#pragma once
#include "Values.h"
#include "IView.h"
#include "ILogic.h"
class WorldContext
{
public:
	/* World Area (LifeGame world) is
	* x = 0 ~ WORLD_WIDTH - 1, y = 0 ~ WORLD_HEIGHT - 1
	* bottm left is (0,0), top right is (WORLD_WIDTH - 1, WORLD_HEIGHT - 1)
	*/
	int WORLD_WIDTH;
	int WORLD_HEIGHT;

	ILogic* m_pLogic;
	IView* m_pView;
	IView* m_pAnalView;

	ALGORITHM m_algorithm;

public:
	WorldContext();
	WorldContext(int width, int height, ALGORITHM algorithm = ALGORITHM_AUTO, int windowX = DEFAULT_WINDOW_X, int windowY = DEFAULT_WINDOW_Y, int windowWidth = DEFAULT_WINDOW_WIDTH, int windowHeight = DEFAULT_WINDOW_HEIGHT);
	~WorldContext();

	static WorldContext* generateFromFile();
	static WorldContext* generateFromFile(const char* filename, int width, int height);
	static void saveToFile(WorldContext* context);
};

