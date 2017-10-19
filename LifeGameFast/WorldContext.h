#pragma once
#include "IWorldView.h"
#include "IWorldLogic.h"
class WorldContext
{
private:
	/* World Area (LifeGame world) is
	* x = 0 ~ WORLD_WIDTH - 1, y = 0 ~ WORLD_HEIGHT - 1
	* bottm left is (0,0), top right is (WORLD_WIDTH - 1, WORLD_HEIGHT - 1)
	*/
	int WORLD_WIDTH;
	int WORLD_HEIGHT;

	IWorldLogic* m_pLogic;
	IWorldView* m_pView;

public:
	WorldContext(int worldWidth, int worldHeight);
	~WorldContext();
};

