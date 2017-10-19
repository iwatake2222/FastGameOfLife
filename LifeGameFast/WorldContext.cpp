#include "stdafx.h"
#include "WorldContext.h"
#include "WorldView.h"
#include "WorldLogic.h"

WorldContext::WorldContext(int worldWidth, int worldHeight)
{
	/* fixed during lifespan */
	WORLD_WIDTH = worldWidth;
	WORLD_HEIGHT = worldHeight;

	m_pLogic = new WorldLogic(WORLD_WIDTH, WORLD_WIDTH);
	m_pView = new WorldView(WORLD_WIDTH, WORLD_WIDTH, m_pLogic);
}


WorldContext::~WorldContext()
{
	delete m_pLogic;
	delete m_pView;
}

