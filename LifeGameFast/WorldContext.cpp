#include "stdafx.h"
#include "WorldContext.h"
#include "WorldView.h"
#include "WorldLogic.h"
#include "ControllerView.h"

WorldContext::WorldContext()
{
	/* fixed during lifespan */
	WORLD_WIDTH = ControllerView::getInstance()->m_worldWidth;
	WORLD_HEIGHT = ControllerView::getInstance()->m_worldHeight;

	m_pLogic = new WorldLogic(WORLD_WIDTH, WORLD_WIDTH);
	m_pView = new WorldView(WORLD_WIDTH, WORLD_WIDTH, m_pLogic);

	ControllerView::getInstance()->setCurrentWorldView(m_pView);
}


WorldContext::~WorldContext()
{
	delete m_pLogic;
	delete m_pView;
}

