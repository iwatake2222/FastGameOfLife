#include "stdafx.h"
#include "WorldContext.h"
#include "WorldView.h"
#include "WorldLogic.h"
#include "AnalView.h"
#include "ControllerView.h"

WorldContext::WorldContext()
{
	/* fixed during lifespan */
	WORLD_WIDTH = ControllerView::getInstance()->m_worldWidth;
	WORLD_HEIGHT = ControllerView::getInstance()->m_worldHeight;

	m_pLogic = new WorldLogic(WORLD_WIDTH, WORLD_HEIGHT);
	m_pView = new WorldView(this);
	m_pAnalView = new AnalView(this);

	ControllerView::getInstance()->setCurrentWorldContext(this);
}


WorldContext::~WorldContext()
{
	delete m_pAnalView;
	delete m_pView;
	delete m_pLogic;
}

