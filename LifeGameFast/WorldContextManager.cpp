#include "stdafx.h"
#include "WorldContextManager.h"
#include "ControllerView.h"

WorldContextManager::WorldContextManager()
{
}


WorldContextManager::~WorldContextManager()
{
}

WorldContextManager* WorldContextManager::getInstance()
{
	static WorldContextManager s_WorldContextManager;
	return &s_WorldContextManager;
}

void WorldContextManager::generateWorld()
{
	m_contextVector.push_back(new WorldContext());
}

void WorldContextManager::deleteWorld(IView* view)
{
	for (std::vector<WorldContext*>::iterator it = m_contextVector.begin(); it != m_contextVector.end(); ) {
		if ((*it)->m_pView == view) {
			delete (*it);
			ControllerView::getInstance()->setCurrentWorldView(NULL);
			
			it = m_contextVector.erase(it);
		} else {
			++it;
		}
	}
}