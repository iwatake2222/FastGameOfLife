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
	//m_contextVector.push_back(new WorldContext());
}

void WorldContextManager::deleteWorld(WorldContext* pContext)
{
	//for (std::vector<WorldContext*>::iterator it = m_contextVector.begin(); it != m_contextVector.end(); ) {
	//	if ((*it) == pContext) {
	//		delete (*it);
	//		ControllerView::getInstance()->setCurrentWorldContext(NULL);
	//		
	//		it = m_contextVector.erase(it);
	//	} else {
	//		++it;
	//	}
	//}
}