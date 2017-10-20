#pragma once
#include "WorldContext.h"
class WorldContextManager
{
private:
	std::vector<WorldContext*> m_contextVector;
private:
	WorldContextManager();
	~WorldContextManager();
public:
	static WorldContextManager* getInstance();
	void generateWorld();
	void deleteWorld(IView* view);
};

