#pragma once
#include "LogicBase.h"
class LogicNormal : public LogicBase
{
private:
	virtual void gameLogic() override;
	void loopWithBorder(int x0, int x1, int y0, int y1, WORLD_INFORMATION* info);
	void loopWithoutBorder(int x0, int x1, int y0, int y1, WORLD_INFORMATION* info);
public:
	LogicNormal(int worldWidth, int worldHeight);
	virtual ~LogicNormal() override;
	
};

