#pragma once
#include "LogicBase.h"
class LogicNormalNonTorusMP : public LogicBase
{
private:
	virtual void gameLogic() override;
	void loopWithBorder(int x0, int x1, int y0, int y1, WORLD_INFORMATION* info);
	void loopWithoutBorder(int x0, int x1, int y0, int y1, WORLD_INFORMATION* info);
public:
	LogicNormalNonTorusMP(int worldWidth, int worldHeight);
	virtual ~LogicNormalNonTorusMP() override;
};

