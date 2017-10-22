#pragma once
#include "LogicBase.h"
class LogicNormalCuda : public LogicBase
{
private:
	virtual void gameLogic() override;
	virtual void allocMemory(int **p, int size) override;
	virtual void freeMemory(int *p) override;

	void loopWithBorder(int x0, int x1, int y0, int y1, WORLD_INFORMATION* info);
	void loopWithoutBorder(int x0, int x1, int y0, int y1, WORLD_INFORMATION* info);
public:
	LogicNormalCuda(int worldWidth, int worldHeight);
	virtual ~LogicNormalCuda() override;

};