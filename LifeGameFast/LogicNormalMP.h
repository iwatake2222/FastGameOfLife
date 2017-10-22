#pragma once
#include "LogicNormal.h"
class LogicNormalMP : public LogicNormal
{
private:
	virtual void loopWithBorderCheck(int x0, int x1, int y0, int y1) override;
	virtual void loopWithoutBorderCheck(int x0, int x1, int y0, int y1) override;

public:
	LogicNormalMP(int worldWidth, int worldHeight);
	virtual ~LogicNormalMP() override;
};

