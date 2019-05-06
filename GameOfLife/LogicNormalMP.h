#pragma once
#include "LogicNormal.h"
class LogicNormalMP : public LogicNormal
{
private:
	virtual void processWithBorderCheck(int x0, int x1, int y0, int y1) override;
	virtual void processWithoutBorderCheck(int x0, int x1, int y0, int y1) override;

public:
	LogicNormalMP(int worldWidth, int worldHeight);
	virtual ~LogicNormalMP() override;
};

