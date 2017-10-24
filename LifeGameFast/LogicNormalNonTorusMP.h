#pragma once
#include "LogicNormal.h"
class LogicNormalNonTorusMP : public LogicNormal
{
private:
	virtual void processWithBorderCheck(int x0, int x1, int y0, int y1) override;
	virtual void processWithoutBorderCheck(int x0, int x1, int y0, int y1) override;

public:
	LogicNormalNonTorusMP(int worldWidth, int worldHeight);
	virtual ~LogicNormalNonTorusMP() override;
};

