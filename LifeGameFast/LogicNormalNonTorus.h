#pragma once
#include "LogicNormal.h"
class LogicNormalNonTorus : public LogicNormal
{
private:
	virtual void processWithBorderCheck(int x0, int x1, int y0, int y1) override;
public:
	LogicNormalNonTorus(int worldWidth, int worldHeight);
	virtual ~LogicNormalNonTorus() override;
};

