#pragma once
#include "LogicNormal.h"
class LogicNormalCuda : public LogicNormal
{
private:
	virtual void gameLogic() override;

public:
	LogicNormalCuda(int worldWidth, int worldHeight);
	virtual ~LogicNormalCuda() override;

};