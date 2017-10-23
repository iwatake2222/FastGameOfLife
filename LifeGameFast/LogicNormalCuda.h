#pragma once
#include "LogicNormal.h"
#include "algorithmCudaNormal.h"

class LogicNormalCuda : public LogicNormal
{
private:
	int *m_originalMatDisplay;
	AlgorithmCudaNormal::ALGORITHM_CUDA_NORMAL_PARAM cudaParam;

private:
	virtual void gameLogic() override;

public:
	LogicNormalCuda(int worldWidth, int worldHeight);
	virtual ~LogicNormalCuda() override;

};