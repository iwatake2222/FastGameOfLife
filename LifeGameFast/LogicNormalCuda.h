#pragma once
#include "LogicNormal.h"
#include "algorithmCudaNormal.h"

class LogicNormalCuda : public LogicNormal
{
private:
	AlgorithmCudaNormal::ALGORITHM_CUDA_NORMAL_PARAM m_cudaParam;

private:
	virtual void gameLogic(int repeatNum) override;

public:
	LogicNormalCuda(int worldWidth, int worldHeight);
	virtual ~LogicNormalCuda() override;
	virtual int* getDisplayMat() override;

};