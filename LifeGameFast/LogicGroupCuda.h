#pragma once
#include "LogicGroup.h"
#include "algorithmCudaGroup.h"

class LogicGroupCuda : public LogicGroup
{
private:
	AlgorithmCudaGroup::ALGORITHM_CUDA_GROUP_PARAM m_cudaParam;

private:
	virtual void gameLogic(int repeatNum) override;

public:
	LogicGroupCuda(int worldWidth, int worldHeight);
	virtual ~LogicGroupCuda() override;
	virtual int* getDisplayMat() override;
};

