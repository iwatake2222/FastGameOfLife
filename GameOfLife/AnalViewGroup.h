#pragma once
#include "AnalView.h"
class AnalViewGroup :
	public AnalView
{
private:
	typedef struct {
		unsigned int numAlive;	// the number of total alive cells
		unsigned int numAliveGroupA_Pure;	// 999 * 1
		unsigned int numAliveGroupA_Quater;	// 999 * 1 ~ 999 * 1.25
		unsigned int numAliveGroup_Half;	// 999 * 1.25 ~ 999 * 1.75
		unsigned int numAliveGroupB_Quater;	// 999 * 1.75 ~ 999 * 2
		unsigned int numAliveGroupB_Pure;	// 999 * 2
	} ANAL_INFORMATION;

	ANAL_INFORMATION m_analInfo;
	
private:
	virtual void updateAnalInfo() override;
	void analyseInformation(ANAL_INFORMATION *info);
	void displayInformationGraph(ANAL_INFORMATION *info);
	void displayInformationText(ANAL_INFORMATION* info);

public:
	AnalViewGroup(WorldContext* pContext);
	virtual ~AnalViewGroup();
};
