#pragma once
#include <deque>
#include "AnalView.h"
class AnalViewVirus :
	public AnalView
{
private:
	typedef struct {
		unsigned int numTotal;
		unsigned int numHealthy;
		unsigned int numInfected;
		unsigned int numOnset;
		unsigned int numContagius;
		unsigned int numImmunized;
	} ANAL_INFORMATION;

	std::deque<ANAL_INFORMATION> analInfoHistory;

private:
	virtual void updateAnalInfo() override;
	void analyseInformation(ANAL_INFORMATION *info);
	void displayInformationGraph();
	void displayInformationText(ANAL_INFORMATION* info);

public:
	AnalViewVirus(WorldContext* pContext);
	virtual ~AnalViewVirus();
};

