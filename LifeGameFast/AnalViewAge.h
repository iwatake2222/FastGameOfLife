#pragma once
#include "AnalView.h"
class AnalViewAge :
	public AnalView
{
private:
	typedef struct {
		unsigned int numAlive;	// the number of total alive cells
		unsigned int numAlive0;	// the number of total alive cells whose age is age0( 0-20) normalized by 100
		unsigned int numAlive1;	// the number of total alive cells whose age is age0(20-40) normalized by 100
		unsigned int numAlive2;	// the number of total alive cells whose age is age0(40-60) normalized by 100
		unsigned int numAlive3;	// the number of total alive cells whose age is age0(60-80) normalized by 100
		unsigned int numAlive4;	// the number of total alive cells whose age is age0(80-100) normalized by 100
	} ANAL_INFORMATION;

	std::deque<ANAL_INFORMATION> analInfoHistory;

private:
	virtual void updateAnalInfo() override;
	void analyseInformation(ANAL_INFORMATION *info);
	void displayInformationGraph();
	void displayInformationText(ANAL_INFORMATION* info);

public:
	AnalViewAge(WorldContext* pContext);
	virtual ~AnalViewAge();
};

