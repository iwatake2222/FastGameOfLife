#pragma once
#include "LogicBase.h"

/* Logic class for group algorithm
*  Each cell has <int> parameter as its group
*/
class LogicGroup : public LogicBase
{
public:
	const static int PURE_GROUP_A = 999 * 1;
	const static int PURE_GROUP_B = 999 * 2;
	const static int CELL_DEAD = 0;
	const static int CELL_ALIVE = 1;
protected:
	typedef struct {
		int group;	// e.g.: PURE GROUP_A = 999*1, PURE GROUP_B = 999*2. A child of them = 999*1.5
		int age;
	} DNA;

protected:
	/* World Area Matrix
	* don't use std::vector<std::vector<int>> because it's too late
	* don't use 2-dimension array for copy operation later
	*/
	DNA *m_matSrc;
	DNA *m_matDst;

protected:
	virtual void gameLogic(int repeatNum) override;
	virtual void processWithBorderCheck(int x0, int x1, int y0, int y1);
	virtual void processWithoutBorderCheck(int x0, int x1, int y0, int y1);
	void updateCell(int x, int yLine, int cnt, int group);
	int convertCell2Display(DNA *cell);

public:
	LogicGroup(int worldWidth, int worldHeight);
	virtual ~LogicGroup() override;

	virtual int* getDisplayMat() override;
	virtual void convertDisplay2Color(int displayedCell, double color[3]) override;
	virtual bool toggleCell(int worldX, int worldY, int prm1, int prm2, int prm3, int prm4) override;
	virtual bool setCell(int worldX, int worldY, int prm1, int prm2, int prm3, int prm4) override;
	virtual bool clearCell(int worldX, int worldY) override;

	

};

