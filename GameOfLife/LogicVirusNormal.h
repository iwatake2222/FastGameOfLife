#pragma once
#include "LogicBase.h"
#include "Person.h"

/* Logic class for normal algorithm
 * Each cell has <int> parameter as its age
*/
class LogicVirusNormal : public LogicBase
{
public:
	typedef enum {
		NONE,
		HEALTHY,
		INFECTED,
		ONSET,
		CONTAGIUS,
		IMMUNIZED,
	} CELL_STATUS;

protected:
	/* World Area Matrix
	* don't use std::vector<std::vector<int>> because it's too late
	* don't use 2-dimension array for copy operation later
	*/
	int *m_matSrc;

protected:
	virtual void gameLogic(int repeatNum) override;
	int convertCell2Display(int cell);

public:
	LogicVirusNormal(int worldWidth, int worldHeight);
	virtual ~LogicVirusNormal() override;

	virtual int* getDisplayMat() override;
	virtual void convertDisplay2Color(int displayedCell, double color[3]) override;
	virtual bool toggleCell(int worldX, int worldY, int prm1, int prm2, int prm3, int prm4) override;
	virtual bool setCell(int worldX, int worldY, int prm1, int prm2, int prm3, int prm4) override;
	virtual bool clearCell(int worldX, int worldY) override;
	virtual void allocCells(int x0, int x1, int y0, int y1, int density, int prm1 = 0, int prm2 = 0, int prm3 = 0, int prm4 = 0) override;

private:
	std::vector<Person*> m_personList;
	int getCellStatus(Person *p);
};

