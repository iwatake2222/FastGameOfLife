#pragma once
#include "LogicBase.h"

/* Logic class for normal algorithm
 * Each cell has <int> parameter as its age
*/
class LogicNormal : public LogicBase
{
protected:
	const static int CELL_DEAD = 0;
	const static int CELL_ALIVE = 1;

protected:
	/* World Area Matrix
	* don't use std::vector<std::vector<int>> because it's too late
	* don't use 2-dimension array for copy operation later
	*/
	int *m_mat[2];
	int m_matIdOld;	// this matrix is ready to display (CLEAN)
	int m_matIdNew;	// currently this matrix is being modified using m_matIdOld (DIRTY)

protected:
	virtual void gameLogic() override;
	virtual void processWithBorderCheck(int x0, int x1, int y0, int y1);
	virtual void processWithoutBorderCheck(int x0, int x1, int y0, int y1);
	void updateCell(int x, int yLine, int cnt);

public:
	LogicNormal(int worldWidth, int worldHeight);
	virtual ~LogicNormal() override;

	virtual bool toggleCell(int worldX, int worldY, int prm1, int prm2, int prm3, int prm4) override;
	virtual bool setCell(int worldX, int worldY, int prm1, int prm2, int prm3, int prm4) override;
	virtual bool clearCell(int worldX, int worldY) override;
};

