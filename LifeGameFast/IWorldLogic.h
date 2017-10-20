#pragma once
class IWorldLogic
{
public:
	virtual ~IWorldLogic() {}
	virtual void toggleRun() = 0;
	virtual void stepRun() = 0;
	virtual int* getDisplayMat() = 0;
	virtual void copyDisplayMat(int* matOut) = 0;
	virtual void toggleCell(int worldX, int worldY) = 0;
	virtual void setCell(int worldX, int worldY) = 0;
	virtual void clearCell(int worldX, int worldY) = 0;
	virtual void allocCells(int x0, int x1, int y0, int y1, int prm0, int prm1, int prm2, int prm3, int prm4) = 0;
};

