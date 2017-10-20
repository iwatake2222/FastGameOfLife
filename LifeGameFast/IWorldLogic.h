#pragma once
class IWorldLogic
{
public:
	typedef struct {
		unsigned int status;	// 0 = not running, 1 = running
		unsigned int generation;
		unsigned int numAlive;	// the number of total alive cells
		unsigned int numBirth;	// the number of birth in the current generation
		unsigned int numDie;		// the number of death in the current generation
		unsigned int calcTime;	// the number of total alive cellsint calcTime;	// current calculation time [ms]
	} WORLD_INFORMATION;
public:
	virtual ~IWorldLogic() {}
	/* for operation */
	virtual void toggleRun() = 0;
	virtual void stepRun() = 0;
	virtual int* getDisplayMat() = 0;
	virtual void copyDisplayMat(int* matOut) = 0;
	virtual void toggleCell(int worldX, int worldY) = 0;
	virtual void setCell(int worldX, int worldY) = 0;
	virtual void clearCell(int worldX, int worldY) = 0;
	virtual void allocCells(int x0, int x1, int y0, int y1, int prm0, int prm1, int prm2, int prm3, int prm4) = 0;

	/* for analysis information */
	virtual void getInformation(WORLD_INFORMATION *info) = 0;
	virtual void resetInformation() = 0;

};

