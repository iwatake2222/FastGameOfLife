#pragma once
class ILogic
{
public:
	typedef struct {
		int status;	// 0 = not running, 1 = running
		int generation;
		int calcTime;	// current calculation time [ms]
	} WORLD_INFORMATION;

public:
	virtual ~ILogic() {}

	/* for thread control (must be called at the same timing as constructor/destructor */
	virtual void initialize() = 0;
	virtual void exit() = 0;

	/* for operation */
	virtual void startRun() = 0;
	virtual void stopRun() = 0;
	virtual void toggleRun() = 0;
	virtual void stepRun() = 0;
	virtual int* getDisplayMat() = 0;
	virtual void convertDisplay2Color(int displayedCell, double color[3]) = 0;
	virtual bool toggleCell(int worldX, int worldY, int prm1 = 0, int prm2 = 0, int prm3 = 0, int prm4 = 0) = 0;
	virtual bool setCell(int worldX, int worldY, int prm1 = 0, int prm2 = 0, int prm3 = 0, int prm4 = 0) = 0;
	virtual bool clearCell(int worldX, int worldY) = 0;
	virtual void allocCells(int x0, int x1, int y0, int y1, int density = 0, int prm1 = 0, int prm2 = 0, int prm3 = 0, int prm4 = 0) = 0;

	/* for analysis information */
	virtual void getInformation(WORLD_INFORMATION *info) = 0;
	virtual void resetInformation() = 0;

};

