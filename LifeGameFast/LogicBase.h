#pragma once
#include "ILogic.h"
class LogicBase : public ILogic
{
private:
	/* commands from public function to internal thread */
	typedef enum {
		CMD_VIEW_2_LOGIC_NONE,
		CMD_VIEW_2_LOGIC_STOP,
		CMD_VIEW_2_LOGIC_START,
		CMD_VIEW_2_LOGIC_STEP,
		CMD_SELF_EXIT,
	} COMMAND;

	const static int CELL_DEAD  = 0;
	const static int CELL_ALIVE = 1;


protected:
	/* World Area (LifeGame world) is
	* x = 0 ~ WORLD_WIDTH - 1, y = 0 ~ WORLD_HEIGHT - 1
	* bottm left is (0,0), top right is (WORLD_WIDTH - 1, WORLD_HEIGHT - 1)
	*/
	int WORLD_WIDTH;
	int WORLD_HEIGHT;

	/* World Area Matrix
	 * don't use std::vector<std::vector<int>> because it's too late
	 * don't use 2-dimension array for copy operation later
	 */
	int *m_mat[2];
	int *m_matDisplay;

	/* command communication */
	/* todo: rendezvous / mutex */
	COMMAND m_cmd;
	bool m_isCmdCompleted;

	/* status control */
	bool m_isRunning;
	int m_matIdOld;	// this matrix is ready to display (CLEAN)
	int m_matIdNew;	// currently this matrix is being modified using m_matIdOld (DIRTY)

	std::mutex m_mutexMat;	// to avoid logic thread update matrix while view thread is copying it
	std::thread m_thread;

	/* Analysis information */
	WORLD_INFORMATION m_info;

private:
	virtual void allocMemory(int **p, int size);
	virtual void freeMemory(int *p);
	virtual void gameLogic();

private:
	void sendCommand(COMMAND cmd);
	void threadFunc();

public:
	LogicBase(int worldWidth, int worldHeigh);
	virtual ~LogicBase();

	void initialize();
	void exit();
	void startRun();
	void stopRun();
	void toggleRun();
	void stepRun();

	int* getDisplayMat();

	void toggleCell(int worldX, int worldY);
	void setCell(int worldX, int worldY);
	void clearCell(int worldX, int worldY);

	void allocCells(int x0, int x1, int y0, int y1, int prm0, int prm1, int prm2, int prm3, int prm4);

	void getInformation(WORLD_INFORMATION *info);
	void resetInformation();

public:
	// todo: should make factory or manager class
	static ILogic* generateLogic(int algorithm, int width, int height);
};

