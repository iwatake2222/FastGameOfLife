#pragma once
#include "ILogic.h"

/* Base Logic class
 *  has thread main loop and treats message for the thread,
 *  doesn't care the status/information/type of each cell (sub class converts cell information into displayed data)
 *  has concrete logic class generator (should be separated from this class) 
*/
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


protected:
	/* World Area (LifeGame world) is
	* x = 0 ~ WORLD_WIDTH - 1, y = 0 ~ WORLD_HEIGHT - 1
	* bottm left is (0,0), top right is (WORLD_WIDTH - 1, WORLD_HEIGHT - 1)
	*/
	int WORLD_WIDTH;
	int WORLD_HEIGHT;

	/* command communication */
	/* todo: rendezvous / mutex */
	COMMAND m_cmd;
	bool m_isCmdCompleted;

	/* status control */
	bool m_isCalculating;

	bool m_isMatrixUpdated;
	int m_lastRetrievedGenration;

	/* matrix data to be displayed */
	int *m_matDisplay;
	std::mutex m_mutexMatDisplay; // to avoid logic thread update matrix while view thread is copying it

	std::thread m_thread;

	/* Analysis information */
	WORLD_INFORMATION m_info;

private:
	virtual void gameLogic() = 0;		// algorithm is implemented in sub class

private:
	void sendCommand(COMMAND cmd);
	void threadFunc();

public:
	LogicBase(int worldWidth, int worldHeigh);
	virtual ~LogicBase() override;

	virtual void initialize() override final;
	virtual void exit() override final;
	virtual void startRun() override final;
	virtual void stopRun() override final;
	virtual void toggleRun() override final;
	virtual void stepRun() override final;
	
	virtual int* getDisplayMat() override;
	virtual bool toggleCell(int worldX, int worldY, int prm1 = 0, int prm2 = 0, int prm3 = 0, int prm4 = 0) override;
	virtual bool setCell(int worldX, int worldY, int prm1 = 0, int prm2 = 0, int prm3 = 0, int prm4 = 0) override;
	virtual bool clearCell(int worldX, int worldY) override;
	virtual void allocCells(int x0, int x1, int y0, int y1, int density, int prm1 = 0, int prm2 = 0, int prm3 = 0, int prm4 = 0) override;
	virtual void getInformation(WORLD_INFORMATION *info) override;
	virtual void resetInformation() override;

public:
	// todo: should make factory or manager class
	static ILogic* generateLogic(int algorithm, int width, int height);
};

