#include <stdio.h>
#include <string.h>
#include "Values.h"
#include "ControllerView.h"
#include "LogicBase.h"
#include "LogicNormal.h"
#include "LogicNormalMP.h"
#include "LogicNormalNonTorus.h"
#include "LogicNormalNonTorusMP.h"
#include "LogicGroup.h"
#ifdef ENABLE_CUDA
#include "LogicNormalCuda.h"
#include "LogicGroupCuda.h"
#endif

LogicBase::LogicBase(int worldWidth, int worldHeight)
{
	WORLD_WIDTH = worldWidth;
	WORLD_HEIGHT = worldHeight;

	m_matDisplay = new int[WORLD_WIDTH * WORLD_HEIGHT];
	memset(m_matDisplay, 0x00, sizeof(int) * WORLD_WIDTH * WORLD_HEIGHT);

	m_cmd = CMD_VIEW_2_LOGIC_NONE;
	m_isCmdCompleted = true;
	m_isCalculating = false;
	m_isMatrixUpdated = true;

	memset(&m_info, 0, sizeof(m_info));
	m_info.calcTime = 999;
	m_info.generation = 1;
	m_lastRetrievedGenration = m_info.generation;
}

LogicBase::~LogicBase()
{
	delete m_matDisplay;
}


void LogicBase::initialize()
{
	/* start main thread now */
	std::thread t(&LogicBase::threadFunc, this);
	m_thread.swap(t);
}

void LogicBase::exit()
{
	m_cmd = CMD_SELF_EXIT;
	if (m_thread.joinable()) m_thread.join();
	
	/* after this, destructor must be called to free memory */
}


void LogicBase::sendCommand(COMMAND cmd) {
	m_isCmdCompleted = false;
	m_cmd = cmd;
	// wait until logic thread completes the command
	// todo: rendezvous / mutex
	while (!m_isCmdCompleted) std::this_thread::yield();
}

void LogicBase::startRun()
{
	sendCommand(CMD_VIEW_2_LOGIC_START);
}

void LogicBase::stopRun()
{
	sendCommand(CMD_VIEW_2_LOGIC_STOP);
}

void LogicBase::toggleRun()
{
	if (m_info.status != 0) {
		sendCommand(CMD_VIEW_2_LOGIC_STOP);
	} else {
		sendCommand(CMD_VIEW_2_LOGIC_START);
	}
}

void LogicBase::stepRun()
{
	sendCommand(CMD_VIEW_2_LOGIC_STEP);
}

int* LogicBase::getDisplayMat() {
	return m_matDisplay;
}


bool LogicBase::toggleCell(int worldX, int worldY, int prm1, int prm2, int prm3, int prm4)
{
	if (m_info.status) sendCommand(CMD_VIEW_2_LOGIC_STOP);
	if (worldX >= 0 && worldX < WORLD_WIDTH && worldY >= 0 && worldY < WORLD_HEIGHT) {
		m_isMatrixUpdated = true;
		return true;
	}
	return false;
}

bool LogicBase::setCell(int worldX, int worldY, int prm1, int prm2, int prm3, int prm4)
{
	if (m_isCalculating) sendCommand(CMD_VIEW_2_LOGIC_STOP);
	if (worldX >= 0 && worldX < WORLD_WIDTH && worldY >= 0 && worldY < WORLD_HEIGHT) {
		if(!m_isMatrixUpdated)m_info.generation++;
		m_isMatrixUpdated = true;
		return true;
	}
	return false;
}

bool LogicBase::clearCell(int worldX, int worldY)
{
	if (m_isCalculating) sendCommand(CMD_VIEW_2_LOGIC_STOP);
	if (worldX >= 0 && worldX < WORLD_WIDTH && worldY >= 0 && worldY < WORLD_HEIGHT) {
		if (!m_isMatrixUpdated)m_info.generation++;
		m_isMatrixUpdated = true;
		return true;
	}
	return false;
}

void LogicBase::allocCells(int x0, int x1, int y0, int y1, int density, int prm1, int prm2, int prm3, int prm4)
{
	if (m_isCalculating) sendCommand(CMD_VIEW_2_LOGIC_STOP);
	if (x0 < 0) x0 = 0;
	if (x1 >= WORLD_WIDTH) x1 = WORLD_WIDTH;
	if (y0 < 0) y0 = 0;
	if (y1 >= WORLD_HEIGHT) y1 = WORLD_HEIGHT;
	if (density < 0) density = 0;
	if (density > 100) density = 100;

	m_info.generation++;

	if (density != 0) {
		int reverseDensity = 100 / density;
		for (int y = y0; y < y1; y++) {
			int yIndex = WORLD_WIDTH * y;
			for (int x = x0; x < x1; x++) {
				if (rand() % reverseDensity == 0) {
					setCell(x, y, prm1, prm2, prm3, prm4);
				} else {
					clearCell(x, y);
				}
			}
		}
	} else {
		int *mat = getDisplayMat();
		for (int y = y0; y < y1; y++) {
			int yIndex = WORLD_WIDTH * y;
			for (int x = x0; x < x1; x++) {
				clearCell(x, y);
			}
		}
	}
}


void LogicBase::threadFunc()
{
	bool isExit = false;
	while (!isExit) {
		/*** Treat command ***/
		COMMAND cmd = m_cmd;
		m_cmd = CMD_VIEW_2_LOGIC_NONE;
		switch (cmd) {
		case CMD_VIEW_2_LOGIC_NONE:
		default:
			break;
		case CMD_VIEW_2_LOGIC_STOP:
			m_isCalculating = false;
			m_isCmdCompleted = true;
			m_info.status = 0;
			break;
		case CMD_VIEW_2_LOGIC_START:
			m_isCalculating = true;
			m_isCmdCompleted = true;
			m_info.status = 1;
			break;
		case CMD_VIEW_2_LOGIC_STEP:
			m_isCalculating = false;
			m_info.status = 1;
			// send comp after one step done
			break;
		case CMD_SELF_EXIT:
			isExit = true;
			break;
		}

		if (m_info.status) {
			/*** Operate game logic for one generation ***/
			std::chrono::system_clock::time_point  timeStart, timeEnd;
			timeStart = std::chrono::system_clock::now();
			gameLogic(ControllerView::getInstance()->m_skipNum);	/*** <- use selected algorithm (implemented in sub class) ***/
			timeEnd = std::chrono::system_clock::now();
			int timeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart).count();
			m_info.calcTime = timeElapsed / ControllerView::getInstance()->m_skipNum;
			//printf("fpsCalc = %lf\n", 1000.0 / timeElapsed);
		}
		if (cmd == CMD_VIEW_2_LOGIC_STEP) {
			m_info.status = 0;
			m_isCmdCompleted = true;
		}
		
		/* workaround. without this, this thread occupies CPU */
		std::this_thread::sleep_for(std::chrono::microseconds(1));	
		//for (int i = 0; i < 1000; i++) std::this_thread::yield();

	}
}

void LogicBase::getInformation(WORLD_INFORMATION *info)
{
	*info = m_info;
}

void LogicBase::resetInformation()
{
	memset(&m_info, 0, sizeof(m_info));
	m_info.calcTime = 999;
}



ILogic* LogicBase::generateLogic(int algorithm, int width, int height)
{
	switch (algorithm) {
	case ALGORITHM_NORMAL:
		return new LogicNormal(width, height);
	case ALGORITHM_NORMAL_MP:
		return new LogicNormalMP(width, height);
	case ALGORITHM_NORMAL_NON_TORUS:
		return new LogicNormalNonTorus(width, height);
	case ALGORITHM_NORMAL_NON_TORUS_MP:
		return new LogicNormalNonTorusMP(width, height);
	//case ALGORITHM_NORMAL_NON_TORUS_CUDA:
	//	return new LogicNormal(width, height);
	case ALGORITHM_GROUP_MP:
		return new LogicGroup(width, height);
#ifdef ENABLE_CUDA
	case ALGORITHM_NORMAL_CUDA:
		return new LogicNormalCuda(width, height);
	case ALGORITHM_GROUP_CUDA:
		return new LogicGroupCuda(width, height);
#endif
	default:
		return new LogicNormal(width, height);
	}
}

