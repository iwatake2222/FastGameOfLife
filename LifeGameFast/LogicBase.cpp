
#include "stdafx.h"
#include "Values.h"
#include "LogicBase.h"
#include "LogicNormal.h"
#include "LogicNormalMP.h"
#include "LogicNormalNonTorus.h"
#include "LogicNormalNonTorusMP.h"

LogicBase::LogicBase(int worldWidth, int worldHeight)
{
	WORLD_WIDTH = worldWidth;
	WORLD_HEIGHT = worldHeight;

	m_cmd = CMD_VIEW_2_LOGIC_NONE;
	m_isCmdCompleted = true;
	m_isRunning = false;
	
	m_matIdOld = 0;
	m_matIdNew = 1;
	for (int i = 0; i < 2; i++) {
		m_mat[i] = new int[WORLD_WIDTH * WORLD_HEIGHT];
		memset(m_mat[i], 0x00, sizeof(int) * WORLD_WIDTH * WORLD_HEIGHT);
	}

	/* start main thread now*/
	std::thread t(&LogicBase::threadFunc, this);
	m_thread.swap(t);

	memset(&m_info, 0, sizeof(m_info));
	m_info.calcTime = 999;
	m_info.generation = 1;
}

LogicBase::~LogicBase()
{
	m_cmd = CMD_SELF_EXIT;
	if (m_thread.joinable()) m_thread.join();

	for (int i = 0; i < 2; i++) {
		delete m_mat[i];
		m_mat[i] = 0;
	}
	
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
	if (m_isRunning) {
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
	/* this is unsafe. the matrix retrieved might be updated while using */
	return m_mat[m_matIdOld];
}

void LogicBase::copyDisplayMat(int* matOut) {
	m_mutexMat.lock();
	memcpy(matOut, m_mat[m_matIdOld], sizeof(int) * WORLD_WIDTH * WORLD_HEIGHT);
	m_mutexMat.unlock();
}

void LogicBase::toggleCell(int worldX, int worldY)
{
	if (m_isRunning) sendCommand(CMD_VIEW_2_LOGIC_STOP);
	int *mat = getDisplayMat();
	if (worldX >= 0 && worldX < WORLD_WIDTH && worldY >= 0 && worldY < WORLD_HEIGHT) {
		mat[WORLD_WIDTH * worldY + worldX] = mat[WORLD_WIDTH * worldY + worldX] == CELL_DEAD ? CELL_ALIVE : CELL_DEAD;
	}
}

void LogicBase::setCell(int worldX, int worldY)
{
	if (m_isRunning) sendCommand(CMD_VIEW_2_LOGIC_STOP);
	int *mat = getDisplayMat();
	if (worldX >= 0 && worldX < WORLD_WIDTH && worldY >= 0 && worldY < WORLD_HEIGHT) {
		mat[WORLD_WIDTH * worldY + worldX] = CELL_ALIVE;
	}
}

void LogicBase::clearCell(int worldX, int worldY)
{
	if (m_isRunning) sendCommand(CMD_VIEW_2_LOGIC_STOP);
	int *mat = getDisplayMat();
	if (worldX >= 0 && worldX < WORLD_WIDTH && worldY >= 0 && worldY < WORLD_HEIGHT) {
		mat[WORLD_WIDTH * worldY + worldX] = CELL_DEAD;
	}
}

void LogicBase::allocCells(int x0, int x1, int y0, int y1, int density, int prm1, int prm2, int prm3, int prm4)
{
	if (m_isRunning) sendCommand(CMD_VIEW_2_LOGIC_STOP);
	if (x0 < 0) x0 = 0;
	if (x1 >= WORLD_WIDTH) x1 = WORLD_WIDTH;
	if (y0 < 0) y0 = 0;
	if (y1 >= WORLD_HEIGHT) y1 = WORLD_HEIGHT;
	if (density < 0) density = 0;
	if (density > 100) density = 100;

	if (density != 0) {
		int reverseDensity = 100 / density;
		int *mat = getDisplayMat();
		for (int y = y0; y < y1; y++) {
			int yIndex = WORLD_WIDTH * y;
			for (int x = x0; x < x1; x++) {
				mat[yIndex + x] = rand() % reverseDensity == 0;
			}
		}
	} else {
		int *mat = getDisplayMat();
		for (int y = y0; y < y1; y++) {
			int yIndex = WORLD_WIDTH * y;
			for (int x = x0; x < x1; x++) {
				mat[yIndex + x] = 0;
			}
		}
	}
}


void LogicBase::threadFunc()
{
	bool isExit = false;
	while (!isExit) {
		std::chrono::system_clock::time_point  timeStart, timeEnd;
		timeStart = std::chrono::system_clock::now();

		COMMAND cmd = m_cmd;
		m_cmd = CMD_VIEW_2_LOGIC_NONE;
		switch (cmd) {
		case CMD_VIEW_2_LOGIC_NONE:
		default:
			break;
		case CMD_VIEW_2_LOGIC_STOP:
			m_isRunning = false;
			m_isCmdCompleted = true;
			break;
		case CMD_VIEW_2_LOGIC_START:
			m_isRunning = true;
			m_isCmdCompleted = true;
			break;
		case CMD_VIEW_2_LOGIC_STEP:
			m_isRunning = false;
			// send comp after one step done
			break;
		case CMD_SELF_EXIT:
			isExit = true;
			break;
		}

		if (m_isRunning || cmd == CMD_VIEW_2_LOGIC_STEP) {
			m_info.status = 1;
			gameLogic();

			/* swap matrix (calc - display) */
			m_mutexMat.lock();	// wait if view module is copying current display buffer
			int tempId = m_matIdOld;
			m_matIdOld = m_matIdNew;
			m_matIdNew = tempId;
			m_mutexMat.unlock();
			if (cmd == CMD_VIEW_2_LOGIC_STEP) {
				m_isCmdCompleted = true;
			}
		} else {
			m_info.status = 0;
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(1));
		timeEnd = std::chrono::system_clock::now();
		int timeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart).count();
		m_info.calcTime = timeElapsed;
		//printf("fpsCalc = %lf\n", 1000.0 / timeElapsed);
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


void LogicBase::gameLogic()
{
	for (int y = 0; y < WORLD_HEIGHT; y++) {
		for (int x = 0; x < WORLD_WIDTH; x++) {
			m_mat[m_matIdNew][WORLD_WIDTH * y + x] = m_mat[m_matIdOld][WORLD_WIDTH * y + x];
		}
	}
}

ILogic* LogicBase::generateLogic(int algorithm, int width, int height)
{
	switch (algorithm) {
	case ALGORITHM_NORMAL:
		return new LogicNormal(width, height);
	case ALGORITHM_NORMAL_MP:
		return new LogicNormalMP(width, height);
	case ALGORITHM_NORMAL_CUDA:
		return new LogicNormal(width, height);
	case ALGORITHM_NORMAL_NON_TORUS:
		return new LogicNormalNonTorus(width, height);
	case ALGORITHM_NORMAL_NON_TORUS_MP:
		return new LogicNormalNonTorusMP(width, height);
	case ALGORITHM_NORMAL_NON_TORUS_CUDA:
		return new LogicNormal(width, height);
	default:
		return new LogicNormal(width, height);
	}
}

