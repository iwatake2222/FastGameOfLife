
#include "stdafx.h"
#include "WorldLogic.h"


WorldLogic::WorldLogic(int worldWidth, int worldHeight)
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
	std::thread t(&WorldLogic::loop, this);
	m_thread.swap(t);
}

WorldLogic::~WorldLogic()
{
	m_cmd = CMD_SELF_EXIT;
	if (m_thread.joinable()) m_thread.join();

	for (int i = 0; i < 2; i++) {
		delete m_mat[i];
		m_mat[i] = 0;
	}
}


void WorldLogic::sendCommand(COMMAND cmd) {
	m_isCmdCompleted = false;
	m_cmd = cmd;
	// wait until logic thread completes the command
	// todo: rendezvous / mutex
	while (!m_isCmdCompleted) std::this_thread::yield();
}

void WorldLogic::toggleRun()
{
	if (m_isRunning) {
		sendCommand(CMD_VIEW_2_LOGIC_STOP);
	} else {
		sendCommand(CMD_VIEW_2_LOGIC_START);
	}
}

void WorldLogic::stepRun()
{
	sendCommand(CMD_VIEW_2_LOGIC_STEP);
}

int* WorldLogic::getDisplayMat() {
	/* this is unsafe. the matrix retrieved might be updated while using */
	return m_mat[m_matIdOld];
}

void WorldLogic::copyDisplayMat(int* matOut) {
	m_mutexMat.lock();
	memcpy(matOut, m_mat[m_matIdOld], sizeof(int) * WORLD_WIDTH * WORLD_HEIGHT);
	m_mutexMat.unlock();
}

void WorldLogic::toggleCell(int worldX, int worldY)
{
	if (m_isRunning) sendCommand(CMD_VIEW_2_LOGIC_STOP);
	int *mat = getDisplayMat();
	if (worldX >= 0 && worldX < WORLD_WIDTH && worldY >= 0 && worldY < WORLD_HEIGHT) {
		mat[WORLD_WIDTH * worldY + worldX] = mat[WORLD_WIDTH * worldY + worldX] == CELL_DEAD ? CELL_ALIVE : CELL_DEAD;
	}
}

void WorldLogic::setCell(int worldX, int worldY)
{
	if (m_isRunning) sendCommand(CMD_VIEW_2_LOGIC_STOP);
	int *mat = getDisplayMat();
	if (worldX >= 0 && worldX < WORLD_WIDTH && worldY >= 0 && worldY < WORLD_HEIGHT) {
		mat[WORLD_WIDTH * worldY + worldX] = CELL_ALIVE;
	}
}

void WorldLogic::clearCell(int worldX, int worldY)
{
	if (m_isRunning) sendCommand(CMD_VIEW_2_LOGIC_STOP);
	int *mat = getDisplayMat();
	if (worldX >= 0 && worldX < WORLD_WIDTH && worldY >= 0 && worldY < WORLD_HEIGHT) {
		mat[WORLD_WIDTH * worldY + worldX] = CELL_DEAD;
	}
}

void WorldLogic::allocCells(int x0, int x1, int y0, int y1, int density, int prm1, int prm2, int prm3, int prm4)
{
	if (m_isRunning) sendCommand(CMD_VIEW_2_LOGIC_STOP);
	if (x0 < 0) x0 = 0;
	if (x1 >= WORLD_WIDTH) x1 = WORLD_WIDTH;
	if (y0 < 0) y0 = 0;
	if (y1 >= WORLD_WIDTH) y1 = WORLD_WIDTH;
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


void WorldLogic::loop()
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
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(1));
		timeEnd = std::chrono::system_clock::now();
		int timeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart).count();
		//printf("fpsCalc = %lf\n", 1000.0 / timeElapsed);
	}
}

void WorldLogic::gameLogic()
{
	for (int y = 0; y < WORLD_HEIGHT; y++) {
		for (int x = 0; x < WORLD_WIDTH; x++) {
			int cnt = 0;
			for (int yy = y - 1; yy <= y + 1; yy++) {
				int roundY = yy;
				if (roundY >= WORLD_HEIGHT) roundY = 0;
				if (roundY < 0) roundY = WORLD_HEIGHT - 1;
				for (int xx = x - 1; xx <= x + 1; xx++) {
					int roundX = xx;
					if (roundX >= WORLD_WIDTH) roundX = 0;
					if (roundX < 0) roundX = WORLD_WIDTH - 1;
					if (m_mat[m_matIdOld][WORLD_WIDTH * roundY + roundX] != 0) {
						cnt++;
					}
				}
			}
			if (m_mat[m_matIdOld][WORLD_WIDTH * y + x] == 0) {
				if (cnt == 3) {
					// birth
					m_mat[m_matIdNew][WORLD_WIDTH * y + x] = 1;
				} else {
					m_mat[m_matIdNew][WORLD_WIDTH * y + x] = 0;
				}
			} else {
				if (cnt <= 2 || cnt >= 5) {
					// die
					m_mat[m_matIdNew][WORLD_WIDTH * y + x] = 0;
				} else {
					m_mat[m_matIdNew][WORLD_WIDTH * y + x] = 1;
				}
			}
		}
	}
}
