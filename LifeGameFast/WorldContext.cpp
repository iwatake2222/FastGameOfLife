#include "stdafx.h"
#include "WorldContext.h"
#include "WorldView.h"
#include "LogicBase.h"
#include "AnalView.h"
#include "ControllerView.h"
#include "FileAccessor.h"
#include "Values.h"

#include <windows.h>
#include <Commdlg.h>
#include <tchar.h>
#pragma comment(lib, "Comdlg32.lib")
#pragma comment(lib, "User32.lib")


WorldContext::WorldContext()
{
	WORLD_WIDTH = ControllerView::getInstance()->m_worldWidth;
	WORLD_HEIGHT = ControllerView::getInstance()->m_worldHeight;

	/* 32 align */
	WORLD_WIDTH = (WORLD_WIDTH / 32) * 32;
	WORLD_HEIGHT = (WORLD_HEIGHT / 32) * 32;

	m_pLogic = LogicBase::generateLogic(ControllerView::getInstance()->m_worldAlgorithm, WORLD_WIDTH, WORLD_HEIGHT);
	m_pLogic->initialize();
	m_pView = new WorldView(this);
	m_pAnalView = new AnalView(this);

	ControllerView::getInstance()->setCurrentWorldContext(this);
}


WorldContext::WorldContext(int width, int height, ALGORITHM algorithm, int windowX, int windowY, int windowWidth, int windowHeight)
{
	WORLD_WIDTH = width;
	WORLD_HEIGHT = height;
	/* 32 align */
	WORLD_WIDTH = (WORLD_WIDTH / 32) * 32;
	WORLD_HEIGHT = (WORLD_HEIGHT / 32) * 32;

	if (algorithm == ALGORITHM_AUTO) algorithm = ControllerView::getInstance()->m_worldAlgorithm;
	m_pLogic = LogicBase::generateLogic(algorithm, WORLD_WIDTH, WORLD_HEIGHT);
	m_pLogic->initialize();
	m_pView = new WorldView(this, windowX, windowY, windowWidth, windowHeight);
	m_pAnalView = new AnalView(this);

	ControllerView::getInstance()->setCurrentWorldContext(this);
}

WorldContext::~WorldContext()
{
	delete m_pAnalView;
	delete m_pView;
	m_pLogic->exit();
	delete m_pLogic;
}

WorldContext* WorldContext::generateFromFile()
{
	WCHAR path[MAX_PATH];
	if (FileAccessor::getFilepath(path, TEXT("(*.txt, *.csv)\0*.txt;*.csv\0"))) {
		return generateFromFile(path);
	}
	return NULL;
}

WorldContext* WorldContext::generateFromFile(LPWSTR filename)
{
	WorldContext* worldContext = NULL;
	
	int width, height;
	int x, y, prm;

	if (FileAccessor::startReadingWorld(filename, &width, &height)) {
		printf("Create world(%d x %d)\n", width, height);
		worldContext = new WorldContext(width, height);

		while (FileAccessor::readPosition(&x, &y, &prm)) {
			y = (worldContext->WORLD_HEIGHT - 1 - y);	// revert Y
			printf("%d %d %d\n", x, y, prm);
			if (prm != 0) {
				worldContext->m_pLogic->setCell(x, y);
			}
		}

		FileAccessor::stop();
	}
	return worldContext;
}

void WorldContext::saveToFile(WorldContext* context)
{
	WCHAR path[MAX_PATH];
	if (!FileAccessor::getFilepath(path, TEXT("(*.txt, *.csv)\0*.txt;*.csv\0"))) {
		return;
	}

	if (!FileAccessor::startWritingWorld(path, context->WORLD_WIDTH, context->WORLD_HEIGHT)) {
		FileAccessor::stop();
		return;
	}

	int *mat = context->m_pLogic->getDisplayMat();
	for (int y = 0; y < context->WORLD_HEIGHT; y++) {
		int yIndex = context->WORLD_WIDTH * (context->WORLD_HEIGHT - 1 - y);	// revert Y
		for (int x = 0; x < context->WORLD_WIDTH; x++) {
			int prm = mat[yIndex + x];
			if (prm != 0) {
				FileAccessor::writePosition(x, y, prm);
			}
		}
	}
	FileAccessor::stop();
}

