#include <stdio.h>
#include "WorldContext.h"
#include "WorldView.h"
#include "LogicBase.h"
#include "AnalViewAge.h"
#include "ControllerView.h"
#include "FileAccessor.h"
#include "Values.h"

WorldContext::WorldContext()
{
	WORLD_WIDTH = ControllerView::getInstance()->m_worldWidth;
	WORLD_HEIGHT = ControllerView::getInstance()->m_worldHeight;

	/* 32 align */
	WORLD_WIDTH = ((WORLD_WIDTH + (32 - 1)) / 32) * 32;
	WORLD_HEIGHT = ((WORLD_HEIGHT + (32 - 1)) / 32) * 32;

	m_algorithm = ControllerView::getInstance()->m_worldAlgorithm;
	m_pLogic = LogicBase::generateLogic(m_algorithm, WORLD_WIDTH, WORLD_HEIGHT);
	m_pLogic->initialize();
	m_pView = new WorldView(this);
	m_pAnalView = AnalView::createAppropreateAnalView(this);

	ControllerView::getInstance()->setCurrentWorldContext(this);
}


WorldContext::WorldContext(int width, int height, ALGORITHM algorithm, int windowX, int windowY, int windowWidth, int windowHeight)
{
	WORLD_WIDTH = width;
	WORLD_HEIGHT = height;
	/* 32 align */
	WORLD_WIDTH = ( (WORLD_WIDTH + (32 - 1)) / 32) * 32;
	WORLD_HEIGHT = ((WORLD_HEIGHT + (32 - 1)) / 32) * 32;

	if (algorithm == ALGORITHM_AUTO) algorithm = ControllerView::getInstance()->m_worldAlgorithm;

	m_algorithm = algorithm;
	m_pLogic = LogicBase::generateLogic(algorithm, WORLD_WIDTH, WORLD_HEIGHT);
	m_pLogic->initialize();
	m_pView = new WorldView(this, windowX, windowY, windowWidth, windowHeight);
	m_pAnalView = AnalView::createAppropreateAnalView(this);

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
	char path[255];
	FileAccessor::getFilepath(path);
	return generateFromFile(path, 32, 32);
}

WorldContext* WorldContext::generateFromFile(const char* filename, int width, int height)
{
	WorldContext* worldContext = NULL;
	
	int patternWidth, patternHeight;	// pattern size
	int patternOffsetX, patternOffsetY;	// position in pattern
	int x, y, prm;

	if (FileAccessor::startReadingPattern(filename, &patternWidth, &patternHeight)) {
		width = fmax(width, patternWidth);
		height = fmax(height, patternHeight);
		printf("Create world(%d x %d)\n", width, height);
		worldContext = new WorldContext(width, height);
		width = worldContext->WORLD_WIDTH;
		height = worldContext->WORLD_HEIGHT;
		int prm;
		while (FileAccessor::readPattern(&patternOffsetX, &patternOffsetY, &prm)) {
			int x = width / 2 + patternOffsetX - patternWidth / 2;
			int y = height / 2 - (patternOffsetY - patternHeight / 2) - 1;
			if (prm != 0) {
				worldContext->m_pLogic->setCell(x, y);
			} else {
				worldContext->m_pLogic->clearCell(x, y);
			}
		}
		FileAccessor::stop();
	}
	return worldContext;
}

void WorldContext::saveToFile(WorldContext* context)
{
	char path[255];
	FileAccessor::getFilepath(path);
	FileAccessor::startWritingPattern(path);
	bool isNewline = false;
	int *mat = context->m_pLogic->getDisplayMat();
	for (int y = context->WORLD_HEIGHT - 1; y >= 0; y--) {
		int yIndex = context->WORLD_WIDTH * y;
		for (int x = 0; x < context->WORLD_WIDTH; x++) {
			int prm = mat[yIndex + x];
			FileAccessor::writePattern(prm, isNewline);
			isNewline = false;
		}
		isNewline = true;
	}
	FileAccessor::stop();
}
