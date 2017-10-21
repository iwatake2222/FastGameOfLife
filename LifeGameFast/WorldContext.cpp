#include "stdafx.h"
#include "WorldContext.h"
#include "WorldView.h"
#include "WorldLogic.h"
#include "AnalView.h"
#include "ControllerView.h"

#include <windows.h>
#include <Commdlg.h>
#include <tchar.h>
#pragma comment(lib, "Comdlg32.lib")
#pragma comment(lib, "User32.lib")


WorldContext::WorldContext()
{
	/* fixed during lifespan */
	WORLD_WIDTH = ControllerView::getInstance()->m_worldWidth;
	WORLD_HEIGHT = ControllerView::getInstance()->m_worldHeight;

	m_pLogic = new WorldLogic(WORLD_WIDTH, WORLD_HEIGHT);
	m_pView = new WorldView(this);
	m_pAnalView = new AnalView(this);

	ControllerView::getInstance()->setCurrentWorldContext(this);
}


WorldContext::WorldContext(int width, int height, int algorithm, int windowX, int windowY, int windowWidth, int windowHeight)
{
	/* fixed during lifespan */
	WORLD_WIDTH = width;
	WORLD_HEIGHT = height;

	switch (algorithm) {
	case ALGORITHM_NORMAL:
	case ALGORITHM_NORMAL_CUDA:
	default:
		m_pLogic = new WorldLogic(WORLD_WIDTH, WORLD_HEIGHT);
		break;
	}
	m_pView = new WorldView(this, windowX, windowY, windowWidth, windowHeight);
	m_pAnalView = new AnalView(this);

	ControllerView::getInstance()->setCurrentWorldContext(this);
}

WorldContext::~WorldContext()
{
	delete m_pAnalView;
	delete m_pView;
	delete m_pLogic;
}

WorldContext* WorldContext::generateFromFile()
{
	OPENFILENAME ofn;
	WCHAR path[MAX_PATH];
	WCHAR name[MAX_PATH];
	memset(path, '\0', sizeof(path));
	memset(name, '\0', sizeof(name));

	memset(&ofn, 0, sizeof(OPENFILENAME));
	ofn.lStructSize = sizeof(OPENFILENAME);
	ofn.lpstrFilter = TEXT("(*.txt, *.csv)\0*.txt;*.csv\0");
	ofn.lpstrFile = path;
	ofn.nMaxFile = MAX_PATH;
	ofn.lpstrFileTitle = name;
	ofn.nMaxFileTitle = MAX_PATH;
	ofn.Flags = OFN_FILEMUSTEXIST;
	if (GetOpenFileName(&ofn)) {
		return generateFromFile(ofn.lpstrFile);
	} else {
	}
	return NULL;
}

WorldContext* WorldContext::generateFromFile(LPWSTR filename)
{
	FILE *fp;
	_wfopen_s(&fp, filename, L"r");
	int width, height;
	int x, y, prm;
	fwscanf_s(fp, L"%d,%d", &width, &height);
	printf("Create world(%d x %d)\n", width, height);
	WorldContext* worldContext = new WorldContext(width, height);

	while (fwscanf_s(fp, L"%d,%d,%d", &x, &y, &prm) != EOF) {
		printf("%d %d %d\n", x, y, prm);
		worldContext->m_pLogic->setCell(x, y);
	}
	fclose(fp);
	return worldContext;
}

void WorldContext::saveToFile(WorldContext* context)
{
	OPENFILENAME ofn;
	WCHAR path[MAX_PATH];
	WCHAR name[MAX_PATH];
	memset(path, '\0', sizeof(path));
	memset(name, '\0', sizeof(name));

	memset(&ofn, 0, sizeof(OPENFILENAME));
	ofn.lStructSize = sizeof(OPENFILENAME);
	ofn.lpstrFilter = TEXT("(*.txt, *.csv)\0*.txt;*.csv\0");
	ofn.lpstrFile = path;
	ofn.nMaxFile = MAX_PATH;
	ofn.lpstrFileTitle = name;
	ofn.nMaxFileTitle = MAX_PATH;
	if (GetOpenFileName(&ofn)) {
		FILE *fp;
		_wfopen_s(&fp, ofn.lpstrFile, L"w");
		
		fwprintf_s(fp, L"%d,%d\n", context->WORLD_WIDTH, context->WORLD_HEIGHT);

		int x, y, prm;
		int *mat = new int[context->WORLD_WIDTH * context->WORLD_HEIGHT];
		context->m_pLogic->copyDisplayMat(mat);
		for (int y = 0; y < context->WORLD_HEIGHT; y++) {
			int yIndex = context->WORLD_WIDTH * y;
			for (int x = 0; x < context->WORLD_WIDTH; x++) {
				prm = mat[yIndex + x];
				if (prm != 0) {
					fwprintf_s(fp, L"%d,%d,%d\n", x, y, prm);
				}
			}
		}
		delete mat;
		fclose(fp);
	} else {
	}
}
