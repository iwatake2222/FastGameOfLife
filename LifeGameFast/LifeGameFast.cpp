// LifeGameFast.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include "WindowManager.h"
#include "WorldContext.h"
#include "ControllerView.h"
#include "WorldContextManager.h"


int main(int argc, char *argv[])
{
	WindowManager::getInstance()->init();

	ControllerView::getInstance();
	WorldContextManager::getInstance()->generateWorld();

	WindowManager::getInstance()->startLoop();
	// never reaches here

	return 0;
}
