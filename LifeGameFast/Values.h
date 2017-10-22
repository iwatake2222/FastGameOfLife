#pragma once
static const int INVALID_NUM = 9999;
static const double COLOR_3D_GRID[] = { 0.4, 0.4, 0.4 };
static const double COLOR_3D_DEAD[] = { 0.0, 0.0, 0.0 };
static const double COLOR_3D_BK[] = { 0.0, 0.0, 0.0 };
static const double COLOR_3D_NORMAL[] = { 1.0, 1.0, 1.0 };
static const double COLOR_3D_ALIVE[] = { 0.8, 0.8, 0.8 };
static const double COLOR_3D_ALIVE0[] = { 1.0, 0.2, 0.2 };
static const double COLOR_3D_ALIVE1[] = { 1.0, 0.5, 0.5 };
static const double COLOR_3D_ALIVE2[] = { 1.0, 1.0, 0.2 };
static const double COLOR_3D_ALIVE3[] = { 0.2, 1.0, 0.2 };
static const double COLOR_3D_ALIVE4[] = { 0.2, 0.8, 1.0 };


typedef enum {
	ALGORITHM_NORMAL = 0,
	ALGORITHM_NORMAL_MP,
	ALGORITHM_NORMAL_CUDA,
	ALGORITHM_NORMAL_NON_TORUS,
	ALGORITHM_NORMAL_NON_TORUS_MP,
	ALGORITHM_NORMAL_NON_TORUS_CUDA,
	ALGORITHM_NUM,
	ALGORITHM_AUTO = 99,	// from controller view
} ALGORITHM;

static const int DEFAULT_WORLD_WIDTH = 2048;
static const int DEFAULT_WORLD_HEIGHT = 2048;
//static const int DEFAULT_WORLD_WIDTH = 512;
//static const int DEFAULT_WORLD_HEIGHT = 512;
static const int DEFAULT_CELLS_DENSITY = 20;
static const int DEFAULT_VIEW_INTERVAL = 1;
static const ALGORITHM DEFAULT_ALGORITHM = ALGORITHM_NORMAL_CUDA;

static const int DEFAULT_WINDOW_WIDTH = 640;
static const int DEFAULT_WINDOW_HEIGHT = 480;
static const int DEFAULT_WINDOW_X = 0;
static const int DEFAULT_WINDOW_Y = 0;


