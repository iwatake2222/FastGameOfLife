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
static const double COLOR_3D_ALIVE_GROUP_A[] = { 2.0, 0.0, 0.2 };
static const double COLOR_3D_ALIVE_GROUP_B[] = { 0.0, 2.0, 0.2 };
static const double COLOR_3D_ALIVE_GROUP_A75[] = { 1.5, 0.5, 0.2 };
static const double COLOR_3D_ALIVE_GROUP_HALF[] = { 1.0, 1.0, 0.2 };
static const double COLOR_3D_ALIVE_GROUP_B75[] = { 0.5, 1.5, 0.2 };
static const double COLOR_3D_VIRUS_NONE[] = { 0.0, 0.0, 0.0 };
static const double COLOR_3D_VIRUS_HEALTHY[] = { 1.0, 1.0, 1.0 };
static const double COLOR_3D_VIRUS_INFECTED[] = { 1.0, 0.2, 0.2 };
static const double COLOR_3D_VIRUS_ONSET[] = { 1.0, 0.5, 0.5 };
static const double COLOR_3D_VIRUS_CONTAGIUS[] = { 1.0, 1.0, 0.2 };
static const double COLOR_3D_VIRUS_IMMUNIZED[] = { 0.2, 1.0, 0.2 };


typedef enum {
	ALGORITHM_NORMAL = 0,
	ALGORITHM_NORMAL_MP,
	ALGORITHM_NORMAL_NON_TORUS,
	ALGORITHM_NORMAL_NON_TORUS_MP,
	//ALGORITHM_NORMAL_NON_TORUS_CUDA,
	ALGORITHM_GROUP_MP,
	ALGORITHM_NORMAL_CUDA,
	ALGORITHM_GROUP_CUDA,
	ALGORITHM_NUM,
	ALGORITHM_AUTO = 99,	// from controller view
} ALGORITHM;

static const int DEFAULT_WORLD_WIDTH = 256;
static const int DEFAULT_WORLD_HEIGHT = 256;
//static const int DEFAULT_WORLD_WIDTH = 512;
//static const int DEFAULT_WORLD_HEIGHT = 512;
static const int DEFAULT_CELLS_DENSITY = 10;
static const int DEFAULT_DRAW_INTERVAL = 1;
static const int DEFAULT_SKIP_NUM = 1;
static const ALGORITHM DEFAULT_ALGORITHM = ALGORITHM_NORMAL;

static const int DEFAULT_WINDOW_WIDTH = 640;
static const int DEFAULT_WINDOW_HEIGHT = 480;
static const int DEFAULT_WINDOW_X = 300;
static const int DEFAULT_WINDOW_Y = 0;


