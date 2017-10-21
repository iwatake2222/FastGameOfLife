#pragma once
static const int INVALID_NUM = 9999;
static const double COLOR_3D_GRID[] = { 0.4, 0.4, 0.4 };
static const double COLOR_3D_DEAD[] = { 0.0, 0.0, 0.0 };
static const double COLOR_3D_BK[] = { 0.0, 0.0, 0.0 };
static const double COLOR_3D_NORMAL[] = { 1.0, 1.0, 1.0 };
static const double COLOR_3D_ALIVE[] = { 0.8, 0.8, 0.8 };
static const double COLOR_3D_ALIVE0[] = { 0.8, 0.2, 0.2 };
static const double COLOR_3D_ALIVE1[] = { 0.8, 0.8, 0.2 };
static const double COLOR_3D_ALIVE2[] = { 0.2, 0.8, 0.2 };
static const double COLOR_3D_ALIVE3[] = { 0.2, 0.8, 0.8 };
static const double COLOR_3D_ALIVE4[] = { 0.2, 0.2, 0.8 };

static const int DEFAULT_WORLD_WIDTH = 512;
static const int DEFAULT_WORLD_HEIGHT = 512;
static const int DEFAULT_CELLS_DENSITY = 20;

static const int DEFAULT_WINDOW_WIDTH = 640;
static const int DEFAULT_WINDOW_HEIGHT = 480;
static const int DEFAULT_WINDOW_X = 0;
static const int DEFAULT_WINDOW_Y = 0;

static const int ALGORITHM_NORMAL = 0;
static const int ALGORITHM_NORMAL_CUDA = 1;
