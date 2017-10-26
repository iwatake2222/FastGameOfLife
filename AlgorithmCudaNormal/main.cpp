/*
* Note:
* To switch output type (dll/exe), Project Configuration Properties -> Configuration Type
* *.exe for unit test
* *.dll for dll library
*/
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <future>

#include "algorithmCudaNormal.h"
#include "algorithmCudaNormalInternal.h"


using namespace AlgorithmCudaNormal;
void processWithBorderCheck(int *matDst, int*matSrc, int width, int height);
void unitTest(int seed, int repeatNum);
void runForAnalysis();
void runReferenceCode(int *matDst, const int* const matSrc, int loopNum);
void runTargetCode(int *matDst, const int* const matSrc, int loopNum);

const int WIDTH = 1 << 12;
const int HEIGHT = 1 << 12;

int main()
{
	//for(int i= 1; i < 100; i++) unitTest(i, i%2 + 2);
	runForAnalysis();

	printf("done\n");
	//getchar();
	return 0;
}

void runForAnalysis()
{
	int *matSrc, *matResult0;
	matSrc = new int[WIDTH * HEIGHT];
	matResult0 = new int[WIDTH * HEIGHT];

	/* populate input test data*/
	for (int i = 0; i < WIDTH * HEIGHT; i++) {
		matSrc[i] = (rand() % 10 == 0);
	}

	/* run target code */
	runTargetCode(matResult0, matSrc, 10);

	delete matSrc;
	delete matResult0;
}


void unitTest(int seed, int repeatNum)
{
	int *matSrc, *matResult0, *matResult1;
	matSrc = new int[WIDTH * HEIGHT];
	matResult0 = new int[WIDTH * HEIGHT];
	matResult1 = new int[WIDTH * HEIGHT];

	/* populate input test data*/
	srand(seed);
	for (int i = 0; i < WIDTH * HEIGHT; i++) {
		matSrc[i] = (rand() % 10 == 0);
	}
	
	//printMatrix(matSrc, WIDTH, HEIGHT);

	/* run reference code */
	runReferenceCode(matResult0, matSrc, repeatNum);

	/* run target code */
	runTargetCode(matResult1, matSrc, repeatNum);

	/* compare the results */
	int checkIndex = 0;
	for (checkIndex = 0; checkIndex < WIDTH * HEIGHT; checkIndex++) {
		if (matResult0[checkIndex] != matResult1[checkIndex]) {
			printf("šššerrorššš: (%d, %d)\n", checkIndex%WIDTH, checkIndex/WIDTH);
			break;
		}
	}
	if (checkIndex == WIDTH * HEIGHT) {
		printf("pass\n");
	}

	delete matSrc;
	delete matResult0;
	delete matResult1;
}

void runReferenceCode(int *matDst, const int* const matSrc, int loopNum)
{
	int *matOld, *matNew;
	matOld = new int[WIDTH * HEIGHT];
	matNew = new int[WIDTH * HEIGHT];
	memcpy(matOld, matSrc, WIDTH * HEIGHT * sizeof(int));

	for (int i = 0; i < loopNum; i++) {
		processWithBorderCheck(matNew, matOld, WIDTH, HEIGHT);
		int *matTemp = matNew;
		matNew = matOld;
		matOld = matTemp;
	}

	memcpy(matDst, matOld, WIDTH * HEIGHT * sizeof(int));
	delete matNew;
	delete matOld;
}

void runTargetCode(int *matDst, const int* const matSrc, int loopNum)
{
	ALGORITHM_CUDA_NORMAL_PARAM param;
	cudaInitialize(&param, WIDTH, HEIGHT);
	for (int y = 0; y < HEIGHT; y++) {
		memcpy(param.hostMatSrc + ( (y + MEMORY_MARGIN) * (WIDTH + 2 * MEMORY_MARGIN)) + MEMORY_MARGIN, matSrc + WIDTH * y, WIDTH * sizeof(int));
	}
	
	
	std::chrono::system_clock::time_point  timeStart, timeEnd;
	timeStart = std::chrono::system_clock::now();
	for (int i = 0; i < loopNum; i++) {
		cudaProcess(&param, WIDTH, HEIGHT);
	}
	timeEnd = std::chrono::system_clock::now();
	int timeElapsed = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();
	printf("total process time = %d [usec]\n", timeElapsed);

	for (int y = 0; y < HEIGHT; y++) {
		memcpy(matDst + WIDTH * y, param.hostMatSrc + ((y + MEMORY_MARGIN) * (WIDTH + 2 * MEMORY_MARGIN)) + MEMORY_MARGIN, WIDTH * sizeof(int));
	}

	cudaFinalize(&param);
}

void printMatrix(int *mat, int width, int height)
{
	printf("\n");
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			printf("%d ", mat[y*width + x]);
		}
		printf("\n");
	}
	printf("\n");
}
