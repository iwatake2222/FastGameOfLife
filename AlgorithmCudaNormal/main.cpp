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


using namespace AlgorithmCudaNormal;
void processWithBorderCheck(int *matDst, int*matSrc, int width, int height);
void unitTest();
void runForAnalysis();
void runReferenceCode(int *matDst, const int* const matSrc, int loopNum);
void runTargetCode(int *matDst, const int* const matSrc, int loopNum);

const int WIDTH = 4096;
const int HEIGHT = 4096;

int main()
{
	unitTest();
	runForAnalysis();

	printf("done\n");
	getchar();
	return 0;
}

void runForAnalysis()
{
	int *matSrc, *matResult0;
	matSrc = new int[WIDTH * HEIGHT];
	matResult0 = new int[WIDTH * HEIGHT];

	// operate several times
	for (int i = 0; i < 1; i++) {
		/* populate input test data*/
		srand(i);
		for (int i = 0; i < WIDTH * HEIGHT; i++) {
			matSrc[i] = (rand() % 10 == 0);
		}

		/* run target code */
		runTargetCode(matResult0, matSrc, 1);
	}

	delete matSrc;
	delete matResult0;
}


void unitTest()
{
	const static int RUN_GENERATION_NUM = 10;
	int *matSrc, *matResult0, *matResult1;
	matSrc = new int[WIDTH * HEIGHT];
	matResult0 = new int[WIDTH * HEIGHT];
	matResult1 = new int[WIDTH * HEIGHT];

	/* populate input test data*/
	srand(10);
	for (int i = 0; i < WIDTH * HEIGHT; i++) {
		matSrc[i] = (rand() % 10 == 0);
	}

	/* run reference code */
	runReferenceCode(matResult0, matSrc, RUN_GENERATION_NUM);

	/* run target code */
	runTargetCode(matResult1, matSrc, RUN_GENERATION_NUM);

	/* compare the results */
	int checkIndex = 0;
	for (checkIndex = 0; checkIndex < WIDTH * HEIGHT; checkIndex++) {
		if (matResult0[checkIndex] != matResult1[checkIndex]) {
			printf("NG: %d\n", checkIndex);
			break;
		}
	}
	if (checkIndex == WIDTH * HEIGHT) {
		printf("OK\n");
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
	memcpy(param.hostMatSrc, matSrc, WIDTH * HEIGHT * sizeof(int));
	
	std::chrono::system_clock::time_point  timeStart, timeEnd;
	timeStart = std::chrono::system_clock::now();
	for (int i = 0; i < loopNum; i++) {
		cudaProcess(&param, WIDTH, HEIGHT);
	}
	timeEnd = std::chrono::system_clock::now();
	int timeElapsed = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();
	printf("total process time = %d [usec]\n", timeElapsed);

	memcpy(matDst, param.hostMatSrc, WIDTH * HEIGHT * sizeof(int));

	cudaFinalize(&param);
}

void printMatrix(int *mat, int width, int height)
{
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			printf("%d ", mat[y*width + x]);
		}
		printf("\n");
	}
}
