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

#include "algorithmCudaGroup.h"
#include "algorithmCudaGroupInternal.h"

using namespace AlgorithmCudaGroup;
void processWithBorderCheck(DNA *matDst, DNA *matSrc, int width, int height);
void unitTest(int seed, int repeatNum);
void runForAnalysis();
void runReferenceCode(DNA *matDst, DNA* matSrc, int repeatNum);
void runTargetCode(DNA *matDst, DNA* matSrc, int repeatNum);

const int WIDTH = 1 << 11;
const int HEIGHT = 1 << 9;
const int REPEAT_NUM = 10;

int main()
{
	for(int i= 1; i < 10; i++) unitTest(i, i%2 + REPEAT_NUM);
	//unitTest(4, 1);
	//runForAnalysis();

	printf("done\n");
	//getchar();
	return 0;
}

void unitTest(int seed, int repeatNum)
{
	DNA *matSrc, *matResult0, *matResult1;
	matSrc = new DNA[WIDTH * HEIGHT];
	matResult0 = new DNA[WIDTH * HEIGHT];
	matResult1 = new DNA[WIDTH * HEIGHT];

	/* populate input test data*/
	srand(seed);
	for (int i = 0; i < WIDTH * HEIGHT; i++) {
		matSrc[i].age = (rand() % 10 == 0);
		//matSrc[i].group = (i % 2 == 0 ? PURE_GROUP_A : PURE_GROUP_B);
		matSrc[i].group = 0;
	}

	//printMatrix(matSrc, WIDTH, HEIGHT);

	/* run reference code */
	runReferenceCode(matResult0, matSrc, repeatNum);

	//printMatrix(matResult0, WIDTH, HEIGHT);

	/* run target code */
	runTargetCode(matResult1, matSrc, repeatNum);

	//printMatrix(matResult1, WIDTH, HEIGHT);

	/* compare the results */
	int checkIndex = 0;
	for (checkIndex = 0; checkIndex < WIDTH * HEIGHT; checkIndex++) {
		if (matResult0[checkIndex].age != matResult1[checkIndex].age || (matResult0[checkIndex].age != 0 && matResult0[checkIndex].group != matResult1[checkIndex].group) ) {
			printf("šššerrorššš: (%d, %d)\n", checkIndex%WIDTH, checkIndex / WIDTH);
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


void runReferenceCode(DNA *matDst, DNA *matSrc, int repeatNum)
{
	DNA *matOld, *matNew;
	matOld = new DNA[WIDTH * HEIGHT];
	matNew = new DNA[WIDTH * HEIGHT];
	memcpy(matOld, matSrc, WIDTH * HEIGHT * sizeof(DNA));

	for (int i = 0; i < repeatNum; i++) {
		processWithBorderCheck(matNew, matOld, WIDTH, HEIGHT);
		DNA *matTemp = matNew;
		matNew = matOld;
		matOld = matTemp;
	}

	memcpy(matDst, matOld, WIDTH * HEIGHT * sizeof(DNA));
	delete matNew;
	delete matOld;
}

void runTargetCode(DNA *matDst, DNA *matSrc, int repeatNum)
{
	ALGORITHM_CUDA_GROUP_PARAM param;
	cudaInitialize(&param, WIDTH, HEIGHT);
	for (int y = 0; y < HEIGHT; y++) {
		memcpy(param.hostMatSrc + ((y + MEMORY_MARGIN) * (WIDTH + 2 * MEMORY_MARGIN)) + MEMORY_MARGIN, matSrc + WIDTH * y, WIDTH * sizeof(DNA));
	}


	std::chrono::system_clock::time_point  timeStart, timeEnd;
	timeStart = std::chrono::system_clock::now();
	if (repeatNum > 7) {	// check multple calls
		cudaProcess(&param, WIDTH, HEIGHT, 1);
		cudaProcess(&param, WIDTH, HEIGHT, 1);
		cudaProcess(&param, WIDTH, HEIGHT, 2);
		cudaProcess(&param, WIDTH, HEIGHT, 3);
		cudaProcess(&param, WIDTH, HEIGHT, repeatNum - 7);
	} else {
		cudaProcess(&param, WIDTH, HEIGHT, repeatNum);
	}
	timeEnd = std::chrono::system_clock::now();
	int timeElapsed = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();
	printf("total process time = %d [usec]\n", timeElapsed);

	for (int y = 0; y < HEIGHT; y++) {
		memcpy(matDst + WIDTH * y, param.hostMatSrc + ((y + MEMORY_MARGIN) * (WIDTH + 2 * MEMORY_MARGIN)) + MEMORY_MARGIN, WIDTH * sizeof(DNA));
	}

	cudaFinalize(&param);
}

void printMatrix(DNA *mat, int width, int height)
{
	printf("\n");
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			printf("%d(%d) ", mat[y*width + x].age, mat[y*width + x].group);
		}
		printf("\n");
	}
	printf("\n");
}
