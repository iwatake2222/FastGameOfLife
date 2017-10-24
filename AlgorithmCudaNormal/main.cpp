/*
* Note:
* To switch output type (dll/exe), Project Configuration Properties -> Configuration Type
* *.exe for unit test
* *.dll for dll library
*/
#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#include "algorithmCudaNormal.h"

using namespace AlgorithmCudaNormal;
void processWithBorderCheck(int *matDst, int*matSrc, int width, int height);
void unitTest();
void runReferenceCode(int *matDst, const int* const matSrc, int loopNum);
void runTargetCode(int *matDst, const int* const matSrc, int loopNum);

const int WIDTH = 512;
const int HEIGHT = 256;

int main()
{
	unitTest();

	getchar();
	return 0;
}

void unitTest()
{
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
	runReferenceCode(matResult0, matSrc, 10);

	/* run target code */
	runTargetCode(matResult1, matSrc, 10);

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
	int *matOld, *matNew;
	matOld = new int[WIDTH * HEIGHT];
	matNew = new int[WIDTH * HEIGHT];
	memcpy(matOld, matSrc, WIDTH * HEIGHT * sizeof(int));

	ALGORITHM_CUDA_NORMAL_PARAM param;
	cudaInitialize(&param, WIDTH, HEIGHT);
	
	cudaProcess(&param, matNew, matOld, WIDTH, HEIGHT, loopNum);

	cudaFinalize(&param);
	memcpy(matDst, matNew, WIDTH * HEIGHT * sizeof(int));
	delete matNew;
	delete matOld;
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
