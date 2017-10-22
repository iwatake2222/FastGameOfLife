/*
* Note:
* To switch output type (dll/exe), Project Configuration Properties -> Configuration Type
* *.exe for unit test
* *.dll for dll library
*/
#include "stdio.h"
#include "stdlib.h"

#include "algorithmCudaNormal.h"

using namespace AlgorithmCudaNormal;

void unitTest();

int main()
{
	unitTest();

	getchar();
	return 0;
}

void unitTest()
{
	int width = 64;
	int height = 32;
	int *src, *dst;
	allocManaged(&src, width * height * sizeof(int));
	allocManaged(&dst, width * height * sizeof(int));
	for (int i = 0; i < width * height; i++) {
		src[i] = i;
		dst[i] = 0;
	}
	logicForOneGeneration(dst, src, width, height);

	//for (int i = 0; i < width * height; i++) {
	//	printf("%d ", dst[i]);
	//}

	freeManaged(src);
	freeManaged(dst);

}