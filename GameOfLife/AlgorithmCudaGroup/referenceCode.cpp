#include "stdio.h"
#include "algorithmCudaGroup.h"

using namespace AlgorithmCudaGroup;

inline void updateCell(DNA* matDst, DNA *matSrc, int x, int yLine, int cnt, int group)
{
	/* Note: yLine is index of array (yLine = y*width) */
	if (matSrc[yLine + x].age == 0) {
		if (cnt == 3) {
			// birth
			matDst[yLine + x].age = 1;
			matDst[yLine + x].group = group / 3;
		} else {
			// keep dead
			matDst[yLine + x].age = 0;
		}
	} else {
		if (cnt <= 2 || cnt >= 5) {
			// die
			matDst[yLine + x].age = 0;
		} else {
			// keep alive (age++)
			matDst[yLine + x].age = matSrc[yLine + x].age + 1;
			matDst[yLine + x].group = matSrc[yLine + x].group;
		}
	}
}

void processWithBorderCheck(DNA *matDst, DNA *matSrc, int width, int height)
{
	for (int y = 0; y < height; y++) {
		int yLine = width * y;
		for (int x = 0; x < width; x++) {
			int cnt = 0;
			int group = 0;
			for (int yy = y - 1; yy <= y + 1; yy++) {
				int roundY = yy;
				if (roundY >= height) roundY = 0;
				if (roundY < 0) roundY = height - 1;
				for (int xx = x - 1; xx <= x + 1; xx++) {
					int roundX = xx;
					if (roundX >= width) roundX = 0;
					if (roundX < 0) roundX = width - 1;
					if (matSrc[width * roundY + roundX].age != 0) {
						cnt++;
						group += matSrc[width * roundY + roundX].group;
					}
				}
			}
			//if (x == 8 && y == 0) {
			//	printf("ref:\n");
			//	printf("cnt = %d\n", cnt);
			//	printf("%d %d %d\n", matSrc[(y - 1)*width + x - 1], matSrc[(y - 1)*width + x - 0], matSrc[(y - 1)*width + x + 1]);
			//	printf("%d %d %d\n", matSrc[(y - 0)*width + x - 1], matSrc[(y - 0)*width + x - 0], matSrc[(y - 0)*width + x + 1]);
			//	printf("%d %d %d\n", matSrc[(y + 1)*width + x - 1], matSrc[(y + 1)*width + x - 0], matSrc[(y + 1)*width + x + 1]);
			//}
			updateCell(matDst, matSrc, x, yLine, cnt, group);
		}
	}
}
