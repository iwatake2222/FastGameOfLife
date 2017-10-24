#include "stdio.h"

inline void updateCell(int* matDst, int *matSrc, int x, int yLine, int cnt)
{
	/* Note: yLine is index of array (yLine = y*width) */
	if (matSrc[yLine + x] == 0) {
		if (cnt == 3) {
			// birth
			matDst[yLine + x] = 1;
		} else {
			// keep dead
			matDst[yLine + x] = 0;
		}
	} else {
		if (cnt <= 2 || cnt >= 5) {
			// die
			matDst[yLine + x] = 0;
		} else {
			// keep alive (age++)
			matDst[yLine + x] = matSrc[yLine + x] + 1;
		}
	}
}

void processWithBorderCheck(int *matDst, int*matSrc, int width, int height)
{
	for (int y = 0; y < height; y++) {
		int yLine = width * y;
		for (int x = 0; x < width; x++) {
			int cnt = 0;
			for (int yy = y - 1; yy <= y + 1; yy++) {
				int roundY = yy;
				if (roundY >= height) roundY = 0;
				if (roundY < 0) roundY = height - 1;
				for (int xx = x - 1; xx <= x + 1; xx++) {
					int roundX = xx;
					if (roundX >= width) roundX = 0;
					if (roundX < 0) roundX = width - 1;
					if (matSrc[width * roundY + roundX] != 0) {
						cnt++;
					}
				}
			}
			updateCell(matDst, matSrc, x, yLine, cnt);
		}
	}
}

