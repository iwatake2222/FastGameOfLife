#include <stdio.h>
#include "FileAccessor.h"


FILE * FileAccessor::m_fp;
int FileAccessor::m_patternOffsetX, FileAccessor::m_patternOffsetY;
int FileAccessor::m_lastReadChar;
int FileAccessor::m_currentX, FileAccessor::m_currentY;

FileAccessor::FileAccessor()
{
}


FileAccessor::~FileAccessor()
{
}

void FileAccessor::skipNewLine()
{
	/* treating new line code (\r, \n, \r\n, \n\r) */
	char readChar;
	do {
		readChar = fgetwc(m_fp);
	} while (readChar == '\r' || readChar == '\n');
	if (readChar != EOF) fseek(m_fp, -1L, SEEK_CUR);
	// current position is the top of new line
}

void FileAccessor::skipComment()
{
	char readChar;
	readChar = fgetwc(m_fp);
	while (readChar == '#' || readChar == '!') {
		do {
			readChar = fgetwc(m_fp);
		} while (readChar != '\r' && readChar != '\n');
		skipNewLine();
		readChar = fgetwc(m_fp);
	}

	fseek(m_fp, -1L, SEEK_CUR);
	// current position is the top of non-comment line
}

void FileAccessor::getFilepath(char *path)
{
	printf("Please Input File name\n");
	scanf("%s", path);
	return;
}

void FileAccessor::stop()
{
	fclose(m_fp);
}

/* Open FILE, get size of the pattern, then move to file pointer to the top of valid line */
bool FileAccessor::startReadingPattern(const char *path, int *width, int *height)
{
	int maxWidth = 0;
	int maxHeight = 0;
	int x = 0;
	try {
		fopen_s(&m_fp, path, "r");
		skipComment();
		char readChar;
		while ((readChar = fgetc(m_fp)) != EOF) {
			//putchar(readChar);
			x++;
			if (readChar == '\r' || readChar == '\n') {
				skipNewLine();
				if (maxWidth < x) maxWidth = x;
				x = 0;
				maxHeight++;
			}
		}
		if (maxWidth < x) maxWidth = x;	// in case the last line without \n
		if (x != 0) maxHeight++;// in case the last line without \n
		maxWidth--;
		*width = maxWidth;
		*height = maxHeight;
		m_patternOffsetX = 0;
		m_patternOffsetY = 0;
		
		fseek(m_fp, 0L, SEEK_SET);
		skipComment();
		// current position is the top of valid line
		return true;
	} catch (...) {
		printf("Error whiel reading %s\n", path);
		return false;
	}
}

bool FileAccessor::readPattern(int *offsetX, int *offsetY, int *prm)
{
	char readChar;
	try {
		if ((readChar = fgetc(m_fp)) != EOF) {
			if (readChar == '\r' || readChar == '\n') {
				skipNewLine();
				m_patternOffsetY++;
				m_patternOffsetX = 0;
				if ((readChar = fgetwc(m_fp)) == EOF) return false;
			}
			*prm = (readChar == 'x') || (readChar == 'X') || (readChar == '*') || (readChar == 'O');
			*offsetX = m_patternOffsetX;
			*offsetY = m_patternOffsetY;
			m_patternOffsetX++;
			return true;
		}
		return false;
	} catch (...) {
		printf("Error whiel reading\n");
		return false;
	}
}

bool FileAccessor::startWritingPattern(const char *path)
{
	bool ret = false;
	try {
		fopen_s(&m_fp, path, "w");
		ret = true;
	} catch (...) {
		printf("Error while opening %s\n", path);
		ret = false;
	}
	return ret;
}


bool FileAccessor::writePattern(int cell, bool isNewline)
{
	bool ret = false;
	try {
		if (isNewline) {
			fprintf_s(m_fp, "\n");
		}
		if (cell != 0) {
			fprintf_s(m_fp, "X");
		} else {
			fprintf_s(m_fp, ".");
		}
		ret = true;
	} catch (...) {
		printf("Error wrhite writing\n");
		ret = false;
	}
	return ret;
}

