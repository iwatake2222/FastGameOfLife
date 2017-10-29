#include "stdafx.h"
#include "FileAccessor.h"
#include <windows.h>
#include <Commdlg.h>
#include <tchar.h>
#pragma comment(lib, "Comdlg32.lib")
#pragma comment(lib, "User32.lib")

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
	wint_t readChar;
	do {
		readChar = fgetwc(m_fp);
	} while (readChar == L'\r' || readChar == L'\n');
	if (readChar != WEOF) fseek(m_fp, -1L, SEEK_CUR);
	// current position is the top of new line
}

void FileAccessor::skipComment()
{
	wint_t readChar;
	readChar = fgetwc(m_fp);
	while (readChar == L'#' || readChar == L'!') {
		do {
			readChar = fgetwc(m_fp);
		} while (readChar != L'\r' && readChar != L'\n');
		skipNewLine();
		readChar = fgetwc(m_fp);
	}

	fseek(m_fp, -1L, SEEK_CUR);
	// current position is the top of non-comment line
}

bool FileAccessor::getFilepath(WCHAR *path, WCHAR *filter)
{
	OPENFILENAME ofn;
	WCHAR name[MAX_PATH];
	memset(path, '\0', sizeof(path));
	memset(name, '\0', sizeof(name));

	memset(&ofn, 0, sizeof(OPENFILENAME));
	ofn.lStructSize = sizeof(OPENFILENAME);
	ofn.lpstrFilter = filter;
	ofn.lpstrFile = path;
	ofn.nMaxFile = MAX_PATH;
	ofn.lpstrFileTitle = name;
	ofn.nMaxFileTitle = MAX_PATH;
	return GetOpenFileName(&ofn);
}

void FileAccessor::stop()
{
	fclose(m_fp);
}

/* Open FILE, get size of the pattern, then move to file pointer to the top of valid line */
bool FileAccessor::startReadingPattern(WCHAR *path, int *width, int *height)
{
	int maxWidth = 0;
	int maxHeight = 0;
	int x = 0;
	try {
		_wfopen_s(&m_fp, path, L"r");
		//setlocale(LC_ALL, "Japanese");
		skipComment();
		wint_t readChar;
		while ((readChar = fgetwc(m_fp)) != WEOF) {
			//putchar(readChar);
			x++;
			if (readChar == L'\r' || readChar == L'\n') {
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
	wint_t readChar;
	try {
		if ((readChar = fgetwc(m_fp)) != WEOF) {
			if (readChar == L'\r' || readChar == L'\n') {
				skipNewLine();
				m_patternOffsetY++;
				m_patternOffsetX = 0;
				if ((readChar = fgetwc(m_fp)) == WEOF) return false;
			}
			*prm = (readChar == L'x') || (readChar == L'X') || (readChar == L'*') || (readChar == L'O');
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

bool FileAccessor::startWritingPattern(WCHAR *path)
{
	bool ret = false;
	try {
		_wfopen_s(&m_fp, path, L"w");
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
			fwprintf_s(m_fp, L"\n");
		}
		if (cell != 0) {
			fwprintf_s(m_fp, L"X");
		} else {
			fwprintf_s(m_fp, L".");
		}
		ret = true;
	} catch (...) {
		printf("Error wrhite writing\n");
		ret = false;
	}
	return ret;
}

