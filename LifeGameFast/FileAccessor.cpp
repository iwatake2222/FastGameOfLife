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

FileAccessor::FileAccessor()
{
}


FileAccessor::~FileAccessor()
{
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

bool FileAccessor::startReadingWorld(WCHAR *path, int *width, int *height)
{
	bool ret = false;
	try {
		_wfopen_s(&m_fp, path, L"r");
		fwscanf_s(m_fp, L"%d,%d", width, height);
		ret = true;
	} catch (...) {
		printf("Error whiel reading %s\n", path);
		ret = false;
	}
	return ret;
}

bool FileAccessor::readPosition(int *x, int *y, int *prm)
{
	bool ret = false;
	try {
		ret = (fwscanf_s(m_fp, L"%d,%d,%d", x, y, prm) != EOF);
	} catch (...) {
		printf("Error whiel reading\n");
		ret = false;
	}
	return ret;
}

bool FileAccessor::startWritingWorld(WCHAR *path, int width, int height)
{
	bool ret = false;
	try {
		_wfopen_s(&m_fp, path, L"w");
		fwprintf_s(m_fp, L"%d,%d\n", width, height);
		ret = true;
	} catch (...) {
		printf("Error whiel writing %s\n", path);
		ret = false;
	}
	return ret;
}

bool FileAccessor::writePosition(int x, int y, int prm)
{
	bool ret = false;
	try {
		fwprintf_s(m_fp, L"%d,%d,%d\n", x, y, prm);
		ret = true;
	} catch (...) {
		printf("Error whiel reading\n");
		ret = false;
	}
	return ret;
}

bool FileAccessor::startReadingPattern(WCHAR *path)
{
	bool ret = false;
	try {
		_wfopen_s(&m_fp, path, L"r");
		ret = true;
	} catch (...) {
		printf("Error whiel reading %s\n", path);
		ret = false;
	}
	m_patternOffsetX = 0;
	m_patternOffsetY = 0;
	return ret;
}

bool FileAccessor::readPattern(int *offsetX, int *offsetY, int *prm)
{
	bool ret = false;
	wint_t readChar;
	try {
		setlocale(LC_ALL, "Japanese");
		readChar = fgetwc(m_fp);
		while (readChar == L'#') {
			do {
				readChar = fgetwc(m_fp);
			} while (readChar != L'\r' && readChar != L'\n');
			readChar = fgetwc(m_fp);
		}
		while (readChar == L'\r' || readChar == L'\n') {
			m_patternOffsetY++;
			m_patternOffsetX = 0;
			printf("\n");
			wint_t ch = fgetwc(m_fp);
			if (ch == L'\r' || ch == L'\n') {	// for \r\n
				readChar = fgetwc(m_fp);
				continue;
			} else {
				readChar = ch;
			}
		}
		if (readChar == WEOF) return false;
		*prm = (readChar == L'x') || (readChar == L'X') || (readChar == L'*') || (readChar == L'■') || (readChar == L'O');
		*offsetX = m_patternOffsetX;
		*offsetY = m_patternOffsetY;
		wprintf(L"(%d, %d) = %c(%d) ", m_patternOffsetX, m_patternOffsetY, readChar, (*prm != 0));
		m_patternOffsetX++;
		ret = true;
	} catch (...) {
		printf("Error whiel reading\n");
		ret = false;
	}
	return ret;
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

