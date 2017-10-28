#pragma once
class FileAccessor
{
private:
	static FILE *m_fp;
	static int m_patternOffsetX, m_patternOffsetY;
	static int m_lastReadChar;
	static int m_currentX, m_currentY;

public:
	FileAccessor();
	~FileAccessor();
	static void skipComment();
	static void skipNewLine();
	static bool getFilepath(WCHAR *path, WCHAR *filter);
	static void stop();
	static bool startReadingPattern(WCHAR *path, int *width, int *height);
	static bool readPattern(int *x, int *y, int *prm);
	static bool startWritingPattern(WCHAR *path);
	static bool writePattern(int cell, bool isNewline);

};

