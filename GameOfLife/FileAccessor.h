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
	static void FileAccessor::getFilepath(char *path);
	static void stop();
	static bool startReadingPattern(const char *path, int *width, int *height);
	static bool readPattern(int *x, int *y, int *prm);
	static bool startWritingPattern(const char *path);
	static bool writePattern(int cell, bool isNewline);

};

