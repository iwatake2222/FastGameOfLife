#pragma once
class FileAccessor
{
private:
	static FILE *m_fp;
	static int m_patternOffsetX, m_patternOffsetY;
	static int m_lastReadChar;
public:
	FileAccessor();
	~FileAccessor();
	static bool getFilepath(WCHAR *path, WCHAR *filter);
	static bool startReadingWorld(WCHAR *path, int *width, int *height);
	static void stop();
	static bool readPosition(int *x, int *y, int *prm);
	static bool writePosition(int x, int y, int prm);
	static bool startWritingWorld(WCHAR *path, int width, int height);



	static bool startReadingPattern(WCHAR *path);
	static bool readPattern(int *offsetX, int *offsetY, int *prm);
	static bool startWritingPattern(WCHAR *path);
	static bool writePattern(int cell, bool isNewline);
};

