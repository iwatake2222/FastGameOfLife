#pragma once
class IWorldView
{
public:
	virtual ~IWorldView() {}
	virtual void onUpdate(void) = 0;
	virtual void onResize(int w, int h) = 0;
	virtual void onClick(int button, int state, int x, int y) = 0;
	virtual void onDrag(int x, int y) = 0;
	virtual void onWheel(int wheel_number, int direction, int x, int y) = 0;
	virtual void onKeyboard(unsigned char key, int x, int y) = 0;
};

