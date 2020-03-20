#pragma once


class Person
{
public:
	int m_age;
	int m_dayInfected;
	bool m_isOnset;
	bool m_isContagious;
	bool m_isImmunized;
	double m_x;	// 0 - 1.0
	double m_y;	// 0 - 1.0
	double m_speedX;	// -1.0 - 1.0
	double m_speedY;	// -1.0 - 1.0

public:
	Person(double x0 = 0.0, double x1 = 1.0, double y0 = 0.0, double y1 = 1.0);
	~Person();
	void updatePos();
	void updateSymptom(std::vector<Person*> &pList);
	void getPosition(int *x, int *y, const int width, const int height);

private:
	double calcDistance(Person *p);
};

