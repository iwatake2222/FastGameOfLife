#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "Values.h"
#include "Person.h"


Person::Person(double x0, double x1, double y0, double y1)
{
	const int POS_DIV = RAND_MAX / 2;
	m_x = (rand() % POS_DIV) / (double)POS_DIV;
	m_y = (rand() % POS_DIV) / (double)POS_DIV;
	m_x = x0 + (x1 - x0) * m_x;
	m_y = y0 + (y1 - y0) * m_y;
	m_speedX =  1.0 - 2 * (rand() % POS_DIV) / (double)POS_DIV;
	m_speedY =  1.0 - 2 * (rand() % POS_DIV) / (double)POS_DIV;

	m_speedX /= 1000.0;
	m_speedY /= 1000.0;

	m_age = 20;
	m_dayInfected = 0;
	m_isOnset = false;
	m_isContagious = false;
	m_isImmunized = false;


	if (rand() % 10 == 0) m_dayInfected = 1;
}

Person::~Person()
{

}

void Person::updatePos()
{
	m_x += m_speedX;
	m_y += m_speedY;
	if (m_x <= 0) { m_x = 0; m_speedX *= -1; }
	if (m_x >= 1.0) { m_x = 1.0; m_speedX *= -1; }
	if (m_y <= 0) { m_y = 0; m_speedY *= -1; }
	if (m_y >= 1.0) { m_y = 1.0; m_speedY *= -1; }
}

void Person::getPosition(int *x, int *y, const int width, const int height)
{
	*x = (int)(m_x * (width - 1));
	*y = (int)(m_y * (height - 1));
}

double Person::calcDistance(Person *p)
{
	double dx = m_x - p->m_x;
	double dy = m_y - p->m_y;
	return sqrt(dx * dx + dy * dy);
}

#define PROB_INFECTED   1000	// prob per day per person
#define PROB_CONTAGIUS  1000	// prob per day
#define DAY_CONTAGIUS     14
#define PROB_IMMUNIZED  1000	// prob per day
#define DAY_IMMUNIZED     30

void Person::updateSymptom(std::vector<Person*> &pList)
{
	if (m_isImmunized) return;

	if (m_dayInfected > 0) m_dayInfected++;
	if (m_dayInfected > DAY_IMMUNIZED && rand() % PROB_IMMUNIZED == 0) {
		m_dayInfected = 0;
		m_isOnset = false;
		m_isContagious = false;
		m_isImmunized = true;
		return;
	}

	if (m_dayInfected == 0) {
		for (int i = 0; i < pList.size(); i++) {
			if (pList[i]->m_isContagious) {
				double d = calcDistance(pList[i]);
				if (d < 0.05 &&  rand() % PROB_INFECTED == 0) m_dayInfected = 1;
			}
		}
	} else if (!m_isContagious) {
		if (m_dayInfected > DAY_CONTAGIUS && rand() % PROB_CONTAGIUS == 0) {
			m_isContagious = true;
		}
	}
	
	//if (m_dayInfected == 14) {
	//	if (rand() % 5 == 0) m_isOnset = true;
	//}
}

