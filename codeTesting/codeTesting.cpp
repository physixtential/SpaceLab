#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <fstream>
#include <time.h>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <omp.h>
#include "../vector3d.h"
//#include "../initializations.h"
//#include "../objects.h"

struct ball
{
	vector3d pos, vel, velh, acc, w;
	double m = 0, R = 0, moi = 0, compression = 0;
	double* distances = 0; // Placeholder for new array after size determined.
};

struct cluster
{
	vector3d com, momentum, angularMomentum;
	double m = 0, radius = 0, PE = 0, KE = 0;
	ball* balls;
};

struct universe
{
	vector3d com, momentum, angularMomentum;
	double mTotal = 0, KE = 0, PE = 0, spaceRange = 0;
	ball* balls;
	cluster* clusters = 0;
};

int main()
{
	cluster clusA;
	cluster clusB;
	universe cosmos;

	clusA.balls = new ball[23];
	std::cout << sizeof(clusA.balls);
}
