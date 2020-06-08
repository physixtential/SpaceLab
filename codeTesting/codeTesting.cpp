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
	double* distances; // Placeholder for new array after size determined.
};

int main()
{
	ball* balls;
	balls = new ball[4];
	balls[3].m = 13;
    std::cout << sizeof(ball);
}
