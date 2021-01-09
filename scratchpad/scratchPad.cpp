#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <fstream>
#include <time.h>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <omp.h>
//#include "../vector3d.h"
//#include "../objects.h"


struct test
{
	double* x;
};

void copyGroup(test* dst, test* src)
{
	memcpy(dst->x, src->x, sizeof(src->x[0]) * 3);
}

int main()
{
	test thing;
	thing.x = new double[3];
	thing.x[0] = 100;
	thing.x[1] = 200;
	thing.x[2] = 300;

	test otherThing;
	otherThing.x = new double[3];

	copyGroup(&otherThing, &thing);

	test combined;
	combined.x = new double[6];

	memcpy(&combined.x[0], thing.x, sizeof(thing.x[0]) * 3);
	memcpy(&combined.x[3], otherThing.x, sizeof(otherThing.x[0]) * 3);


	std::cout << thing.x[0] << ' ' << thing.x[1] << ' ' << thing.x[2] << std::endl;
	std::cout << otherThing.x[0] << ' ' << otherThing.x[1] << ' ' << otherThing.x[2] << std::endl;
	std::cout << combined.x[0] << ' ' << combined.x[1] << ' ' << combined.x[2] << ' ' << combined.x[3] << ' ' << combined.x[4] << ' ' << combined.x[5] << std::endl;
}
