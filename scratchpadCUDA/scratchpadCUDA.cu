#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <fstream>
#include <time.h>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include "../vector3d.h"
#include "../cuVectorMath.h"

//#include "../initializations.h"
//#include "../objects.h"

double m = 0;

int main()
{
	double* arr;
	arr = new double[5];

	arr[0] = 1.0;
	arr[1] = 1.1;
	arr[2] = 1.2;
	arr[3] = 1.3;
	arr[4] = 1.4;

	double* a = &arr[2];

	m = *(a + 2);
	m = arr[2 + 2];

	double3 test = make_double3(1, 2, 3);
	test = test + test;
	test *= test;
	test += dot(test, test);

	std::cout << test.x << test.y << test.z;
}