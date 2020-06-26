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

	double3 test1 = make_double3(1, 2, 3);
	double3 test2 = make_double3(2, 2, 3);
	double3 test3 = make_double3(1.5, 2, 3);
	double3 test4 = make_double3(0, 0, 0);

	test4 = smoothstep(test1, test3, test2);

	std::cout << test4.x <<'\t'<< test4.y << '\t' << test4.z;
}