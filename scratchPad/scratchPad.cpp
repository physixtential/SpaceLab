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

double m = 0;

int main()
{
	double* dist = new double[10];

	for (size_t i = 1; i < 5; i++)
	{
		for (size_t j = 0; j < i; j++)
		{
			int e = i * (i - 1) * .5+j;
			dist[e] = 1.34;
			printf("%d\t%d\t%d\t%lf\n", i, j, e, dist[e]);
		}
	}

	for (size_t i = 0; i < 10; i++)
	{
		printf("%lf\n", dist[i]);
	}

	return 0;
}
