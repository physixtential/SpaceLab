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
#include "../vector3d.h"

void makeVec(vector3d* vec)
{
	// Init pos
	for (size_t i = 0; i < balls; i++)
	{
		vec[i] = { (double)i,(double)i,(double)i };
	}
}
//#include "../initializations.h"
//#include "../objects.h"

int balls = 5;
int pairs = balls * (balls - 1) / 2;

int main()
{
	double* dist = new double[pairs]();
	vector3d* pos = new vector3d[balls]();
	double sum = 0;

	makeVec(pos);

	double time0 = 0, time1 = 0;

	//////////////////////////////////////////////
	time0 = omp_get_wtime();
	for (size_t i = 1; i < balls; i++)
	{
		for (size_t j = 0; j < i; j++)
		{
			int e = i * (i - 1) * .5; // Get to the right first ball region in array
			dist[e + j] = (pos[i] - pos[j]).norm(); // add j to get to the other ball in the pair.
			//printf("%d\t%d\t%d\t%lf\n", i, j, e, dist[e]);
		}
	}
	time1 = omp_get_wtime();

	for (size_t i = 0; i < pairs; i++)
	{
		sum += dist[i];
	}
	
	printf("%lf\n", sum);

	delete[] dist;
	delete[] pos;

	std::vector<std::vector<double>> test;



	////////////////////////////////////////////////
	time0 = omp_get_wtime();
	for (size_t i = 0; i < balls; i++)
	{
		for (size_t j = i+1; j < balls; j++)
		{
			//int e = i * (i - 1) * .5; // Get to the right first ball region in array
			distBig[i][j] = (pos[i] - pos[j]).norm(); // add j to get to the other ball in the pair.
			//printf("%d\t%d\t%d\t%lf\n", i, j, e, dist[e]);
		}
	}
	time1 = omp_get_wtime();

	return 0;
}
