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

void makeVec(vector3d* vec, int len)
{
	// Init pos
	for (size_t i = 0; i < len; i++)
	{
		vec[i] = { (double)i,(double)i,(double)i };
	}
}
//#include "../initializations.h"
//#include "../objects.h"

const int balls = 5000;
int pairs = balls * (balls - 1) / 2;

double distBig[balls][balls];

int main()
{
	for (size_t i = 0; i < 10; i++)
	{
		double* dist = new double[pairs]();
		vector3d* pos = new vector3d[balls]();
		double sum = 0;

		makeVec(pos, balls);

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

		printf("Time:\t\t%lf\n", time1 - time0);

		//for (size_t i = 0; i < pairs; i++)
		//{
		//	printf("%.3e\n", dist[i]);
		//}

		for (size_t i = 0; i < pairs; i++)
		{
			sum += dist[i];
		}

		printf("First total: %lf\n", sum);

		delete[] dist;

		////////////////////////////////////////////////
		time0 = omp_get_wtime();
		for (size_t i = 0; i < balls; i++)
		{
			for (size_t j = i + 1; j < balls; j++)
			{
				//int e = i * (i - 1) * .5; // Get to the right first ball region in array
				distBig[i][j] = (pos[i] - pos[j]).norm(); // add j to get to the other ball in the pair.
				//printf("%d\t%d\t%d\t%lf\n", i, j, e, dist[e]);
			}
		}
		time1 = omp_get_wtime();

		printf("Time:\t\t%lf\n", time1 - time0);
		//for (size_t i = 0; i < balls; i++)
		//{
		//	printf("%.3e\t%.3e\t%.3e\t%.3e\t%.3e\n", distBig[i][0], distBig[i][1], distBig[i][2], distBig[i][3], distBig[i][4]);
		//}
		sum = 0;
		for (size_t i = 0; i < balls; i++)
		{
			for (size_t j = 0; j < balls; j++)
			{
				sum += distBig[i][j];
			}
		}

		printf("Second Total: %lf\n", sum);

		delete[] pos;
	}
	return 0;
}
