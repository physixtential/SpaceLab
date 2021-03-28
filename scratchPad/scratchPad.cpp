#include <iostream>
#include <string>
#include <omp.h>
#include <vector>


std::vector<int> two_oldest_ages(std::vector<int> ages)
{
	int oldest = 0, secold = 0;
	for (int i = 0; i < ages.size(); ++i)
	{
		if (oldest < ages[i])
		{
			secold = oldest;
			oldest = ages[i];
		}
	}
	return { secold, oldest };
}

std::vector<int> two_oldest_ages2(std::vector<int> ages)
{
	int oldest = 0, secold = 0;
	for (int i = 0; i < ages.size(); ++i)
	{
		if (oldest < ages[i])
			oldest = ages[i];
	}
	for (int i = 0; i < ages.size(); ++i)
	{
		if (secold < ages[i] && ages[i] != oldest)
			secold = ages[i];
	}
	return { secold, oldest };
}

int main()
{
	std::vector<int> result1;
	std::vector<int> result2;
	double start1;
	double end1;
	double start2;
	double end2;
	double best1 = 100.;
	double best2 = 100.;

	for (size_t i = 0; i < 1000; i++)
	{
		start1 = omp_get_wtime();
		for (size_t i = 0; i < 10000; i++)
		{
			result1 = two_oldest_ages({ 1,5,8,10,3,5,7,23,54,56,1,3,2,67,4,32 });
		}
		end1 = omp_get_wtime();

		start2 = omp_get_wtime();
		for (size_t i = 0; i < 10000; i++)
		{
			result1 = two_oldest_ages2({ 1,5,8,10,3,5,7,23,54,56,1,3,2,67,4,32 });
		}
		end2 = omp_get_wtime();

		if (best1 > end1 - start1)
		{
			best1 = end1 - start1;
		}

		if (best2 > end2 - start2)
		{
			best2 = end2 - start2;
		}
	}
	std::cout << result1[0] << result1[1] << result2[0] << result2[1] << '\n';
	std::cout << best1 << '\n';
	std::cout << best2;
}