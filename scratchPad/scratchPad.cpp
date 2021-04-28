#include "scratchPad.h"

#include <ppl.h>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <unordered_map>
#include <windows.h>



int main()
{
	std::vector<int> a(1'000'000);
	std::vector<int> b(1'000'000);
	int total = 0;

	for (size_t i = 0; i < a.size(); i++)
	{
		a[i] = i ;
	}

	for (size_t i = 0; i < a.size(); i++)
	{
		total += a[i];
	}

	std::cout << total << '\n';

	concurrency::parallel_transform(a.begin(), a.end(), b.begin(), calcThing);

	total = 0;
	for (size_t i = 0; i < b.size(); i++)
	{
		total += b[i];
	}

	std::cout << total << '\n';
}