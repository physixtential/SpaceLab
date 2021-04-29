#include "scratchPad.h"

#include <ppl.h>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <unordered_map>
#include <windows.h>

void unique_rand_numbers(int& x, int& y, int& z)
{
	srand(time(0));
	while (x == y or x == z or z == y)
	{
		y = 1 + rand() % 10;
		z = 1 + rand() % 10;
	}
}

int main()
{
	int a = 1;
	int b = 1;
	int c = 3;

	unique_rand_numbers(a, b, c);

	cout << a << '\t' << b << '\t' << c;
}