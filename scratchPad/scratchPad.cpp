#include "scratchPad.h"

#include <ppl.h>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <unordered_map>
#include <windows.h>

void changeArr(float* arr)
{
	arr[0] = 1;
	arr[1] = 2;
	arr[2] = 3;
}


int main()
{
	float arr[3];
	changeArr(arr);

	for (size_t i = 0; i < 3; i++)
	{
		cout << arr[i] << '\n';
	}
}