#include "scratchPad.h"

int main()
{
	for (size_t i = 0; i < 1000; i++)
	{
		double fi = i;
		int n = i * i / 2 - i / 2;
		if (n != fi * fi / 2. - fi / 2.)
		{
			cout << "wrong";
		}
	}
}