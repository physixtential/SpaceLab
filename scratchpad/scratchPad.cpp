#include "scratchPad.h"

int add_three(int x, int y)
{
	return x + y;
}

int main()
{
	std::vector<int> a = { 1, 2, 3, 4, 5 };
	std::vector<int> b = { 5, 4, 3, 2, 1 };
	std::vector<int> c(5);

	std::transform(a.begin(), a.end(), b.begin(), c.begin(), add_three);

	for (size_t i = 0; i < c.size(); i++)
	{
		std::cout << a[i] << " + " << b[i] << " = " << c[i] << '\n';

	}
}