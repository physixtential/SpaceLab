#include <vector>
#include <iostream>
#include <random>

inline double rand_between(const double min, const double max)
{
	std::uniform_real_distribution<double> test(min, max);
	std::default_random_engine re;
	return test(re);
}

int main()
{
	for (size_t i = 0; i < 10; i++)
	{
		std::cout << rand_between(3, 5);

	}
}