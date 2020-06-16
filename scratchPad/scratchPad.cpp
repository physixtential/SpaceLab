#include <cmath>
#include <stdio.h>

size_t numBalls = 1 << 10;
double* balls;
// Easy motion component reference in array structure:
constexpr unsigned int x = 0;
constexpr unsigned int y = 1;
constexpr unsigned int z = 2;
constexpr unsigned int vx = 3;
constexpr unsigned int vy = 4;
constexpr unsigned int vz = 5;
constexpr unsigned int ax = 6;
constexpr unsigned int ay = 7;
constexpr unsigned int az = 8;
constexpr unsigned int wx = 9;
constexpr unsigned int wy = 10;
constexpr unsigned int wz = 11;
constexpr unsigned int R = 12;
constexpr unsigned int m = 13;
constexpr unsigned int moi = 14;
// Therefore:
constexpr unsigned int numProps = 15;

int main()
{
	balls = new double[numBalls * numProps];

	for (size_t i = 0; i < numBalls; i++)
	{
		for (size_t j = 0; j < numProps; j++)
		{
			balls[numProps * i + j] = i + j;
		}
	}

	size_t ball = 3;

	double test = balls[numProps * ball + y];

	printf("%lf", test);

	delete[] balls;
	return 0;
}

