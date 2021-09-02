#include <vector>
#include <iostream>

struct Duck
{
	const int feathers = 0;
	Duck(int feathers) :feathers(feathers) {}

	int a = 0;

	void hello()
	{
		a += 1;
	}
};

template<typename T>
void swap(T& a, T& b)
{
	T temp{ a };
	a = b;
	b = temp;
}

int main()
{
	double x{ 3.0 };
	double y{ 2.4 };
	swap(x,y);
}