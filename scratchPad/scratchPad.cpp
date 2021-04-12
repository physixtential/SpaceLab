#include "scratchPad.h"


template<typename sType, typename vecType>
struct Triangle
{
	// Three points that describe the triangle:
	vecType a, b, c;

	// Current point in triangle:
	vecType marker;

	// Weight at each triangle corner:
	sType mixA;
	sType mixB;
	sType mixC;

	Triangle(const vecType& a, const vecType& b, const vecType& c, const vecType& marker)
	{
		this->a = a;
		this->b = b;
		this->c = c;
		this->marker = marker;

		calcWeights();
	}

	void calcWeights()
	{
		mixA = (marker.y - c.y) * (b.x - c.x) + (marker.x - c.x)
			* (c.y - b.y) / ((a.y - c.y) * (b.x - c.x) + (a.x - c.x)
				* (c.y - b.y));

		mixB = (marker.x - mixA * a.x - c.x + mixA *
			c.x) / (b.x - c.x);

		mixC = 1.f - mixA - mixB;
	}
};


template<typename sType, typename vecType>
struct Rectangle
{
	// Three points that describe the triangle:
	vecType a, b, c, d;

	// Current point in rectangle:
	vecType marker;

	// Weight at each rectangle corner:
	sType mixA;
	sType mixB;
	sType mixC;
	sType mixD;

	Rectangle(const vecType& a, const vecType& b, const vecType& c, const vecType& d, const vecType& marker)
	{
		this->a = a;
		this->b = b;
		this->c = c;
		this->d = d;
		this->marker = marker;

		calcWeights();
	}

	void calcWeights()
	{
		//mixA = (marker.y - c.y) * (b.x - c.x) + (marker.x - c.x)
		//	* (c.y - b.y) / ((a.y - c.y) * (b.x - c.x) + (a.x - c.x)
		//		* (c.y - b.y));

		//mixB = (marker.x - mixA * a.x - c.x + mixA *
		//	c.x) / (b.x - c.x);

		//mixC = 1.f - mixA - mixB;
	}
};


int main()
{
	vector3d a = { 1, 1, 0 };
	vector3d b = { 4, 1, 0 };
	vector3d c = { 3, 2, 0 };
	vector3d marker = { 2, 2, 0 };

	Triangle<double, vector3d> test(a, b, c, marker);

	std::cout << test.mixA << ',' << test.mixB << ',' << test.mixC;
}