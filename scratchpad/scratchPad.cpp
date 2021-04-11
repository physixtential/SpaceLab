#include "scratchPad.h"

class Triangle
{
public:
	Triangle();
	~Triangle();

private:

};

Triangle::Triangle()
{
}

Triangle::~Triangle()
{
}

float getWeightA()
{
    return ((marker.y - c.y) * (b.x - c.x) + (marker.x - c.x)
        * (c.y - b.y) / ((a.y - c.y) * (b.x - c.x) + (a.x - c.x)
            * (c.y - b.y));
}

void calcMix()
{
    auto mixA = getWeightA();

    auto mixB = (marker.x - mixA * a.x - c.x + mixA *
        c.x) / (b.x - c.x);

    auto mixC = 1.f - mixA - mixB;
}

int main()
{

}