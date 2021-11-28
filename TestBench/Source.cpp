#include <iostream>
#include "../Line_sphere_intersection.h"


int main()
{
	// Make line:
	vector3d origin = { 3,.9,0 };
	vector3d direction = { -1,0,0 };
	// Make sphere:
	vector3d center = { 0,0,0 };
	double radius = 1;

	line_sphere_intersect(origin, direction, center, radius);

	return 0;
}