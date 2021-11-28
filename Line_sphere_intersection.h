#pragma once

#include <iostream>
#include <cmath>
#include "vector3d.hpp"

// I got this from https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
bool line_sphere_intersect(const vector3d& origin, const vector3d& direction, const vector3d& center, const double radius) {
	vector3d u_vec = direction.normalized();

	double grad =
		u_vec.dot((origin - center)) *
		u_vec.dot((origin - center)) -
		(
			(origin - center).normsquared() -
			radius * radius
			);

	// If I ever want to distance to the two intersections:
	//double d1 = -u_vec.dot(origin - center) + sqrtf(grad);
	//double d2 = -u_vec.dot(origin - center) - sqrtf(grad);
	//std::cout << '\t' << d1 << '\t' << d2 << '\n';



	if (grad < 0)
	{
		std::cout << "Origin"; origin.print();
		std::cout << "Direction"; direction.print();
		std::cout << "U_vec"; u_vec.print();
		std::cout << "Center"; center.print();
		std::cout << "Radius " << radius << '\n';
		std::cout << "Grad " << grad << '\n';
		return false;
	}
	else
	{
		//std::cout << "intersection exists.";
		return true;
	}
}