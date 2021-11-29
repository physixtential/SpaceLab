#pragma once

#include <iostream>
#include <cmath>
#include "vector3d.hpp"
#include "linalg.hpp"

using namespace linalg;
using linalg::aliases::double3;
using linalg::aliases::double3x3;

// Generate a vector orthogonal to the given
double3 arbitrary_orthogonal(const double3& vec)
{
	bool b0 = (vec.x < vec.y) && (vec.x < vec.z);
	bool b1 = (vec.y <= vec.x) && (vec.y < vec.z);
	bool b2 = (vec.z <= vec.x) && (vec.z <= vec.y);

	return linalg::cross(vec, double3(int(b0), int(b1), int(b2)));
}

// Make a basis coordinate system given a looking direction
double3x3 make_basis(double3 look) {
	double3 up = arbitrary_orthogonal(look);
	double3 side = linalg::cross(up, look);

	look = normalize(look);
	up = normalize(up);
	side = normalize(side);

	return double3x3(look, up, side);
}


// Takes a position as origin and velocity (looking) as x axis to define a coordinate system in which y and z allow you to pan in that system and then return the transformed coordinates in the original coordinate system.
double3 perpendicular_move(const double3& position, const double3& looking, const double y, const double z) {
	const auto basis33 = make_basis(looking);
	const auto basis33_transpose = transpose(basis33);
	return position + mul(basis33_transpose, double3(0, y, z));
}

//Convert from vector3d to double3
double3 to_double3(const vector3d& vec) {
	return { vec.x,vec.y,vec.z };
}

//Convert from double3 to vector3d
vector3d to_vector3d(const double3& vec) {
	return { vec.x,vec.y,vec.z };
}

double3 perpendicular_move(const vector3d& position, const vector3d& looking, const double y, const double z) {

	const auto basis33 = make_basis(to_double3(looking));
	const auto basis33_transpose = transpose(basis33);
	return to_double3(position) + mul(basis33_transpose, double3(0, y, z));
}

void print_m33(double3x3& m33) {
	for (const auto& row : m33) {
		for (const auto& el : row) {
			std::cout << '\t' << el;
		}
		std::cout << "\n";
	}
}

std::string vec_string(const double3 vec) {
	return '(' + std::to_string(vec.x) + ',' + std::to_string(vec.y) + ',' + std::to_string(vec.z) + ')';
}

// Rounding
inline std::string rounder(double value, int digits)
{
	return std::to_string(value).substr(0, digits);
}

// Scientific Notation
inline std::string scientific(double value)
{
	std::stringstream ss;
	ss << std::setprecision(0) << std::scientific << value;
	return ss.str();
}

// Output a nice title bar in terminal:
inline void titleBar(std::string title)
{
	std::cerr << '\n';
	for (size_t i = 0; i < ((62 - title.size()) / 2); i++)
	{
		std::cerr << '=';
	}
	std::cerr << ' ' << title << ' ';
	for (size_t i = 0; i < ((62 - title.size()) / 2); i++)
	{
		std::cerr << '=';
	}
	std::cerr << "\n\n";
}

// // Print anything:
// template <typename theType>
// void print(theType value)
// {
// 	std::cerr << value;
// }

// Ask a yes or no question:
inline bool input(const std::string& question)
{
	char answer;
	std::cerr << question;
	std::cin >> answer;
	if (answer == 'y')
	{
		return true;
	}
	else
	{
		return false;
	}
}

// Generate a random double from -.5lim to .5lim so that numbers are distributed evenly around 0:
inline double rand_double(const double lim)
{
	return lim * (static_cast<double>(rand()) / static_cast<double>(RAND_MAX) - .5);
}

inline double rand_between(const double min, const double max)
{
	double f = (double)rand() / RAND_MAX;
	return min + f * (max - min);
}

// Returns a vector within the desired radius, resulting in spherical random distribution
inline vector3d rand_spherical_vec(double radius)
{
	vector3d vec = { rand_double(radius), rand_double(radius), rand_double(radius) };
	const double halfLim = radius;
	while (vec.norm() > halfLim)
	{
		vec = { rand_double(radius), rand_double(radius), rand_double(radius) };
	}
	return vec;
}

// Returns a vector within the desired radius, resulting in spherical random distribution
inline vector3d rand_shell_vec(double outer_radius, double inner_radius)
{
	vector3d vec = { rand_double(outer_radius), rand_double(outer_radius), rand_double(outer_radius) };
	const double halfLim = outer_radius * .5;
	if (halfLim < inner_radius)
	{
		std::cerr << "Inner radius is larger than boundary. Impossible.\n";
		exit(EXIT_FAILURE);
	}
	while (vec.norm() > halfLim || vec.norm() < inner_radius)
	{
		vec = { rand_double(outer_radius), rand_double(outer_radius), rand_double(outer_radius) };
	}
	return vec;
}
