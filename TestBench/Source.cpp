#include <iostream>
#include "../Line_sphere_intersection.hpp"
#include "../vector3d.hpp"
#include "../linalg.hpp"
#include "../Utils.hpp"

using namespace linalg;
using linalg::aliases::double3;





int main()
{
	double3 pos = { 1,1,1 };
	double3 vel = double3(-1, -1, -1);

	// Make a random displacement in y and z, convert to world coords, add to position vector.
	std::vector<double3> test_positions;
	test_positions.emplace_back(perpendicular_move(pos, vel, 0, 0));
	test_positions.emplace_back(perpendicular_move(pos, vel, 1, 0));
	test_positions.emplace_back(perpendicular_move(pos, vel, -1, 0));
	test_positions.emplace_back(perpendicular_move(pos, vel, 0, 1));
	test_positions.emplace_back(perpendicular_move(pos, vel, 1, 1));
	test_positions.emplace_back(perpendicular_move(pos, vel, -1, 1));
	test_positions.emplace_back(perpendicular_move(pos, vel, 0, -1));
	test_positions.emplace_back(perpendicular_move(pos, vel, 1, -1));
	test_positions.emplace_back(perpendicular_move(pos, vel, -1, -1));

	for (auto& vec : test_positions) {
		std::cout << vec_string(vec) << '\n';
	}

}