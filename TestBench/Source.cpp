#include "../Utils.hpp"
#include "../ball_group.hpp"


Ball_group group(1);

int main() {
	for (size_t i = 0; i < 100; i++)
	{
		group = group.add_projectile();
	}

	for (size_t i = 0; i < group.num_particles; i++)
	{
		std::cout << group.pos[i] << '\n';
	}
	//for (size_t i = 0; i < 100; i++)
	//{
	//	//std::cout<<rand_between(-3, 3)<<'\n';
	//	//rand_spherical_vec(1).print();
	//}
}