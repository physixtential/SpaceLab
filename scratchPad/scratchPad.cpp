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

int main()
{
	std::vector<Duck> plump; // Three ducks that will have their feathers specified later.
	plump.reserve(3);
	plump.emplace_back(100);
	plump.emplace_back(200);
	plump.emplace_back(300);

	std::cout << plump.size();
	std::cout << plump[0].feathers;
	std::cout << plump[1].feathers;
	std::cout << plump[2].feathers;
}