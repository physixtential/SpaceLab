#include <vector>
#include <iostream>

void printBigOlVectorception(std::vector<std::vector<std::vector<int>>>& matList)
{
	// Print
	for (size_t i = 0; i < matList.size(); i++)
	{
		std::cout << std::endl;

		for (size_t j = 0; j < matList[i].size(); j++)
		{
			std::cout << std::endl;
			for (size_t k = 0; k < matList[i].size(); k++)
			{
				printf("%i ", matList[i][j][k]);
			}
		}
	}
}



int main()
{

}