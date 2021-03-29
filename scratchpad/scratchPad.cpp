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
	int numMatrices = 4;

	std::vector<std::vector<std::vector<int>>> matList;

	for (size_t matSize = 3; matSize < numMatrices + 3; matSize++)
	{
		std::vector<std::vector<int>> oneMatrix(matSize);

		// Set the size of the internal vectors:
		for (size_t i = 0; i < matSize; i++)
		{
			oneMatrix[i].resize(matSize);
		}

		// Populate
		for (size_t i = 0; i < matSize; i++)
		{
			for (size_t j = 0; j < matSize; j++)
			{
				oneMatrix[i][j] = i + j; // Whatever you want here.
			}
		}

		matList.push_back(oneMatrix);
	}

	printBigOlVectorception(matList);
}