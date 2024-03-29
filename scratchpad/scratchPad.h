#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <Windows.h>
#include <array>
#include <chrono>
#include <thread>
#include <limits>
#include <algorithm>
#include "../vector3d.hpp"

void unique_rand_numbers(int& x, int& y, int& z)
{
	srand(time(0));
	while (x == y or x == z or z == y)
	{
		y = 1 + rand() % 10;
		z = 1 + rand() % 10;
	}
}

void matrixMaker(std::vector<std::vector<std::vector<int>>>& matList)
{
	int numMatrices = matList.size();

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

int fun(int n)
{
	int x = 1, k;
	if (n == 1)
	{
		return x;
	}
	for (k = 1; k < n; ++k)
	{
		x += fun(k) * fun(n - k);
	}
	return x;
}




unsigned long long int fib(unsigned long long x)
{
	//recursive
	if ((x == 1) || (x == 0))
	{
		return(x);
	}
	else
	{
		return(fib(x - 1) + fib(x - 2));
	}
}

unsigned long long int fib_i(int num)
{
	//iterative
	unsigned long long int
		x = 0,
		y = 1,
		z = 0;

	for (int i = 0; i < num; i++)
	{
		if (z > 2'147'483'640 or x > 2'147'483'640 or y > 2'147'483'640)
		{
			std::cout << i << ' ' << x << ' ' << y << ' ' << z << ' ';
			break;
		}
		else if (i % 100'000'000 == 0)
		{
			std::cout << i << '\r';
		}

		z = x + y;
		x = y;
		y = z;
	}
	return 0;
}

void zabko()
{
	std::string r_name;
	std::string r_pass;
	std::string l_name;
	std::string l_pass;

	int choice;

	std::string line;

	std::fstream file;
	file.open("test.txt", std::ios::out | std::ios::in);

	std::cout << "1.Login\n2.Register\n";
	std::cout << "What you want?";
	std::cin >> choice;

	if (file.is_open())
	{
		switch (choice)
		{
		case 1:
			//get name, pass
			std::cout << "Name: "; std::cin >> l_name;
			std::cout << "Password: "; std::cin >> l_pass;

			//find if it exist
			std::getline(file, line);
			if (line.find(l_name) != std::string::npos && line.find(l_pass) != std::string::npos)
			{
				std::cout << "Successfully logged in!";
			}
			else
			{
				std::cout << "You're not registered! Please register and try again..";
			}
			break;

		case 2:
			//get name,pass
			std::cout << "Name: ";
			std::cin >> r_name;
			std::cout << "Password: ";
			std::cin >> r_pass;

			//Write it into file
			file << "Username: " << r_name << '\t';
			file << "Password: " << r_pass << '\n';
			file << "------------------" << '\n';
			std::cout << "recorded" << std::endl;
			file.close();
			break;

		default:
			std::cout << "Invalid input..";
			break;
		}
	}
	else
	{
		std::cout << "File not opened.";
	}
}

struct thing
{


	void foo(const std::array<int, 3> arr)
	{
		//stuff
	}
};



//#include <immintrin.h>
//void AVX()
//{
//	float a[8] = { 42, 69, 32, 64, 16, 8, 3, 48 };
//	float b[8] = { 5, 6, 7, 8, 1, 4, 3, 2 };
//	float res[8];
//
//	__m256 ar = _mm256_load_ps(&a[0]);
//	__m256 br = _mm256_load_ps(&b[0]);
//	_mm256_store_ps(&res[0], _mm256_add_ps(ar, br));
//
//	for (int i = 0; i < 8; i++)
//		std::cout << res[i] << ", ";
//}

void usingCArrays(const int n)
{
	unsigned char* arr;
	arr = new unsigned char[n];
	// do stuff
	delete[] arr;
}

void eclipse()
{
	using namespace std::chrono;

	steady_clock::time_point now;


	steady_clock::time_point start = steady_clock::now();
	while (true)
	{
		now = steady_clock::now();
		if (now - start >= milliseconds(1000))
		{
			std::cout << (now - start).count() << '\n';
			start = steady_clock::now();
		}
		else
		{
			std::this_thread::sleep_for(milliseconds(1));
		}
	}
}


using namespace std;
class Solution
{
public:
	void printArrs(int arr1[], int arr2[], int n, int m)
	{
		for (int i = 0; i < n; i++)
		{
			cout << arr1[i] << ' ';
		}
		cout << endl;
		for (int i = 0; i < m; i++)
		{
			cout << arr2[i] << ' ';
		}
		cout << endl;
	}

	// This takes two sorted arrays of increasing values and reorders the values within them such that they increase across both arrays. 
	void sortAcrossTwoArraysMinimalExtraMemory(int arr1[], int arr2[], int n, int m)
	{
		printArrs(arr1, arr2, n, m);

		int r = (n < m) ? n : m;

		for (int i = 0; i < r; i++)
		{
			cout << "Comp: " << n - i - 1 << ',' << i << '\t' << arr1[n - i - 1] << ' ' << arr2[i] << endl;
			if (arr1[n - i - 1] > arr2[i])
			{
				cout << "swapping " << arr2[i] << " with " << arr1[n - i - 1] << '\n';
				swap(arr1[n - i - 1], arr2[i]);
			}
		}
		sort(arr1 + 0, arr1 + n);
		sort(arr2 + 0, arr2 + m);

		printArrs(arr1, arr2, n, m);
	}
};

// How to time benchmark a section of code.
void chronoTimer()
{
	std::chrono::steady_clock::time_point begin;
	std::chrono::steady_clock::time_point end;

	std::chrono::milliseconds best;
	bool firstRun = true;
	for (size_t i = 0; i < 20; i++)
	{
		begin = std::chrono::steady_clock::now();

		// TIMED CODE HERE

		end = std::chrono::steady_clock::now();
		auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

		if (firstRun)
		{
			best = time;
		}
		else
		{
			if (time < best)
			{
				best = time;
			}
		}
	}


	std::cout << best.count() << " ms\n";
}




