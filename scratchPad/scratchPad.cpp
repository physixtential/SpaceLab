#include "scratchPad.h"

#include<iostream>
#include<math.h>
using namespace std;

void inp(int& a, int& b)
{
	do
	{
		cout << " Donner deux entiers a et b tel que a<b";
		cin >> a >> b;
	} while (a > b);
}

void n_av(int& a, int& b, int& n, int& av)
{

	n = (b - a) - 1;
	av = a + b / 2;

}
void max_min(int& a, int& b, int& n, float& max, float& min)
{
	int* A = new int[n];
	int k, counter = 0;

	cout << "donner les valeurs" << n << "qui sont entre a et b";
	cin >> k;

	for (int i = 0; i < n; i++)
	{
		if (k >= a && k <= b)
		{
			A[i] = k;
		}
		else
		{
			cout << "Value not in range " << a << " to " << b << '\n';
			i--;
		}
	}

	delete[] A; // You have to clean up c style arrays like this.
}
void add(int a, int n)
{

}
void main()
{
	int n;
	float a, b, max, min;
	

}