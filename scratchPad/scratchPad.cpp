#include <iostream>
#include <string>
using namespace std;

int check(string str, string a)
{
	if (str.find(a))
	{
		return NULL;
	}
}

int main()
{
	int output;
	string str;
	string a;
	cout << "Enter the first string" << endl;
	getline(cin, str);
	cout << "Enter what character you're looking for" << endl;
	getline(cin, a);



	system("pause");
	return 0;
}