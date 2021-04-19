#include "scratchPad.h"

int showSelectionMenu(double subtotal)
{
	int choice = 0;

	system("cls");
	cout << "Welcome to Crazy Nan's Pie Truck of Paradise" << endl;
	cout << "Enter your crazy selection below" << endl;
	cout << "1. Apple Pie" << endl;
	cout << "2. Cherry Pie" << endl;
	cout << "3. Pumpkin Pie" << endl;
	cout << "4. Pecan Pie" << endl;
	cout << "5. Coke" << endl;
	cout << "6. Diet Coke" << endl;
	cout << "Subtotal: $" << subtotal << endl;
	cout << "Enter Selection (enter 9 to exit): ";

	cin >> choice;

	return choice;
}

int askForQuantity()
{
	int quantity = 0;

	cout << "Enter Quantity: ";
	cin >> quantity;

	return quantity;
}

void processSelection(ofstream& oFile, double& subtotal, int quantity, int choice)
{
	if (choice == 1)
	{
		subtotal = subtotal + (1.99 * quantity);
		cout << "Apple Pie $1.99 each" << endl;
		//oFile << "Apple Pie $1.99 each" << "..." << quantity << "..." << subtotal << endl;
	}

	else if (choice == 2)
	{
		subtotal = subtotal + (2.99 * quantity);
		cout << "Cherry Pie $2.99 each" << endl;
		//oFile << "Cherry Pie $2.99 each" << "..." << quantity << "..." << subtotal << endl;
	}
}


int main()
{
	int choice = 0;
	int quantity = 0;
	double subtotal = 0;

	ofstream oFile;
	oFile.open("receipt.txt");

	while (choice != 9)
	{
		//function 1
		choice = showSelectionMenu(subtotal);

		//function 2
		quantity = askForQuantity();

		//function 3
		processSelection(oFile, subtotal, quantity, choice);

		// else if, else if so on up to 5
	}
}