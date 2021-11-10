#include <iostream>

//Prototypes
int getNum();
void changeArrayValues(int numbers[10]);
int minArrayValue(int numbers[10]);
void printArray(int numbers[10]);

int main()
{
	int numbers[10] = { 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 }; //Array with 10 elements
	//int i = 0;
	int index = minArrayValue(numbers);
	printArray(numbers);
	//printf("\nThe Lowest Value is %d at index %d ", index, index);

	//changeArrayValues(numbers);


	return 0;
}

void printArray(int numbers[10])
{
	for (size_t i = 0; i < 10; i++)
	{
		std::cout << numbers[i] << ' ';
	}
}

/*
Function: changeArrayValues
Description: Takes the array and the number of elements in the arrayas parameters
Parameters: int numbers[10]
Returns: Nothing
*/
void changeArrayValues(int numbers[10])
{
	int inputNum;
	int i = 0;
	while (i < 10)
	{
		printf("\nPlease Enter a Number at Index %d:", i);
		inputNum = getNum();
		i++;
	}
}

/*
Function: minArrayValue
Description: Find the minimum value in the array
Parameters: int numbers[10]
Returns:
*/
int minArrayValue(int numbers[10])
{
	int min = 1e10; //minimum value
	int i = 0;
	int index = 0; //tracks what index min is at
	while (i < 10)
	{

		if (numbers[i] < min) //Goes through each input and checks if it is the minimum
		{
			min = numbers[i];
			index = i;
		}
		i++;
	}
	return index;
}

/*
Function: getNum
Description: Gets a number from the user, and returns it
Parameters: void
Returns: number
*/
#pragma warning(disable: 4996)
int getNum()
{
	char record[121] = { 0 }; /* record stores the string */
	int number = 0;
	/* use fgets() to get a string from the keyboard */
	fgets(record, 121, stdin);
	/* extract the number from the string; sscanf() returns a number
	* corresponding with the number of items it found in the string */
	if (sscanf(record, "%d", &number) != 1)
	{
		/* if the user did not enter a number recognizable by
		* the system, set number to -1 */
		number = -1;
	}
	return number;
}