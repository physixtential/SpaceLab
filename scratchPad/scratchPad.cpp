#include "scratchPad.h"
#include <iostream>
#include <vector>

using namespace std;

class movie
{
private:
    string title;
    int year;
    float rating;

public:
    movie()
    {
        title = "Forest Gump";
        year = 1994;
        rating = 4.4;
    }
    movie(string i, int j, float k)
    {
        title = i;
        year = j;
        rating = k;
    }

    void setTitle(string i)
    {
        title = i;
    }

    void setYear(int j)
    {
        year = j;
    }

    void setRating(float k)
    {
        rating = k;
    }

    string getTitle()
    {
        return title;
    }

    int getYear()
    {
        return year;
    }

    float getRating()
    {
        return rating;
    }




};

int main()
{
    string title, search;
    int year, choice;
    float rating;
    movie m;
    vector<movie>list(10);
    while (choice != 4)
    {
        cout << "1 to add a movie" << endl << "2 to search a movie" << endl << "3 to display all movies." << endl << "4 to quit." << endl;
        cin >> choice;

        if (choice == 1)
        {
            cout << "Enter the name of the movie" << endl;
            cin.ignore();
            getline(cin, title);

            cout << "Enter the year of the movie's release" << endl;
            cin >> year;

            cout << "Enter the rating of the movie on a scale of 1-5" << endl;
            cin >> rating;

            list.push_back(movie(title, year, rating));
        }
        /* if(choice==2)
        {
          cout << "Please enter the name of the movie" << endl;
          cin.ignore();
          getline(cin, search);
          if(find(list.begin(), list.end(), search) !=list.end())
          {   */
        if (choice == 3)
        {

            cout << m.getTitle() << endl;
            cout << m.getYear() << endl;
            cout << m.getRating() << endl;
        }



    }
    system("pause");
    return 0;
}