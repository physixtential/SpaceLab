#include <iostream>
#include <unordered_map>
#include <string>
#include <iomanip>
#include <unordered_set>
#include <set>

int
main()
{
    std::set<int> mySet;

    mySet.insert(3);
    mySet.insert(4);
    mySet.insert(5);

    std:: << mySet[0];

    for (const auto& i : mySet) { std::cout << i << '\n'; }
}
