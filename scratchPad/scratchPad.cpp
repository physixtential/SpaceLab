#include "scratchPad.h"

#include <iostream>
using namespace std;

class Base
{
public:
    int a;
    Base()
    {
        a = 1;
    }
    Base& operator = (Base& obj)
    {
        a = obj.a;
        return *this;
    }

    virtual int eq() const
    {
        cout << "asd1";
        return 0;
    }
};

class Derived
{
public:
    int b;
    Derived()
    {
        b = 2;
    }

    virtual int eq() override
    {
        cout << "asd2";
        return 1;
    }

};

int main()
{
    Derived ob1;
    Derived ob2;

    cout << ob1.b;

    ob1 = ob2;

    cout << ob1.b;
}