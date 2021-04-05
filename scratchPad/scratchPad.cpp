#include "scratchPad.h"

#include <iostream>
#include <cmath>
double integral(float a, float b)
{
    double n = 16777215; //number of parts
    float Dx = (b - a) / n;
    float i = 1;
    double sum = 0;

    float x;
    float f;
    while (i <= n)
    {
        x = a + Dx * i;
        f = cos(x); //define expression here
        //printf("%f",f);
        sum = sum + Dx * f;
        if (i == n)
        {
            //printf("Area = %f\n", sum);
            return sum;
        }

        i = i + 1;
    }

}

int main()
{
    printf("%f", integral(1, 3));
    return 0;
}