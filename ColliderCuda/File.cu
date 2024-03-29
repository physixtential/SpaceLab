﻿#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <sstream>

#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/detail/config.h>
#include <thrust/zip_function.h>

//#include "../vector3d.hpp"
//#include "../initializations.hpp"
//#include "../ballGroup.hpp"

struct add_multiply
{
    __host__ __device__
        void operator()(const float& a, const float& b, const float& c, float& d)
    {
        // D[i] = A[i] + B[i] * C[i];
        d = a + b * c;
    }
};

__host__ __device__ void fadd_multiply(const float& a, const float& b, const float& c, float& d)
{
    d = a + b * c;
}

int main(void)
{
    // allocate storage
    thrust::device_vector<float> A(5);
    thrust::device_vector<float> B(5);
    thrust::device_vector<float> C(5);
    thrust::device_vector<float> D(5);

    // initialize input vectors
    A[0] = 3;  B[0] = 6;  C[0] = 2;
    A[1] = 4;  B[1] = 7;  C[1] = 5;
    A[2] = 0;  B[2] = 2;  C[2] = 7;
    A[3] = 8;  B[3] = 1;  C[3] = 4;
    A[4] = 2;  B[4] = 8;  C[4] = 3;


    // apply the transformation using zip_function
    thrust::for_each
    (
        thrust::make_zip_iterator(thrust::make_tuple(A.begin(), B.begin(), C.begin(), D.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(A.end(), B.end(), C.end(), D.end())),
        thrust::make_zip_function(add_multiply())
    );

    // print the output
    std::cout << "N-ary functor" << std::endl;
    for (int i = 0; i < 5; i++)
        std::cout << A[i] << " + " << B[i] << " * " << C[i] << " = " << D[i] << std::endl;
}