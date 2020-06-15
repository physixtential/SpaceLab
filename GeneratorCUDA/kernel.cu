#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include "../vector3d.h"
#include "../initializations.h"
#include "../objects.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Create handy shorthand for error checking each step of CUDA without a bulky conditional every time:
#define CHECK (cudaStatus != cudaSuccess) ? fprintf(stderr, "Error at line %i\n", __LINE__ - 1) : NULL;

cudaError_t intAddWithCuda(int* c, const int* a, const int* b, unsigned int size);

__global__ void addKernel(int* c, const int* a, const int* b)
{
	unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
	c[gid] = a[gid] + b[gid];
}

size_t numBalls = 1 << 24;
size_t blockSize = 64;
size_t numBlocks = numBalls / blockSize;


int main()
{
	int* a = new int[numBalls];
	int* b = new int[numBalls];
	for (size_t i = 0; i < numBalls; i++)
	{
		a[i] = i;
		b[i] = i;
	}
	int* c = new int[numBalls];

	// Add vectors in parallel.
	cudaError_t cudaStatus = intAddWithCuda(c, a, b, numBalls);
	
	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t intAddWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
	int* dev_a = 0;
	int* dev_b = 0;
	int* dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	CHECK;

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	CHECK;
	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	CHECK;
	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	CHECK;

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	CHECK;
	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	CHECK;

	// Launch a kernel on the GPU with one thread for each element.
	addKernel <<<numBlocks, blockSize >>> (dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	CHECK;

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	CHECK;

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	CHECK;

	cudaStatus = cudaDeviceSynchronize();
	CHECK;

	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}
