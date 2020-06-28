#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "../cuVectorMath.h"
#include <iostream>

// nvcc does not seem to like variadic macros, so we have to define
// one for each kernel parameter list:
#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif

cudaError_t addWithCuda(double3* pos, const double3* vel, const double3* acc, unsigned int size);

__global__ void addKernel(double3* pos, const double3* vel, const double3* acc)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	pos[i] = vel[i] * acc[i];
}

int main()
{
	const int arraySize = 5;
	double3* pos = new double3[arraySize];
	double3* vel = new double3[arraySize];
	double3* acc = new double3[arraySize];

	for (size_t i = 0; i < arraySize; i++)
	{
		vel[i] = make_double3((double)i, (double)i, (double)i);
		acc[i] = make_double3((double)i, (double)i, (double)i);
	}

	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(pos, vel, acc, arraySize);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	for (size_t i = 0; i < arraySize; i++)
	{
		printf("\nHost data:\n", vel[i].x, vel[i].y, vel[i].z);
		printf("Velocity:\t%lf\t%lf\t%lf\n", vel[i].x, vel[i].y, vel[i].z);
		printf("Accel:   \t%lf\t%lf\t%lf\n", acc[i].x, acc[i].y, acc[i].z);
		printf("Position:\t%lf\t%lf\t%lf\n", pos[i].x, pos[i].y, pos[i].z);
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(double3* pos, const double3* vel, const double3* acc, unsigned int size)
{

	double3* dev_pos;
	double3* dev_vel;
	double3* dev_acc;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output).
	cudaStatus = cudaMalloc((void**)&dev_pos, size * sizeof(double3));
	cudaStatus = cudaMalloc((void**)&dev_vel, size * sizeof(double3));
	cudaStatus = cudaMalloc((void**)&dev_acc, size * sizeof(double3));

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_vel, vel, size * sizeof(double3), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(dev_acc, acc, size * sizeof(double3), cudaMemcpyHostToDevice);

	// Launch a kernel on the GPU with one thread for each element.

	dim3 numBlocks(1, 1, 1);
	dim3 tpb(size, 1, 1);

	addKernel << <numBlocks, tpb >> > (dev_pos, dev_vel, dev_acc);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(pos, dev_pos, size * sizeof(double3), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_pos);
	cudaFree(dev_vel);
	cudaFree(dev_acc);

	return cudaStatus;
}
