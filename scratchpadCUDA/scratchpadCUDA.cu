#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "../cuVectorMath.h"
#include <iostream>
#include <omp.h>
#include "../misc.h"


cudaError_t double3Math(double3* pos, const double3* vel, const double3* acc, unsigned int size);

__global__ void addKernel(double3* pos, const double3* vel, const double3* acc)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	pos[i] = vel[i] * acc[i];
}



int main()
{
	// Data size
	int const fullSet = 1 << 15;
	int pairs = fullSet * (fullSet - 1) / 2;
	int2* pair = new int2[pairs];
	int count = 0;

	// Create random ball positions.
	double3* pos = new double3[fullSet];
	for (size_t i = 0; i < fullSet; i++)
	{
		pos[i] = make_double3(randDouble(1000), randDouble(1000), randDouble(1000));
	}
	double* dist = new double[pairs];

	// Create unique pairs.
	for (size_t i = 0; i < fullSet; i++)
	{
		for (size_t j = i + 1; j < fullSet; j++)
		{
			pair[count] = make_int2(i, j);
			count++;
		}
	}

	printf("Combinations = %d\n", pairs);


	double time0 = 0;
	double time1 = 0;

	/////////////////////////////////////////
	// Run pair strategy
	time0 = omp_get_wtime();
	for (size_t i = 0; i < pairs; i++)
	{
		dist[i] = mag(pos[pair[i].x] - pos[pair[i].y]);
		//printf("%d\t%d\n", pair[i].x, pair[i].y);
	}
	time1 = omp_get_wtime();
	printf("Time = %lf\n", time1 - time0);

	//// Print for optimization
	double sum = 0;
	for (size_t i = 0; i < fullSet; i++)
	{
		sum += dist[i];
	}

	printf("sum = %lf\n", sum);

	/////////////////////////////////////////
	// Run double loop with counter to minimize dist array size.
	count = 0;
	time0 = omp_get_wtime();
	for (size_t i = 0; i < fullSet; i++)
	{
		for (size_t j = i + 1; j < fullSet; j++)
		{
			dist[count] = mag(pos[i] - pos[j]);
			count++;
		}
	}
	time1 = omp_get_wtime();
	printf("Time = %lf\n", time1 - time0);

	//// Print for optimization
	sum = 0;
	for (size_t i = 0; i < fullSet; i++)
	{
		sum += dist[i];
	}

	printf("sum = %lf\n", sum);

	/////////////////////////////////////////
	// Without counter
	time0 = omp_get_wtime();
	for (size_t i = 0; i < fullSet; i++)
	{
		for (size_t j = i + 1; j < fullSet; j++)
		{
			dist[i] = mag(pos[i] - pos[j]);
		}
	}
	time1 = omp_get_wtime();
	printf("Time = %lf\n", time1 - time0);

	//// Print for optimization
	sum = 0;
	for (size_t i = 0; i < fullSet; i++)
	{
		sum += dist[i];
	}
	printf("sum = %lf\n", sum);


	//const int arraySize = 5;
	//double3* pos = new double3[arraySize];
	//double3* vel = new double3[arraySize];
	//double3* acc = new double3[arraySize];

	//for (size_t i = 0; i < arraySize; i++)
	//{
	//	vel[i] = { (double)i, (double)i, (double)i };
	//	acc[i] = { (double)i,(double)i, (double)i };
	//}

	//// Add vectors in parallel.
	//cudaError_t cudaStatus = double3Math(pos, vel, acc, arraySize);
	//if (cudaStatus != cudaSuccess)
	//{
	//	fprintf(stderr, "addWithCuda failed!");
	//	return 1;
	//}

	//for (size_t i = 0; i < arraySize; i++)
	//{
	//	printf("\nHost data:\n", vel[i].x, vel[i].y, vel[i].z);
	//	printf("Velocity:\t%lf\t%lf\t%lf\n", vel[i].x, vel[i].y, vel[i].z);
	//	printf("Accel:   \t%lf\t%lf\t%lf\n", acc[i].x, acc[i].y, acc[i].z);
	//	printf("Position:\t%lf\t%lf\t%lf\n", pos[i].x, pos[i].y, pos[i].z);
	//}


	//// cudaDeviceReset must be called before exiting in order for profiling and
	//// tracing tools such as Nsight and Visual Profiler to show complete traces.
	//cudaStatus = cudaDeviceReset();
	//if (cudaStatus != cudaSuccess)
	//{
	//	fprintf(stderr, "cudaDeviceReset failed!");
	//	return 1;
	//}

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t double3Math(double3* ans, const double3* a, const double3* b, unsigned int size)
{

	double3* dev_ans;
	double3* dev_a;
	double3* dev_b;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output).
	cudaStatus = cudaMalloc((void**)&dev_ans, size * sizeof(double3));
	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(double3));
	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(double3));

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(double3), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(double3), cudaMemcpyHostToDevice);

	// Launch a kernel on the GPU with one thread for each element.
	// Later set blockSize to 64 or something to make multiple warps per SM.
	dim3 numBlocks(1, 1, 1);
	dim3 tpb(size, 1, 1);

	addKernel << <numBlocks, tpb >> > (dev_ans, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	//cudaStatus = cudaDeviceSynchronize();
	//if (cudaStatus != cudaSuccess)
	//{
	//	fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	//	goto Error;
	//}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(ans, dev_ans, size * sizeof(double3), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_ans);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}
