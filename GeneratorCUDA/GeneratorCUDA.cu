#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <fstream>
#include <time.h>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include "../initializations.h"
#include "../objects.h"
#include "../cuVectorMath.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Create handy shorthand for error checking each step of CUDA without a bulky conditional every time:
#define CHECK (cudaStatus != cudaSuccess) ? fprintf(stderr, "Error at line %i\n", __LINE__ - 1) : NULL;


__global__ void updatePosition(double3* velh, double3* pos, const double3* vel, double3* acc, const double dt)
{
	unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
	velh[gid] = .5 * dt * acc[gid] + vel[gid];
	pos[gid] = velh[gid] * dt + pos[gid];
	acc[gid] = make_double3(0, 0, 0);
}

size_t blockSize = 64;
size_t numBlocks = numBalls / blockSize;


// File streams
std::ofstream
ballWrite,   // All ball data, pos, vel, rotation, boundness, etc
energyWrite, // Total energy of system, PE, KE, etc
constWrite;  // Ball radius, m`ass, and moi

// String buffer to hold data in memory until worth writing to file
std::stringstream
ballBuffer,
energyBuffer;

// Prototypes
cudaError_t intAddWithCuda(int* c, const int* a, const int* b, unsigned int size);


int main(int argc, char const* argv[])
{
	// Create random cluster:
	cluster clus;
	clus.populate(numBalls);
	clus.generateRandomCluster(scaleBalls, spaceRange);
	clus.initConditions(numBalls);



	outputPrefix =
		std::to_string(numBalls) +
		"-R" + scientific(clus.radius) +
		"-k" + scientific(kin) +
		"-cor" + rounder(pow(cor, 2), 4) +
		"-mu" + rounder(mu, 3) +
		"-rho" + rounder(density, 4) +
		"-dt" + rounder(dt, 4) +
		"_";
	std::cout << "New file tag: " << outputPrefix;
	// Save file names:
	std::string simDataName = outputPrefix + "simData.csv",
		constantsName = outputPrefix + "Constants.csv",
		energyName = outputPrefix + "Energy.csv";

	std::ofstream::openmode myOpenMode = std::ofstream::app;

	// Check if file name already exists.
	std::ifstream checkForFile;
	checkForFile.open(simDataName, std::ifstream::in);
	int counter = 1;
	// Add a counter to the file name until it isn't overwriting anything:
	while (checkForFile.is_open())
	{
		simDataName = std::to_string(counter) + '_' + simDataName;
		constantsName = std::to_string(counter) + '_' + constantsName;
		energyName = std::to_string(counter) + '_' + energyName;
		checkForFile.close();
		checkForFile.open(simDataName, std::ifstream::in);
		counter++;
	}

	// Open all file streams:
	energyWrite.open(energyName, myOpenMode);
	ballWrite.open(simDataName, myOpenMode);
	constWrite.open(constantsName, myOpenMode);

	// Make column headers:
	energyWrite << "Time,PE,KE,E,p,L,Bound,Unbound,m";
	ballWrite << "x0,y0,z0,w_x0,w_y0,w_z0,w_mag0,v_x0,v_y0,v_z0,comp0";
	for (int Ball = 1; Ball < numBalls; Ball++) // Start at 2nd ball because first one was just written^.
	{
		std::string thisBall = std::to_string(Ball);
		ballWrite
			<< ",x" + thisBall
			<< ",y" + thisBall
			<< ",z" + thisBall
			<< ",wx" + thisBall
			<< ",wy" + thisBall
			<< ",wz" + thisBall
			<< ",wmag" + thisBall
			<< ",vx" + thisBall
			<< ",vy" + thisBall
			<< ",vz" + thisBall
			<< ",comp" + thisBall;
	}

	std::cout << "\nSim data, energy, and constants file streams and headers created.";

	// Write constant data:
	for (int Ball = 0; Ball < numBalls; Ball++)
	{

		constWrite
			<< all[Ball].R
			<< ','
			<< all[Ball].m
			<< ','
			<< all[Ball].moi
			<< std::endl;
	}

	// Write energy data to buffer:
	energyBuffer
		<< std::endl
		<< dt << ','
		<< clus.PE << ','
		<< clus.KE << ','
		<< clus.PE + clus.KE << ','
		<< mag(clus.mom) << ','
		<< mag(clus.angMom) << ','
		<< 0 << ',' //boundMass
		<< 0 << ',' //unboundMass
		<< clus.m;
	energyWrite << energyBuffer.rdbuf();
	energyBuffer.str("");

	// Reinitialize energies for next step:
	clus.KE = 0;
	clus.PE = 0;
	clus.mom = { 0, 0, 0 };
	clus.angMom = { 0, 0, 0 };

	// ball buffer:
	ballBuffer << std::endl; // Necessary new line after header.
	ballBuffer
		<< clus.pos.x << ','
		<< clus.pos.y << ','
		<< clus.pos.z << ','
		<< clus.w.x << ','
		<< clus.w.y << ','
		<< clus.w.z << ','
		<< clus.w.norm() << ','
		<< clus.vel.x << ','
		<< clus.vel.y << ','
		<< clus.vel.z << ','
		<< 0; //bound[0];
	for (int Ball = 1; Ball < numBalls; Ball++)
	{
		ballBuffer
			<< ',' << all[Ball].pos.x << ','
			<< all[Ball].pos.y << ','
			<< all[Ball].pos.z << ','
			<< all[Ball].w.x << ','
			<< all[Ball].w.y << ','
			<< all[Ball].w.z << ','
			<< all[Ball].w.norm() << ','
			<< all[Ball].vel.x << ','
			<< all[Ball].vel.y << ','
			<< all[Ball].vel.z << ','
			<< 0; //bound[Ball];
	}
	// Write position and rotation data to file:
	ballWrite << ballBuffer.rdbuf();
	ballBuffer.str(""); // Resets the stream buffer to blank.

	// Close Streams for user viewing:
	energyWrite.close();
	ballWrite.close();
	constWrite.close();

	std::cout << "\nInitial conditions exported and file streams closed.\nSimulating " << steps * dt / 60 / 60 << " hours.\n";
	std::cout << "\n===============================================================\n";

	//////////////////////////////////////////////////////////
	// Loop Start ///////////////////////////////////////////
	////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	//////////////////////////////////////////////////////
	/////////////////////////////////////////////////////
	time_t start = time(NULL);         // For end of program anlysis
	time_t startProgress = time(NULL); // For progress reporting (gets reset)
	time_t lastWrite = time(NULL);     // For write control (gets reset)
	bool writeStep = false;            // This prevents writing to file every step (which is slow).
	std::cout << "Beginning simulation at...\n";

	for (int Step = 1; Step < steps; Step++) // Steps start at 1 because the 0 step is initial conditions.
	{
		// Check if this is a write step:
		if (Step % skip == 0)
		{
			writeStep = true;

			// Progress reporting:
			float eta = ((time(NULL) - startProgress) / skip * (steps - Step)) / 3600.; // In seconds.
			sizeof(int);
			float elapsed = (time(NULL) - start) / 3600.;
			float progress = ((float)Step / (float)steps * 100.f);
			printf("Step: %i\tProgress: %2.0f%%\tETA: %5.2lf\tElapsed: %5.2f\r", Step, progress, eta, elapsed);
			startProgress = time(NULL);
		}
		else
		{
			writeStep = false;
		}


		// FIRST PASS - Position, send to buffer, velocity half step:
		for (int Ball = 0; Ball < numBalls; Ball++)
		{
			// Update velocity half step:
			all[Ball].velh = all[Ball].vel + .5 * all[Ball].acc * dt;

			// Update position:
			all[Ball].pos += all[Ball].velh * dt;

			// Reinitialize acceleration to be recalculated:
			all[Ball].acc = { 0, 0, 0 }; // Don't really need to do this because setting = now not +=, but safer in case of += usage.
		}


		// SECOND PASS - Check for collisions, apply forces and torques:
		double k;
		for (int A = 0; A < numBalls; A++) //cuda
		{
			ball& a = all[A];

			for (int B = A + 1; B < numBalls; B++)
			{

				ball& b = all[B];
				double sumRaRb = balls[a + R_] + b.R;
				double dist = (a.pos - b.pos).norm();
				double3 rVecab = b.pos - a.pos;
				double3 rVecba = -1 * rVecab;

				// Check for collision between Ball and otherBall:
				double overlap = sumRaRb - dist;
				double3 totalForce = { 0, 0, 0 };
				double3 aTorque = { 0, 0, 0 };
				double3 bTorque = { 0, 0, 0 };

				// Check for collision between Ball and otherBall.
				if (overlap > 0)
				{
					// Apply coefficient of restitution to balls leaving collision.
					if (dist >= a.distances[B])
					{
						k = kout;
						if (springTest)
						{
							if (a.distances[B] < 0.9 * balls[a + R_] || a.distances[B] < 0.9 * b.R)
							{
								if (balls[a + R_] >= b.R)
								{
									std::cout << "Warning: Ball compression is " << .5 * (sumRaRb - a.distances[B]) / b.R << "of radius = " << b.R << std::endl;
								}
								else
								{
									std::cout << "Warning: Ball compression is " << .5 * (sumRaRb - a.distances[B]) / balls[a + R_] << "of radius = " << balls[a + R_] << std::endl;
								}
								int garbo;
								std::cin >> garbo;
							}
						}
					}
					else
					{
						k = kin;
					}

					// Calculate force and torque for a:
					double3 dVel = b.vel - a.vel;
					double3 relativeVelOfA = (dVel)-((dVel).dot(rVecab)) * (rVecab / (dist * dist)) - a.w.cross(balls[a + R_] / sumRaRb * rVecab) - b.w.cross(b.R / sumRaRb * rVecab);
					double3 elasticForceOnA = -k * overlap * .5 * (rVecab / dist);
					double3 frictionForceOnA = { 0,0,0 };
					if (relativeVelOfA.norm() > 1e-14) // When relative velocity is very low, dividing its vector components by its magnitude below is unstable.
					{
						frictionForceOnA = mu * elasticForceOnA.norm() * (relativeVelOfA / relativeVelOfA.norm());
					}
					aTorque = (balls[a + R_] / sumRaRb) * rVecab.cross(frictionForceOnA);

					// Calculate force and torque for b:
					dVel = a.vel - b.vel;
					double3 relativeVelOfB = (dVel)-((dVel).dot(rVecba)) * (rVecba / (dist * dist)) - b.w.cross(b.R / sumRaRb * rVecba) - a.w.cross(balls[a + R_] / sumRaRb * rVecba);
					double3 elasticForceOnB = -k * overlap * .5 * (rVecba / dist);
					double3 frictionForceOnB = { 0,0,0 };
					if (relativeVelOfB.norm() > 1e-14)
					{
						frictionForceOnB = mu * elasticForceOnB.norm() * (relativeVelOfB / relativeVelOfB.norm());
					}
					bTorque = (b.R / sumRaRb) * rVecba.cross(frictionForceOnB);

					double3 gravForceOnA = (G * balls[a + m_] * b.m / pow(dist, 2)) * (rVecab / dist);
					totalForce = gravForceOnA + elasticForceOnA + frictionForceOnA;
					a.w += aTorque / balls[a + moi_] * dt;
					b.w += bTorque / b.moi * dt;


					if (writeStep)
					{
						// Calculate potential energy. Important to recognize that the factor of 1/2 is not in front of K because this is for the spring potential in each ball and they are the same potential.
						clus.PE += -G * all[A].m * all[B].m / dist + k * pow((all[A].R + all[B].R - dist) * .5, 2);
						a.compression += elasticForceOnA.norm();
						b.compression += elasticForceOnB.norm();
					}
				}
				else
				{
					// No collision: Include gravity only:
					double3 gravForceOnA = (G * balls[a + m_] * b.m / pow(dist, 2)) * (rVecab / dist);
					totalForce = gravForceOnA;
					if (writeStep)
					{
						clus.PE += -G * all[A].m * all[B].m / dist;
					}
				}
				// Newton's equal and opposite forces applied to acceleration of each ball:
				a.acc = totalForce / balls[a + m_];
				b.acc = -totalForce / b.m;

				// So last distance can be known for cor:
				a.distances[B] = b.distances[A] = dist;
			}
		}

		// THIRD PASS - Calculate velocity for next step:
		if (writeStep)
		{
			ballBuffer << std::endl; // Prepares a new line for incoming data.
		}
		for (int Ball = 0; Ball < numBalls; Ball++)
		{
			ball& a = all[Ball];

			// Velocity for next step:
			a.vel = a.velh + .5 * a.acc * dt;
			if (writeStep)
			{
				// Adds the mass of the each ball to unboundMass if it meats these conditions:
				//bound[Ball] = false;

				// Send positions and rotations to buffer:
				if (Ball == 0)
				{
					ballBuffer << a.pos[0] << ',' << a.pos[1] << ',' << a.pos[2] << ',' << a.w[0] << ',' << a.w[1] << ',' << a.w[2] << ',' << a.w.norm() << ',' << a.vel[0] << ',' << a.vel[1] << ',' << a.vel[2] << ',' << a.compression;
				}
				else
				{
					ballBuffer << ',' << a.pos[0] << ',' << a.pos[1] << ',' << a.pos[2] << ',' << a.w[0] << ',' << a.w[1] << ',' << a.w[2] << ',' << a.w.norm() << ',' << a.vel[0] << ',' << a.vel[1] << ',' << a.vel[2] << ',' << a.compression;
				}
				a.compression = 0; // for next write step compression.

				clus.KE += .5 * balls[a + m_] * a.vel.normsquared() + .5 * balls[a + moi_] * a.w.normsquared(); // Now includes rotational kinetic energy.
				clus.momentum += balls[a + m_] * a.vel;
				clus.angMom += balls[a + m_] * a.pos.cross(a.vel) + balls[a + moi_] * a.w;
			}
		}
		if (writeStep)
		{
			// Write energy to stream:
			energyBuffer << std::endl
				<< dt * Step << ',' << clus.PE << ',' << clus.KE << ',' << clus.PE + clus.KE << ',' << clus.momentum.norm() << ',' << clus.angMom.norm() << ',' << 0 << ',' << 0 << ',' << clus.m; // the two zeros are bound and unbound mass

   // Reinitialize energies for next step:
			clus.KE = 0;
			clus.PE = 0;
			clus.momentum = { 0, 0, 0 };
			clus.angMom = { 0, 0, 0 };
			// unboundMass = 0;
			// boundMass = clus.m;

			////////////////////////////////////////////////////////////////////
			// Data Export /////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////
			if (writeStep)//time(NULL) - lastWrite > 1800 || Step == steps - 1)
			{ // ballBuffer.tellp() >= 100000000
				std::cout << "\nData Write" << std::endl;
				//std::cout << "\nWriting: " << ballBuffer.tellp() << " Bytes. Dumped to file.\n";
				//auto begin = std::chrono::high_resolution_clock::now();
				ballWrite.open(simDataName, myOpenMode);
				ballWrite << ballBuffer.rdbuf(); // Barf buffer to file.
				ballBuffer.str("");              // Resets the stream for that ball to blank.
				ballWrite.close();

				// Write Energy data to file.
				energyWrite.open(energyName, myOpenMode);
				energyWrite << energyBuffer.rdbuf();
				energyBuffer.str(""); // Wipe energy buffer after write.
				energyWrite.close();

				//auto end = std::chrono::high_resolution_clock::now();
				//std::cout << "Write time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1000000 << " milliseconds\n";
				lastWrite = time(NULL);
			} // Data export end
		}     // THIRD PASS END
	}         // Steps end
	double end = time(NULL);
	//////////////////////////////////////////////////////////
	// Loop End /////////////////////////////////////////////
	////////////////////////////////////////////////////////

	std::cout << "Simulation complete!\n"
		<< numBalls << " Particles and " << steps << " Steps.\n"
		<< "Simulated time: " << steps * dt << " seconds\n"
		<< "Computation time: " << end - start << " seconds\n";
	std::cout << "\n===============================================================\n";
	// I know the number of balls in each file and the order they were brought in, so I can effect individual clusters.
	//
	// Implement calculation of total momentum vector and make it 0 mag

	clus.freeMemory();
	return 0;
}

//int main()
//{
//	int* a = new int[numBalls];
//	int* b = new int[numBalls];
//	for (size_t i = 0; i < numBalls; i++)
//	{
//		a[i] = i;
//		b[i] = i;
//	}
//	int* c = new int[numBalls];
//
//	// Add vectors in parallel.
//	cudaError_t cudaStatus = intAddWithCuda(c, a, b, numBalls);
//	
//	return 0;
//}

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
	updatePosition << <numBlocks, blockSize >> > (dev_c, dev_a, dev_b);

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


