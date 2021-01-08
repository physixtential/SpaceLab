#define _USE_MATH_DEFINES
#include <iostream>
#include <fstream>
#include <time.h>
#include <sstream>
#include <stdio.h>
#include "math.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../cuVectorMath.h"
#include "../initializations.h"
#include "../misc.h"
#include "../objects.h"

// Create handy shorthand for error checking each step of CUDA without a bulky conditional every time:
#define CHECK (cudaStatus != cudaSuccess) ? fprintf(stderr, "Error at line %i\n", __LINE__ - 1) : NULL;


__global__ void updatePosition(double3* velh, double3* pos, const double3* vel, double3* acc, const double dt)
{
	unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
	// Update velocity half step:
	velh[gid] = vel[gid] + .5 * acc[gid] * dt;

	// Update position:
	pos[gid] += velh[gid] * dt;

	// Reinitialize acceleration to be recalculated:
	acc[gid] = { 0, 0, 0 }; // Must reset because += acc from all other balls, not just =.
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
cudaError_t loopOneCUDA(double3* velh, double3* pos, double3* vel, double3* acc, const double dt, const unsigned int size, const unsigned int numSteps);

int main(int argc, char const* argv[])
{
	// Create random cluster:
	ballGroup clus;
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
		constantsName = outputPrefix + "constants.csv",
		energyName = outputPrefix + "energy.csv";

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
	ballWrite << "x0,y0,z0,wx0,wy0,wz0,wmag0,vx0,vy0,vz0,comp0";
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
			<< clus.R[Ball]
			<< ','
			<< clus.m[Ball]
			<< ','
			<< clus.moi[Ball]
			<< std::endl;
	}

	// Write energy data to buffer:
	energyBuffer
		<< std::endl
		<< dt << ','
		<< clus.PE << ','
		<< clus.KE << ','
		<< clus.PE + clus.KE << ','
		<< length(clus.mom) << ','
		<< length(clus.angMom) << ','
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
		<< clus.pos[0].x << ','
		<< clus.pos[0].y << ','
		<< clus.pos[0].z << ','
		<< clus.w[0].x << ','
		<< clus.w[0].y << ','
		<< clus.w[0].z << ','
		<< length(clus.w[0]) << ','
		<< clus.vel[0].x << ','
		<< clus.vel[0].y << ','
		<< clus.vel[0].z << ','
		<< 0; //bound[0];
	for (int Ball = 1; Ball < numBalls; Ball++)
	{
		ballBuffer
			<< ',' << clus.pos[Ball].x << ','
			<< clus.pos[Ball].y << ','
			<< clus.pos[Ball].z << ','
			<< clus.w[Ball].x << ','
			<< clus.w[Ball].y << ','
			<< clus.w[Ball].z << ','
			<< length(clus.w[Ball]) << ','
			<< clus.vel[Ball].x << ','
			<< clus.vel[Ball].y << ','
			<< clus.vel[Ball].z << ','
			<< 0; //bound[0];
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
		//for (int Ball = 0; Ball < numBalls; Ball++)
		//{
		//	// Update velocity half step:
		//	clus.velh[Ball] = clus.vel[Ball] + .5 * clus.acc[Ball] * dt;

		//	// Update position:
		//	clus.pos[Ball] += clus.velh[Ball] * dt;

		//	// Reinitialize acceleration to be recalculated:
		//	clus.acc[Ball] = { 0, 0, 0 }; // Must reset because += acc from all other balls, not just =.
		//}
		loopOneCUDA(clus.velh, clus.pos, clus.vel, clus.acc, dt, numBalls, steps);


		// SECOND PASS - Check for collisions, apply forces and torques:
		double k;
		for (int A = 1; A < numBalls; A++) //cuda
		{
			for (int B = 0; B < A; B++)
			{
				double sumRaRb = clus.R[A] + clus.R[B];
				double dist = length(clus.pos[A] - clus.pos[B]);
				double3 rVecab = clus.pos[B] - clus.pos[A];
				double3 rVecba = -1 * rVecab;

				// Check for collision between Ball and otherBall:
				double overlap = sumRaRb - dist;
				double3 totalForce = { 0, 0, 0 };
				double3 aTorque = { 0, 0, 0 };
				double3 bTorque = { 0, 0, 0 };

				// Distance array element: 1,0    2,0    2,1    3,0    3,1    3,2 ...
				int e = (A * (A - 1) * .5) + B;
				double oldDist = clus.distances[e];

				// Check for collision between Ball and otherBall.
				if (overlap > 0)
				{
					// Apply coefficient of restitution to balls leaving collision.
					if (dist >= oldDist)
					{
						k = kout;
						//if (springTest)
						//{
						//	if (oldDist < 0.9 * clus.R[A] || oldDist < 0.9 * clus.R[B])
						//	{
						//		if (clus.R[A] >= clus.R[B])

						//		{
						//			std::cout << "Warning: Ball compression is " << .5 * (sumRaRb - oldDist) / clus.R[B] << "of radius = " << clus.R[B] << std::endl;
						//		}
						//		else
						//		{
						//			std::cout << "Warning: Ball compression is " << .5 * (sumRaRb - oldDist) / clus.R[A] << "of radius = " << clus.R[A] << std::endl;
						//		}
						//		int garbo;
						//		std::cin >> garbo;
						//	}
						//}
					}
					else
					{
						k = kin;
					}

					// Calculate force and torque for a:
					double3 dVel = clus.vel[B] - clus.vel[A];
					double3 relativeVelOfA = dVel - dot(dVel, rVecab) * (rVecab / (dist * dist)) - cross(clus.w[A], clus.R[A] / sumRaRb * rVecab) - cross(clus.w[B], clus.R[B] / sumRaRb * rVecab);
					double3 elasticForceOnA = -kin * overlap * .5 * (rVecab / dist);
					double3 frictionForceOnA = { 0,0,0 };
					if (length(relativeVelOfA) > 1e-12) // When relative velocity is very low, dividing its vector components by its magnitude below is unstable.
					{
						frictionForceOnA = mu * length(elasticForceOnA) * (relativeVelOfA / length(relativeVelOfA));
					}
					aTorque = (clus.R[A] / sumRaRb) * cross(rVecab, frictionForceOnA);

					// Calculate force and torque for b:
					dVel = clus.vel[A] - clus.vel[B];
					double3 relativeVelOfB = dVel - dot(dVel, rVecba) * (rVecba / (dist * dist)) - cross(clus.w[B], clus.R[B] / sumRaRb * rVecba) - cross(clus.w[A], clus.R[A] / sumRaRb * rVecba);
					double3 elasticForceOnB = -kin * overlap * .5 * (rVecba / dist);
					double3 frictionForceOnB = { 0,0,0 };
					if (length(relativeVelOfB) > 1e-12)
					{
						frictionForceOnB = mu * length(elasticForceOnB) * (relativeVelOfB / length(relativeVelOfB));
					}
					bTorque = (clus.R[B] / sumRaRb) * cross(rVecba, frictionForceOnB);

					double3 gravForceOnA = (G * clus.m[A] * clus.m[B] / pow(dist, 2)) * (rVecab / dist);
					totalForce = gravForceOnA + elasticForceOnA + frictionForceOnA;
					clus.w[A] += aTorque / clus.moi[A] * dt;
					clus.w[B] += bTorque / clus.moi[B] * dt;
					clus.PE += -G * clus.m[A] * clus.m[B] / dist + kin * pow((sumRaRb - dist) * .5, 2);


					if (writeStep)
					{
						// Calculate potential energy. Important to recognize that the factor of 1/2 is not in front of K because this is for the spring potential in each ball and they are the same potential.
						clus.PE += -G * clus.m[A] * clus.m[B] / dist + k * pow((clus.R[A] + clus.R[B] - dist) * .5, 2);
					}
				}
				else
				{
					// No collision: Include gravity only:
					double3 gravForceOnA = (G * clus.m[A] * clus.m[B] / pow(dist, 2)) * (rVecab / dist);
					totalForce = gravForceOnA;
					if (writeStep)
					{
						clus.PE += -G * clus.m[A] * clus.m[B] / dist;
					}
				}
				// Newton's equal and opposite forces applied to acceleration of each ball:
				clus.acc[A] += totalForce / clus.m[A];
				clus.acc[B] += -totalForce / clus.m[B];

				// So last distance can be known for cor:
				clus.distances[e] = dist;
			}
		}

		// THIRD PASS - Calculate velocity for next step:
		if (writeStep)
		{
			ballBuffer << std::endl; // Prepares a new line for incoming data.
		}
		for (int Ball = 0; Ball < numBalls; Ball++)
		{
			// Velocity for next step:
			clus.vel[Ball] = clus.velh[Ball] + .5 * clus.acc[Ball] * dt;
			if (writeStep)
			{
				// Adds the mass of the each ball to unboundMass if it meats these conditions:
				//bound[Ball] = false;

				// Send positions and rotations to buffer:
				if (Ball == 0)
				{
					ballBuffer
						<< clus.pos[0].x << ','
						<< clus.pos[0].y << ','
						<< clus.pos[0].z << ','
						<< clus.w[0].x << ','
						<< clus.w[0].y << ','
						<< clus.w[0].z << ','
						<< length(clus.w[0]) << ','
						<< clus.vel[0].x << ','
						<< clus.vel[0].y << ','
						<< clus.vel[0].z << ','
						<< 0;
				}
				else
				{
					ballBuffer
						<< ',' << clus.pos[Ball].x << ','
						<< clus.pos[Ball].y << ','
						<< clus.pos[Ball].z << ','
						<< clus.w[Ball].x << ','
						<< clus.w[Ball].y << ','
						<< clus.w[Ball].z << ','
						<< length(clus.w[Ball]) << ','
						<< clus.vel[Ball].x << ','
						<< clus.vel[Ball].y << ','
						<< clus.vel[Ball].z << ','
						<< 0;
				}

				clus.KE += .5 * clus.m[Ball] * dot(clus.vel[Ball], clus.vel[Ball]) + .5 * clus.moi[Ball] * dot(clus.w[Ball], clus.w[Ball]); // Now includes rotational kinetic energy.
				clus.mom += clus.m[Ball] * clus.vel[Ball];
				clus.angMom += clus.m[Ball] * cross(clus.pos[Ball], clus.vel[Ball]) + clus.moi[Ball] * clus.w[Ball];
			}
		}
		if (writeStep)
		{
			// Write energy to stream:
			energyBuffer << std::endl
				<< dt * Step << ',' << clus.PE << ',' << clus.KE << ',' << clus.PE + clus.KE << ',' << length(clus.mom) << ',' << length(clus.angMom) << ',' << 0 << ',' << 0 << ',' << clus.mTotal; // the two zeros are bound and unbound mass

   // Reinitialize energies for next step:
			clus.KE = 0;
			clus.PE = 0;
			clus.mom = { 0, 0, 0 };
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
	// Implement calculation of total mom vector and make it 0 length

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
cudaError_t loopOneCUDA(double3* velh, double3* pos, double3* vel, double3* acc, const double dt, const unsigned int size, const unsigned int numSteps)
{
	double3* dev_velh = 0;
	double3* dev_pos = 0;
	double3* dev_vel = 0;
	double3* dev_acc = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	CHECK;

	// Allocate GPU buffers for 4 vectors.
	cudaStatus = cudaMalloc((void**)&dev_velh, size * sizeof(double3));
	CHECK;
	cudaStatus = cudaMalloc((void**)&dev_pos, size * sizeof(double3));
	CHECK;
	cudaStatus = cudaMalloc((void**)&dev_vel, size * sizeof(double3));
	CHECK;
	cudaStatus = cudaMalloc((void**)&dev_acc, size * sizeof(double3));
	CHECK;

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_velh, velh, size * sizeof(double3), cudaMemcpyHostToDevice);
	CHECK;
	cudaStatus = cudaMemcpy(dev_pos, pos, size * sizeof(double3), cudaMemcpyHostToDevice);
	CHECK;
	cudaStatus = cudaMemcpy(dev_vel, vel, size * sizeof(double3), cudaMemcpyHostToDevice);
	CHECK;
	cudaStatus = cudaMemcpy(dev_acc, acc, size * sizeof(double3), cudaMemcpyHostToDevice);
	CHECK;

	// Need to copy all ball data to GPU so we can just iterate all physics loops and stay on gpu
	// The kernel launch loop is per time step not per loop. All 3 loops will happen per thread (ball or ball pair)

	// Launch a kernel on the GPU with one thread for each element.
	//for (size_t step = 0; step < numSteps; step++) // actually need to stop 500 or 1000 and copy back then launch again.
	//{
	updatePosition <<<numBlocks, blockSize >>> (dev_velh, dev_pos, dev_vel, dev_acc, dt);
	//}

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	CHECK;

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	CHECK;

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(velh, dev_velh, size * sizeof(double3), cudaMemcpyDeviceToHost);
	CHECK;
	cudaStatus = cudaMemcpy(pos, dev_pos, size * sizeof(double3), cudaMemcpyDeviceToHost);
	CHECK;
	cudaStatus = cudaMemcpy(vel, dev_vel, size * sizeof(double3), cudaMemcpyDeviceToHost);
	CHECK;
	cudaStatus = cudaMemcpy(velh, dev_acc, size * sizeof(double3), cudaMemcpyDeviceToHost);
	CHECK;



	cudaStatus = cudaDeviceSynchronize();
	CHECK;

	cudaFree(dev_velh);
	cudaFree(dev_pos);
	cudaFree(dev_vel);
	cudaFree(dev_acc);

	return cudaStatus;
}


