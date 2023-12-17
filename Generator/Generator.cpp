#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <fstream>
#include <time.h>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <omp.h>
#include "../vec3.hpp"
// #include "../vector3d.h"
#include "../dust_const.hpp"
// #include "../initializations.h"
#include "../ball_group.hpp"
// #include "../objects.h"

// File streams
std::ofstream
ballWrite,   // All ball data, pos, vel, rotation, boundness, etc
energyWrite, // Total energy of system, PE, KE, etc
constWrite;  // Ball radius, mass, and moi

// String buffer to hold data in memory until worth writing to file
std::stringstream
ballBuffer,
energyBuffer;

cluster generateBallField()
{
	cluster clus;
	clus.balls.resize(genBalls);
	// Create new random number set.
	int seedSave = time(NULL);
	srand(seedSave);

	// Make genBalls of 3 sizes in CGS with ratios such that the mass is distributed evenly among the 3 sizes (less large genBalls than small genBalls).
	int smalls = std::round((double)genBalls * 27 / 31.375); // Just here for reference. Whatever genBalls are left will be smalls.
	int mediums = std::round((double)genBalls * 27 / (8 * 31.375));
	int larges = std::round((double)genBalls * 1 / 31.375);


	for (int Ball = 0; Ball < larges; Ball++)
	{
		ball& a = clus.balls[Ball];
		a.R = 3. * scaleBalls;//pow(1. / (double)genBalls, 1. / 3.) * 3. * scaleBalls;
		a.m = density * 4. / 3. * 3.14159 * pow(a.R, 3);
		a.moi = .4 * a.m * a.R * a.R;
		a.w = { 0, 0, 0 };
		a.pos = randSphericalVec(spaceRange, spaceRange, spaceRange);
	}

	for (int Ball = larges; Ball < (larges + mediums); Ball++)
	{
		ball& a = clus.balls[Ball];
		a.R = 2. * scaleBalls;//pow(1. / (double)genBalls, 1. / 3.) * 2. * scaleBalls;
		a.m = density * 4. / 3. * 3.14159 * pow(a.R, 3);
		a.moi = .4 * a.m * a.R * a.R;
		a.w = { 0, 0, 0 };
		a.pos = randSphericalVec(spaceRange, spaceRange, spaceRange);
	}
	for (int Ball = (larges + mediums); Ball < genBalls; Ball++)
	{
		ball& a = clus.balls[Ball];
		a.R = 1. * scaleBalls;//pow(1. / (double)genBalls, 1. / 3.) * 1. * scaleBalls;
		a.m = density * 4. / 3. * 3.14159 * pow(a.R, 3);
		a.moi = .4 * a.m * a.R * a.R;
		a.w = { 0, 0, 0 };
		a.pos = randSphericalVec(spaceRange, spaceRange, spaceRange);
	}

	std::cout << "Smalls: " << smalls << " Mediums: " << mediums << " Larges: " << larges << std::endl;

	// Generate non-overlapping spherical particle field:
	int collisionDetected = 0;
	int oldCollisions = genBalls;

	for (int failed = 0; failed < attempts; failed++)
	{
		for (int A = 0; A < genBalls; A++)
		{
			ball& a = clus.balls[A];
			for (int B = A + 1; B < genBalls; B++)
			{
				ball& b = clus.balls[B];
				// Check for Ball overlap.
				double dist = (a.pos - b.pos).norm();
				double sumRaRb = a.R + b.R;
				double overlap = dist - sumRaRb;
				if (overlap < 0)
				{
					collisionDetected += 1;
					// Move the other ball:
					b.pos = randSphericalVec(spaceRange, spaceRange, spaceRange);
				}
			}
		}
		if (collisionDetected < oldCollisions)
		{
			oldCollisions = collisionDetected;
			std::cout << "Collisions: " << collisionDetected << "                        \r";
		}
		if (collisionDetected == 0)
		{
			std::cout << "\nSuccess!\n";
			break;
		}
		if (failed == attempts - 1 || collisionDetected > int(1.5 * (double)genBalls)) // Added the second part to speed up spatial constraint increase when there are clearly too many collisions for the space to be feasable.
		{
			std::cout << "Failed " << spaceRange << ". Increasing range " << spaceRangeIncrement << "cm^3.\n";
			spaceRange += spaceRangeIncrement;
			failed = 0;
			for (int Ball = 0; Ball < genBalls; Ball++)
			{
				clus.balls[Ball].pos = randSphericalVec(spaceRange, spaceRange, spaceRange); // Each time we fail and increase range, redistribute all balls randomly so we don't end up with big balls near mid and small balls outside.
			}
		}
		collisionDetected = 0;
	}
	std::cout << "Final spacerange: " << spaceRange << std::endl;
	// Calculate approximate radius of imported cluster and center mass at origin:
	vector3d comNumerator;
	for (int Ball = 0; Ball < clus.numBalls; Ball++)
	{
		ball& a = clus.balls[Ball];
		clus.m += a.m;
		comNumerator += a.m * a.pos;
	}
	clus.com = comNumerator / clus.m;

	for (int Ball = 0; Ball < clus.numBalls; Ball++)
	{
		double dist = (clus.balls[Ball].pos - clus.com).norm();
		if (dist > clus.radius)
		{
			clus.radius = dist;
		}
	}
	std::cout << "Initial Radius: " << clus.radius << std::endl;
	std::cout << "Mass: " << clus.m << std::endl;

	return clus;
}

int main(int argc, char const* argv[])
{
	cluster clus = generateBallField();

	// Cosmos has been filled with balls. Size is known:
	int ballTotal = clus.numBalls;
	std::vector<ball>& all = clus.balls;

	clus.initConditions();
	// Re-center universe mass to origin:
	for (int Ball = 0; Ball < ballTotal; Ball++)
	{
		clus.balls[Ball].pos -= clus.com;
	}
	clus.com = { 0, 0, 0 };

	outputPrefix =
		std::to_string(ballTotal) +
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
	for (int Ball = 1; Ball < ballTotal; Ball++) // Start at 2nd ball because first one was just written^.
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
	for (int Ball = 0; Ball < ballTotal; Ball++)
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
		<< clus.momentum.norm() << ','
		<< clus.angularMomentum.norm() << ','
		<< 0 << ',' //boundMass
		<< 0 << ',' //unboundMass
		<< clus.m;
	energyWrite << energyBuffer.rdbuf();
	energyBuffer.str("");

	// Reinitialize energies for next step:
	clus.KE = 0;
	clus.PE = 0;
	clus.momentum = { 0, 0, 0 };
	clus.angularMomentum = { 0, 0, 0 };

	// ball buffer:
	ballBuffer << std::endl; // Necessary new line after header.
	ballBuffer
		<< all[0].pos.x << ','
		<< all[0].pos.y << ','
		<< all[0].pos.z << ','
		<< all[0].w.x << ','
		<< all[0].w.y << ','
		<< all[0].w.z << ','
		<< all[0].w.norm() << ','
		<< all[0].vel.x << ','
		<< all[0].vel.y << ','
		<< all[0].vel.z << ','
		<< 0; //bound[0];
	for (int Ball = 1; Ball < ballTotal; Ball++)
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
	bool recordStep = false;            // This prevents writing to file every step (which is slow).
	std::cout << "Beginning simulation at...\n";

	for (int Step = 1; Step < steps; Step++) // Steps start at 1 because the 0 step is initial conditions.
	{
		// Check if this is a write step:
		if (Step % skip == 0)
		{
			recordStep = true;

			// Progress reporting:
			float eta = ((time(NULL) - startProgress) / 500.0 * (steps - Step)) / 3600.; // In seconds.
			sizeof(int);
			float elapsed = (time(NULL) - start) / 3600.;
			float progress = ((float)Step / (float)steps * 100.f);
			printf("Step: %i\tProgress: %2.0f%%\tETA: %5.2lf\tElapsed: %5.2f\n", Step, progress, eta, elapsed);
			startProgress = time(NULL);
		}
		else
		{
			recordStep = false;
		}

		// FIRST PASS - Position, send to buffer, velocity half step:
#pragma omp parallel for default(none) shared(all,ballTotal,dt)
		for (int Ball = 0; Ball < ballTotal; Ball++)
		{
			// Update velocity half step:
			all[Ball].velh = all[Ball].vel + .5 * all[Ball].acc * dt;

			// Update position:
			all[Ball].pos += all[Ball].velh * dt;

			// Reinitialize acceleration to be recalculated:
			all[Ball].acc = { 0, 0, 0 };
		}

		// SECOND PASS - Check for collisions, apply forces and torques:
		double PEchange = 0;
#pragma omp parallel for default(none) collapse(2) shared(all,recordStep,ballTotal,dt) reduction(+:PEchange)
		for (int A = 0; A < ballTotal; A++) //cuda
		{
			//#pragma omp parallel for shared(all,recordStep,A,ballTotal,dt,a) reduction(+:PEchange)
			for (int B = 0; B < ballTotal; B++)
			{
				if (B < A + 1)
				{
					continue; // I have to do this because OMP won't let me use the first iterator as a variable to the second.
				}
				double k;

				ball& a = all[A]; // THIS IS BAD. But necessary because collapse doesn't like stuff between the two loops.
				ball& b = all[B];
				double sumRaRb = a.R + b.R;
				double dist = (a.pos - b.pos).norm();
				vector3d rVecab = b.pos - a.pos;
				vector3d rVecba = a.pos - b.pos;

				// Check for collision between A and B:
				double overlap = sumRaRb - dist;
				vector3d totalForce = { 0, 0, 0 };
				vector3d aTorque = { 0, 0, 0 };
				vector3d bTorque = { 0, 0, 0 };

				// Check for collision between A and B.
				if (overlap > 0)
				{
					// Apply coefficient of restitution to balls leaving collision.
					if (dist >= a.distances[B]) // <<< huge balls x balls array
					{
						k = kout;
						//if (springTest)
						//{
						//	if (a.distances[B] < 0.9 * a.R || a.distances[B] < 0.9 * b.R)
						//	{
						//		if (a.R >= b.R)
						//		{
						//			std::cout << "Warning: Ball compression is " << .5 * (sumRaRb - a.distances[B]) / b.R << " of radius = " << b.R << std::endl;
						//		}
						//		else
						//		{
						//			std::cout << "Warning: Ball compression is " << .5 * (sumRaRb - a.distances[B]) / a.R << " of radius = " << a.R << std::endl;
						//		}
						//		//int garbo;
						//		//std::cin >> garbo;
						//	}
						//}
					}
					else
					{
						k = kin;
					}

					// Calculate force and torque for a:
					vector3d dVel = b.vel - a.vel;
					vector3d relativeVelOfA = (dVel)-((dVel).dot(rVecab)) * (rVecab / (dist * dist)) - a.w.cross(a.R / sumRaRb * rVecab) - b.w.cross(b.R / sumRaRb * rVecab);
					vector3d elasticForceOnA = -k * overlap * .5 * (rVecab / dist);
					vector3d frictionForceOnA = { 0,0,0 };
					if (relativeVelOfA.norm() > 1e-14) // When relative velocity is very low, dividing its vector components by its magnitude below is unstable.
					{
						frictionForceOnA = mu * elasticForceOnA.norm() * (relativeVelOfA / relativeVelOfA.norm());
					}
					aTorque = (a.R / sumRaRb) * rVecab.cross(frictionForceOnA);

					// Calculate force and torque for b:
					dVel = a.vel - b.vel;
					vector3d relativeVelOfB = (dVel)-((dVel).dot(rVecba)) * (rVecba / (dist * dist)) - b.w.cross(b.R / sumRaRb * rVecba) - a.w.cross(a.R / sumRaRb * rVecba);
					vector3d elasticForceOnB = -k * overlap * .5 * (rVecba / dist);
					vector3d frictionForceOnB = { 0,0,0 };
					if (relativeVelOfB.norm() > 1e-14)
					{
						frictionForceOnB = mu * elasticForceOnB.norm() * (relativeVelOfB / relativeVelOfB.norm());
					}
					bTorque = (b.R / sumRaRb) * rVecba.cross(frictionForceOnB);

					vector3d gravForceOnA = (G * a.m * b.m / pow(dist, 2)) * (rVecab / dist);
					totalForce = gravForceOnA + elasticForceOnA + frictionForceOnA;
					a.w += aTorque / a.moi * dt;
					b.w += bTorque / b.moi * dt;

					if (recordStep)
					{
						// Calculate potential energy. Important to recognize that the factor of 1/2 is not in front of K because this is for the spring potential in each ball and they are the same potential.
						PEchange += -G * all[A].m * all[B].m / dist + k * pow((all[A].R + all[B].R - dist) * .5, 2);
						//clus.PE += -G * all[A].m * all[B].m / dist + k * pow((all[A].R + all[B].R - dist) * .5, 2);
					}
				}
				else
				{
					// No collision: Include gravity only:
					vector3d gravForceOnA = (G * a.m * b.m / pow(dist, 2)) * (rVecab / dist);
					totalForce = gravForceOnA;
					if (recordStep)
					{
						PEchange += -G * all[A].m * all[B].m / dist;
						//clus.PE += -G * all[A].m * all[B].m / dist;
					}
				}
				// Newton's equal and opposite forces applied to acceleration of each ball:
				a.acc += totalForce / a.m;
				b.acc -= totalForce / b.m;

				// So last distance can be known for cor:
				a.distances[B] = b.distances[A] = dist;
			}
		}

		// THIRD PASS - Calculate velocity for next step:
		if (recordStep)
		{
			ballBuffer << std::endl; // Prepares a new line for incoming data.
		}
		for (int Ball = 0; Ball < ballTotal; Ball++)
		{
			ball& a = all[Ball];

			// Velocity for next step:
			a.vel = a.velh + .5 * a.acc * dt;

			if (recordStep)
			{
				// Adds the mass of the each ball to unboundMass if it meats these conditions:
				//bound[Ball] = false;

				// Send positions and rotations to buffer:
				if (Ball == 0)
				{
					ballBuffer
						<< a.pos[0] << ','
						<< a.pos[1] << ','
						<< a.pos[2] << ','
						<< a.w[0] << ','
						<< a.w[1] << ','
						<< a.w[2] << ','
						<< a.w.norm() << ','
						<< a.vel[0] << ','
						<< a.vel[1] << ','
						<< a.vel[2] << ','
						<< 0; //bound[0];
				}
				else
				{
					ballBuffer
						<< ','
						<< a.pos[0] << ','
						<< a.pos[1] << ','
						<< a.pos[2] << ','
						<< a.w[0] << ','
						<< a.w[1] << ','
						<< a.w[2] << ','
						<< a.w.norm() << ','
						<< a.vel[0] << ','
						<< a.vel[1] << ','
						<< a.vel[2] << ','
						<< 0; //bound[Ball];
				}

				clus.KE += .5 * a.m * a.vel.normsquared() + .5 * a.moi * a.w.normsquared(); // Now includes rotational kinetic energy.
				clus.momentum += a.m * a.vel;
				clus.angularMomentum += a.m * a.pos.cross(a.vel) + a.moi * a.w;
			}
		}
		if (recordStep)
		{
			clus.PE += PEchange; // Necessary for OMP reduction.

			// Write energy to stream:
			energyBuffer
				<< std::endl
				<< dt * Step << ','
				<< clus.PE << ','
				<< clus.KE << ','
				<< clus.PE + clus.KE << ','
				<< clus.momentum.norm() << ','
				<< clus.angularMomentum.norm() << ','
				<< 0 << ','
				<< 0 << ','
				<< clus.m; // the two zeros are bound and unbound mass

   // Reinitialize energies for next step:
			clus.KE = 0;
			clus.PE = 0;
			PEchange = 0;
			clus.momentum = { 0, 0, 0 };
			clus.angularMomentum = { 0, 0, 0 };
			// unboundMass = 0;
			// boundMass = clus.m;

			////////////////////////////////////////////////////////////////////
			// Data Export /////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////
			if (recordStep)//time(NULL) - lastWrite > 1800 || Step == steps - 1)
			{
				std::cout << "\nWriting to file.\n" << std::endl;
				// Write simData to file and clear buffer.
				ballWrite.open(simDataName, myOpenMode);
				ballWrite << ballBuffer.rdbuf(); // Barf buffer to file.
				ballBuffer.str("");              // Resets the stream for that ball to blank.
				ballWrite.close();

				// Write Energy data to file.
				energyWrite.open(energyName, myOpenMode);
				energyWrite << energyBuffer.rdbuf();
				energyBuffer.str(""); // Wipe energy buffer after write.
				energyWrite.close();

				lastWrite = time(NULL);
			} // Data export end
		}     // THIRD PASS END
	}         // Steps end
	double end = time(NULL);
	//////////////////////////////////////////////////////////
	// Loop End /////////////////////////////////////////////
	////////////////////////////////////////////////////////

	std::cout << "Simulation complete!\n"
		<< ballTotal << " Particles and " << steps << " Steps.\n"
		<< "Simulated time: " << steps * dt << " seconds\n"
		<< "Computation time: " << end - start << " seconds\n";
	std::cout << "\n===============================================================\n";
	// I know the number of balls in each file and the order they were brought in, so I can effect individual clusters.
	//
	// Implement calculation of total momentum vector and make it 0 mag

	return 0;
}
