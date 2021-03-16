#pragma once
#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <fstream>
#include <time.h>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include "../vector3d.h"
#include "../initializations.h"
#include "../objects.h"

// File streams
std::ofstream
ballWrite,   // All ball data, pos, vel, rotation, boundness, etc
energyWrite, // Total energy of system, PE, KE, etc
constWrite;  // Ball radius, mass, and moi

// String buffer to hold data in memory until worth writing to file
std::stringstream
ballBuffer,
energyBuffer;

ballGroup O;

// Prototypes
void simInitTwoCluster();
void simContinue();
void simInitCondAndCenter();
void simInitWrite();
void simOneStep(int Step);
void simLooper();
ballGroup importDataFromFile(std::string initDataFileName, std::string initConstFileName);
void generateBallField();
void safetyChecks();
void calibrateDT(int Step, bool superSafe, bool doK);
void setGuidDT(double vel);
void setGuidK(double vel);
void setLazzDT(double vel);
void setLazzK(double vel);

// Main function
int main(int argc, char const* argv[])
{
	// Runtime arguments:
	double spins[3] = { 0 };
	if (argc > 1)
	{
		//numThreads = atoi(argv[1]);
		//printf("\nThread count set to %i.\n", numThreads);
		//projectileName = argv[2];
		//targetName = argv[3];
		//KEfactor = atof(argv[4]);
	}

	//simInitTwoCluster();
	//simContinue();
	generateBallField();
	simInitCondAndCenter();
	safetyChecks();
	simInitWrite();
	simLooper();

	return 0;
} // end main

void simInitTwoCluster()
{
	// Load file data:
	std::cerr << "TWO CLUSTER SIM\nFile 1: " << projectileName << '\t' << "File 2: " << targetName << std::endl;
	//ballGroup projectile = importDataFromFile(path + projectileName + "simData.csv", path + projectileName + "constants.csv");

	// DART PROBE
	ballGroup projectile;
	projectile.allocateGroup(1);
	projectile.pos[0] = { 8800,0,0 };
	projectile.w[0] = { 0,0,0 };
	projectile.vel[0] = { 0,0,0 };
	projectile.R[0] = 78.5;
	projectile.m[0] = 560000;
	projectile.moi[0] = .4 * projectile.m[0] * projectile.R[0] * projectile.R[0];

	ballGroup target = importDataFromFile(path + targetName + "simData.csv", path + targetName + "constants.csv");

	// DO YOU WANT TO STOP EVERYTHING?
	projectile.zeroMotion();
	target.zeroMotion();

	// Calc info to determined cluster positioning and collisions velocity:
	projectile.updateComAndMass();
	target.updateComAndMass();

	projectile.updateRadius();
	target.updateRadius();

	projectile.updatePE();
	target.updatePE();

	//projectile.offset(projectile.radius, target.radius + (projectile.R[0]), impactParameter);

	double PEsys = projectile.PE + target.PE + (-G * projectile.mTotal * target.mTotal / (projectile.com - target.com).norm());

	// Collision velocity calculation:
	double mSmall = projectile.mTotal;
	double mBig = target.mTotal;
	double mTot = mBig + mSmall;
	double vSmall = -sqrt(2 * KEfactor * fabs(PEsys) * (mBig / (mSmall * mTot))); // Negative because small offsets right.
	//vSmall = -600000; // DART probe override.
	double vBig = -(mSmall / mBig) * vSmall; // Negative to be opposing projectile.
	//vBig = 0; // Dymorphous override.
	fprintf(stdout, "\nTarget Velocity: %.2e\nProjectile Velocity: %.2e\n", vBig, vSmall);

	if (isnan(vSmall) || isnan(vBig))
	{
		fprintf(stderr, "A VELOCITY WAS NAN!!!!!!!!!!!!!!!!!!!!!!\n\n");
		exit(EXIT_FAILURE);
	}
	projectile.kick(vSmall, 0, 0);
	target.kick(vBig, 0, 0);

	std::cout << std::endl;
	projectile.checkMomentum("Projectile");
	target.checkMomentum("Target");

	O.allocateGroup(projectile.cNumBalls + target.cNumBalls);

	O.addBallGroup(&target);
	O.addBallGroup(&projectile); // projectile second so smallest ball at end and largest ball at front for dt/k calcs.

	outputPrefix =
		projectileName + targetName +
		"T" + rounder(KEfactor, 4) +
		"-vBig" + scientific(vBig) +
		"-vSmall" + scientific(vSmall) +
		"-IP" + rounder(impactParameter * 180 / 3.14159, 2) +
		"-rho" + rounder(density, 4);
}


void simContinue()
{
	// Load file data:
	std::cerr << "Continuing Sim...\nFile: " << targetName << std::endl;

	O = importDataFromFile(path + targetName + "simData.csv", path + targetName + "constants.csv");

	std::cout << std::endl;
	O.checkMomentum("O");

	std::cout << "==================" << std::endl;
	std::cout << "dt: " << dt << std::endl;
	std::cout << "k: " << kin << std::endl;
	std::cout << "Steps: " << steps << std::endl;
	std::cout << "==================" << std::endl;


	// Name the file based on info above:
	outputPrefix =
		projectileName + targetName +
		"T" + rounder(KEfactor, 4) +
		"-rho" + rounder(density, 4);
}


void simInitCondAndCenter()
{


	calibrateDT(0, false, true);


	//////////////////////////////////////////
	// k and dt override to stabilize cluster.
	// 
	//kin = 1.01787e16;
	//kout = cor * kin;
	//dt = 1e-6;
	//steps = (size_t)(simTimeSeconds / dt);

	std::cout << "==================" << std::endl;
	std::cout << "dt: " << dt << std::endl;
	std::cout << "k: " << kin << std::endl;
	std::cout << "Steps: " << steps << std::endl;
	std::cout << "==================" << std::endl;



	O.checkMomentum("After Zeroing"); // Is total mom zero like it should be?

	O.toOrigin();

	// Compute physics between all balls. Distances, collision forces, energy totals, total mass:
	O.initConditions();

	// Name the file based on info above:
	outputPrefix +=
		"-k" + scientific(kin) +
		"-dt" + scientific(dt) +
		"_";
}




std::string simDataName;
std::string constantsName;
std::string energyName;
std::ofstream::openmode myOpenMode = std::ofstream::app;

void simInitWrite()
{
	// Create string for file name identifying spin combination negative is 2, positive is 1 on each axis.
	//std::string spinCombo = "";
	//for (int i = 0; i < 3; i++)
	//{
	//	if (spins[i] < 0) { spinCombo += "2"; }
	//	else if (spins[i] > 0) { spinCombo += "1"; }
	//	else { spinCombo += "0"; }
	//}

	// Save file names:
	simDataName = outputPrefix + "simData.csv";
	constantsName = outputPrefix + "constants.csv";
	energyName = outputPrefix + "energy.csv";



	// Check if file name already exists.
	std::ifstream checkForFile;
	checkForFile.open(simDataName, std::ifstream::in);
	int counter = 0;
	// Add a counter to the file name until it isn't overwriting anything:
	if (checkForFile.is_open())
	{
		while (true)
		{
			if (checkForFile.is_open())
			{
				counter++;
				checkForFile.close();
				checkForFile.open(std::to_string(counter) + '_' + simDataName, std::ifstream::in);
			}
			else
			{
				simDataName = std::to_string(counter) + '_' + simDataName;
				constantsName = std::to_string(counter) + '_' + constantsName;
				energyName = std::to_string(counter) + '_' + energyName;
				break;
			}
		}
	}
	std::cout << "New file tag: " << simDataName;

	// Open all file streams:
	energyWrite.open(energyName, myOpenMode);
	ballWrite.open(simDataName, myOpenMode);
	constWrite.open(constantsName, myOpenMode);

	// Make column headers:
	energyWrite << "Time,PE,KE,E,p,L,Bound,Unbound,mTotal";
	ballWrite << "x0,y0,z0,wx0,wy0,wz0,wmag0,vx0,vy0,vz0,bound0";

	for (int Ball = 1; Ball < O.cNumBalls; Ball++) // Start at 2nd ball because first one was just written^.
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
			<< ",bound" + thisBall;
	}

	std::cout << "\nSim data, energy, and constants file streams and headers created.";

	// Write constant data:
	for (int Ball = 0; Ball < O.cNumBalls; Ball++)
	{

		constWrite
			<< O.R[Ball] << ','
			<< O.m[Ball] << ','
			<< O.moi[Ball]
			<< std::endl;
	}

	// Write energy data to buffer:
	energyBuffer
		<< std::endl
		<< simTimeElapsed << ','
		<< O.PE << ','
		<< O.KE << ','
		<< O.PE + O.KE << ','
		<< O.mom.norm() << ','
		<< O.angMom.norm() << ','
		<< 0 << ',' //boundMass
		<< 0 << ',' //unboundMass
		<< O.mTotal;
	energyWrite << energyBuffer.rdbuf();
	energyBuffer.str("");

	// Reinitialize energies for next step:
	O.KE = 0;
	O.PE = 0;
	O.mom = { 0, 0, 0 };
	O.angMom = { 0, 0, 0 };

	// Send position and rotation to buffer:
	ballBuffer << std::endl; // Necessary new line after header.
	ballBuffer
		<< O.pos[0].x << ','
		<< O.pos[0].y << ','
		<< O.pos[0].z << ','
		<< O.w[0].x << ','
		<< O.w[0].y << ','
		<< O.w[0].z << ','
		<< O.w[0].norm() << ','
		<< O.vel[0].x << ','
		<< O.vel[0].y << ','
		<< O.vel[0].z << ','
		<< 0; //bound[0];
	for (int Ball = 1; Ball < O.cNumBalls; Ball++)
	{
		ballBuffer
			<< ',' << O.pos[Ball].x << ',' // Needs comma start so the last bound doesn't have a dangling comma.
			<< O.pos[Ball].y << ','
			<< O.pos[Ball].z << ','
			<< O.w[Ball].x << ','
			<< O.w[Ball].y << ','
			<< O.w[Ball].z << ','
			<< O.w[Ball].norm() << ','
			<< O.vel[Ball].x << ','
			<< O.vel[Ball].y << ','
			<< O.vel[Ball].z << ','
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
	std::cout << "Total mass: " << O.mTotal << std::endl;
	std::cout << "\n===============================================================\n";
}




time_t start = time(NULL);        // For end of program analysis
time_t startProgress; // For progress reporting (gets reset)
time_t lastWrite;     // For write control (gets reset)
bool writeStep;       // This prevents writing to file every step (which is slow).

void simOneStep(int Step)
{
	// Check if this is a write step:
	if (Step % skip == 0)
	{
		writeStep = true;

		// Progress reporting:
		float eta = ((time(NULL) - startProgress) / 500.0 * (steps - Step)) / 3600.; // In seconds.
		sizeof(int);
		float elapsed = (time(NULL) - start) / 3600.;
		float progress = ((float)Step / (float)steps * 100.f);
		printf("Step: %i\tProgress: %2.0f%%\tETA: %5.2lf hr\tElapsed: %5.2f hr\n", Step, progress, eta, elapsed);
		startProgress = time(NULL);
	}
	else
	{
		writeStep = false;
	}

	/// FIRST PASS - Position, send to buffer, velocity half step:
	for (int Ball = 0; Ball < O.cNumBalls; Ball++)
	{
		// Update velocity half step:
		O.velh[Ball] = O.vel[Ball] + .5 * O.acc[Ball] * dt;

		// Update angular velocity half step:
		O.wh[Ball] = O.w[Ball] + .5 * O.aacc[Ball] * dt;

		// Update position:
		O.pos[Ball] += O.velh[Ball] * dt;

		// Reinitialize acceleration to be recalculated:
		O.acc[Ball] = { 0, 0, 0 };

		// Reinitialize angular acceleration to be recalculated:
		O.aacc[Ball] = { 0, 0, 0 };
	}

	/// SECOND PASS - Check for collisions, apply forces and torques:
	for (int A = 1; A < O.cNumBalls; A++) //cuda
	{
		/// DONT DO ANYTHING HERE. A STARTS AT 1.
		for (int B = 0; B < A; B++)
		{
			double k;
			double sumRaRb = O.R[A] + O.R[B];
			vector3d rVecab = O.pos[B] - O.pos[A];
			vector3d rVecba = -1 * rVecab;
			double dist = (rVecab).norm();

			// Check for collision between Ball and otherBall:
			double overlap = sumRaRb - dist;
			vector3d totalForce = { 0, 0, 0 };
			vector3d aTorque = { 0, 0, 0 };
			vector3d bTorque = { 0, 0, 0 };

			// Distance array element: 1,0    2,0    2,1    3,0    3,1    3,2 ...
			int e = (A * (A - 1) * .5) + B;
			double oldDist = O.distances[e];

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
				vector3d dVel = O.vel[B] - O.vel[A];
				vector3d relativeVelOfA = (dVel)-((dVel).dot(rVecab)) * (rVecab / (dist * dist)) - O.w[A].cross(O.R[A] / sumRaRb * rVecab) - O.w[B].cross(O.R[B] / sumRaRb * rVecab);
				vector3d elasticForceOnA = -k * overlap * .5 * (rVecab / dist);
				vector3d frictionForceOnA = { 0,0,0 };
				if (relativeVelOfA.norm() > 1e-12) // When relative velocity is very low, dividing its vector components by its magnitude below is unstable.
				{
					frictionForceOnA = mu * elasticForceOnA.norm() * (relativeVelOfA / relativeVelOfA.norm());
				}
				aTorque = (O.R[A] / sumRaRb) * rVecab.cross(frictionForceOnA);

				// Calculate force and torque for b:
				dVel = O.vel[A] - O.vel[B];
				vector3d relativeVelOfB = (dVel)-((dVel).dot(rVecba)) * (rVecba / (dist * dist)) - O.w[B].cross(O.R[B] / sumRaRb * rVecba) - O.w[A].cross(O.R[A] / sumRaRb * rVecba);
				vector3d elasticForceOnB = -k * overlap * .5 * (rVecba / dist);
				vector3d frictionForceOnB = { 0,0,0 };
				if (relativeVelOfB.norm() > 1e-12)
				{
					frictionForceOnB = mu * elasticForceOnB.norm() * (relativeVelOfB / relativeVelOfB.norm());
				}
				bTorque = (O.R[B] / sumRaRb) * rVecba.cross(frictionForceOnB);

				vector3d gravForceOnA = (G * O.m[A] * O.m[B] / (dist * dist)) * (rVecab / dist);
				totalForce = gravForceOnA + elasticForceOnA + frictionForceOnA;
				O.aacc[A] += aTorque / O.moi[A];
				O.aacc[B] += bTorque / O.moi[B];

				if (writeStep)
				{
					// Calculate potential energy. Important to recognize that the factor of 1/2 is not in front of K because this is for the spring potential in each ball and they are the same potential.
					O.PE += -G * O.m[A] * O.m[B] / dist + k * ((O.R[A] + O.R[B] - dist) * .5) * ((O.R[A] + O.R[B] - dist) * .5);
				}
			}
			else
			{
				// No collision: Include gravity only:
				vector3d gravForceOnA = (G * O.m[A] * O.m[B] / (dist * dist)) * (rVecab / dist);
				totalForce = gravForceOnA;
				if (writeStep)
				{
					O.PE += -G * O.m[A] * O.m[B] / dist;
				}

				// For expanding overlappers:
				//O.vel[A] = { 0,0,0 };
				//O.vel[B] = { 0,0,0 };
			}

			// Newton's equal and opposite forces applied to acceleration of each ball:
			O.acc[A] += totalForce / O.m[A];
			O.acc[B] -= totalForce / O.m[B];

			// So last distance can be known for cor:
			O.distances[e] = dist;
		}
		// DONT DO ANYTHING HERE. A STARTS AT 1.
	}

	if (writeStep)
	{
		ballBuffer << std::endl; // Prepares a new line for incoming data.
	}

	// THIRD PASS - Calculate velocity for next step:
	for (int Ball = 0; Ball < O.cNumBalls; Ball++)
	{

		// Velocity for next step:
		O.vel[Ball] = O.velh[Ball] + .5 * O.acc[Ball] * dt;
		O.w[Ball] = O.wh[Ball] + .5 * O.aacc[Ball] * dt;

		if (writeStep)
		{

			// Send positions and rotations to buffer:
			if (Ball == 0)
			{
				ballBuffer << O.pos[Ball][0] << ',' << O.pos[Ball][1] << ',' << O.pos[Ball][2] << ',' << O.w[Ball][0] << ',' << O.w[Ball][1] << ',' << O.w[Ball][2] << ',' << O.w[Ball].norm() << ',' << O.vel[Ball].x << ',' << O.vel[Ball].y << ',' << O.vel[Ball].z << ',' << 0;
			}
			else
			{
				ballBuffer << ',' << O.pos[Ball][0] << ',' << O.pos[Ball][1] << ',' << O.pos[Ball][2] << ',' << O.w[Ball][0] << ',' << O.w[Ball][1] << ',' << O.w[Ball][2] << ',' << O.w[Ball].norm() << ',' << O.vel[Ball].x << ',' << O.vel[Ball].y << ',' << O.vel[Ball].z << ',' << 0;
			}

			O.KE += .5 * O.m[Ball] * O.vel[Ball].normsquared() + .5 * O.moi[Ball] * O.w[Ball].normsquared(); // Now includes rotational kinetic energy.
			O.mom += O.m[Ball] * O.vel[Ball];
			O.angMom += O.m[Ball] * O.pos[Ball].cross(O.vel[Ball]) + O.moi[Ball] * O.w[Ball];
		}
	} // THIRD PASS END

	if (writeStep || Step == steps - 1)
	{
		// Write energy to stream:
		energyBuffer << std::endl
			<< simTimeElapsed << ',' << O.PE << ',' << O.KE << ',' << O.PE + O.KE << ',' << O.mom.norm() << ',' << O.angMom.norm() << ',' << 0 << ',' << 0 << ',' << O.mTotal; // the two zeros are bound and unbound mass

		// Reinitialize energies for next step:
		O.KE = 0;
		O.PE = 0;
		O.mom = { 0, 0, 0 };
		O.angMom = { 0, 0, 0 };
		// unboundMass = 0;
		// boundMass = O.mTotal;

		////////////////////////////////////////////////////////////////////
		// Data Export /////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////
		if (time(NULL) - lastWrite > 1800 || Step / skip % 20 == 0 || Step == steps - 1)
		{
			std::cout << "\nData Write" << std::endl;

			// Write simData to file and clear buffer.
			ballWrite.open(simDataName, myOpenMode);
			ballWrite << ballBuffer.rdbuf(); // Barf buffer to file.
			ballBuffer.str("");              // Empty the stream for next filling.
			ballWrite.close();

			// Write Energy data to file and clear buffer.
			energyWrite.open(energyName, myOpenMode);
			energyWrite << energyBuffer.rdbuf();
			energyBuffer.str(""); // Empty the stream for next filling.
			energyWrite.close();

			lastWrite = time(NULL);
		} // Data export end
		//calibrateDT(Step, false, false);
		simTimeElapsed += dt * skip;
	} // writestep end
} // Steps end


void simLooper()
{
	std::cout << "Beginning simulation...\n";

	for (int Step = 1; Step < steps; Step++) // Steps start at 1 because the 0 step is initial conditions.
	{
		simOneStep(Step);
	}
	time_t end = time(NULL);

	std::cout << "Simulation complete!\n"
		<< O.cNumBalls << " Particles and " << steps << " Steps.\n"
		<< "Simulated time: " << steps * dt << " seconds\n"
		<< "Computation time: " << end - start << " seconds\n";
	std::cout << "\n===============================================================\n";
	// I know the number of balls in each file and the order they were brought in, so I can effect individual clusters.
	//
	// Implement calculation of total mom vector and make it 0 mag

	exit(EXIT_SUCCESS);
} // end main


/////////////////////////////////////////////////////////////////////////////////////
// Sets ICs from file:
/////////////////////////////////////////////////////////////////////////////////////

ballGroup importDataFromFile(std::string initDataFileName, std::string initConstFileName)
{
	ballGroup tclus;

	// Get position and angular velocity data:
	if (auto simDataStream = std::ifstream(initDataFileName, std::ifstream::in))
	{
		std::string line, lineElement;
		std::cout << "\nParsing last line of data.\n";

		simDataStream.seekg(-1, std::ios_base::end); // go to one spot before the EOF

		bool keepLooping = true;
		while (keepLooping)
		{
			char ch;
			simDataStream.get(ch); // Get current byte's data

			if ((int)simDataStream.tellg() <= 1)
			{                           // If the data was at or before the 0th byte
				simDataStream.seekg(0); // The first line is the last line
				keepLooping = false;    // So stop there
			}
			else if (ch == '\n')
			{                        // If the data was a newline
				keepLooping = false; // Stop at the current position.
			}
			else
			{                                                // If the data was neither a newline nor at the 0 byte
				simDataStream.seekg(-2, std::ios_base::cur); // Move to the front of that data, then to the front of the data before it
			}
		}


		std::getline(simDataStream, line);                                              // Read the current line
		int count = std::count(line.begin(), line.end(), ',') / properties + 1;
		tclus.allocateGroup(count); // Get number of balls in file

		std::stringstream chosenLine(line); // This is the last line of the read file, containing all data for all balls at last time step

		for (int A = 0; A < tclus.cNumBalls; A++)
		{

			for (int i = 0; i < 3; i++) // Position
			{
				std::getline(chosenLine, lineElement, ',');
				tclus.pos[A][i] = std::stod(lineElement);
				//std::cout << tclus.pos[A][i]<<',';
			}
			for (int i = 0; i < 3; i++) // Angular Velocity
			{
				std::getline(chosenLine, lineElement, ',');
				tclus.w[A][i] = std::stod(lineElement);
			}
			std::getline(chosenLine, lineElement, ','); // Angular velocity magnitude skipped
			for (int i = 0; i < 3; i++)                 // velocity
			{
				std::getline(chosenLine, lineElement, ',');
				tclus.vel[A][i] = std::stod(lineElement);
			}
			for (int i = 0; i < properties - 10; i++) // We used 10 elements. This skips the rest.
			{
				std::getline(chosenLine, lineElement, ',');
			}
		}
	}
	else
	{
		std::cerr << "Could not open simData file: " << initDataFileName << "... Existing program." << std::endl;
		exit(EXIT_FAILURE);
	}

	// Get radius, mass, moi:
	if (auto ConstStream = std::ifstream(initConstFileName, std::ifstream::in))
	{
		std::string line, lineElement;
		for (int A = 0; A < tclus.cNumBalls; A++)
		{
			std::getline(ConstStream, line); // Ball line.
			std::stringstream chosenLine(line);
			std::getline(chosenLine, lineElement, ','); // Radius.
			tclus.R[A] = std::stod(lineElement);
			std::getline(chosenLine, lineElement, ','); // Mass.
			tclus.m[A] = std::stod(lineElement);
			std::getline(chosenLine, lineElement, ','); // Moment of inertia.
			tclus.moi[A] = std::stod(lineElement);
		}
	}
	else
	{
		std::cerr << "Could not open constants file: " << initConstFileName << "... Existing program." << std::endl;
		exit(EXIT_FAILURE);
	}

	// Bring cluster to origin and calc its radius:
	tclus.toOrigin();
	tclus.updateRadius();

	std::cout << "Balls: " << tclus.cNumBalls << std::endl;
	std::cout << "Mass: " << tclus.mTotal << std::endl;
	std::cout << "Approximate radius: " << tclus.radius << " cm.\n";
	return tclus;
}

void twoSizeSphereShell5000()
{

	for (int Ball = 0; Ball < 1000; Ball++)
	{
		O.R[Ball] = 700;
		O.m[Ball] = density * 4. / 3. * 3.14159 * pow(O.R[Ball], 3);
		O.moi[Ball] = .4 * O.m[Ball] * O.R[Ball] * O.R[Ball];
		O.w[Ball] = { 0, 0, 0 };
		O.pos[Ball] = randShellVec(spaceRange, O.radius);
	}

	for (int Ball = 1000; Ball < 2000; Ball++)
	{
		O.R[Ball] = 400;
		O.m[Ball] = density * 4. / 3. * 3.14159 * pow(O.R[Ball], 3);
		O.moi[Ball] = .4 * O.m[Ball] * O.R[Ball] * O.R[Ball];
		O.w[Ball] = { 0, 0, 0 };
		O.pos[Ball] = randShellVec(spaceRange, O.radius);
	}

	int ballsInPhase1 = 2000;
	std::cout << "Balls in phase: " << ballsInPhase1 << "\n";

	// Generate non-overlapping spherical particle field:
	int collisionDetected = 0;
	int oldCollisions = 1e10;

	for (int failed = 0; failed < attempts; failed++)
	{
		for (int A = 0; A < ballsInPhase1; A++)
		{
			for (int B = A + 1; B < ballsInPhase1; B++)
			{
				// Check for Ball overlap.
				double dist = (O.pos[A] - O.pos[B]).norm();
				double sumRaRb = O.R[A] + O.R[B];
				double overlap = dist - sumRaRb;
				if (overlap < 0)
				{
					collisionDetected += 1;
					// Move the other ball:
					O.pos[B] = randShellVec(spaceRange, O.radius);
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
		if (failed == attempts - 1 || collisionDetected > int(1.5 * (double)ballsInPhase1)) // Added the second part to speed up spatial constraint increase when there are clearly too many collisions for the space to be feasable.
		{
			std::cout << "Failed " << spaceRange << ". Increasing range " << spaceRangeIncrement << "cm^3.\n";
			spaceRange += spaceRangeIncrement;
			failed = 0;
			for (int Ball = 0; Ball < ballsInPhase1; Ball++)
			{
				O.pos[Ball] = randShellVec(spaceRange, O.radius); // Each time we fail and increase range, redistribute all balls randomly so we don't end up with big balls near mid and small balls outside.
			}
		}
		collisionDetected = 0;
	}

	// Calculate cluster radius:
	vector3d comNumerator;
	for (int Ball = 0; Ball < O.cNumBalls; Ball++)
	{
		O.mTotal += O.m[Ball];
		comNumerator += O.m[Ball] * O.pos[Ball];
	}
	O.com = comNumerator / O.mTotal;

	O.updateRadius();

	spaceRange += 2 * O.R[0] + 4 * 250;
	O.radius += O.R[0] + 250;
	std::cout << "Making shell between " << O.radius << " and " << spaceRange * .5 << std::endl;

	// PHASE 2

	for (int Ball = 2000; Ball < 3500; Ball++)
	{
		O.R[Ball] = 250;
		O.m[Ball] = density * 4. / 3. * 3.14159 * pow(O.R[Ball], 3);
		O.moi[Ball] = .4 * O.m[Ball] * O.R[Ball] * O.R[Ball];
		O.w[Ball] = { 0, 0, 0 };
		O.pos[Ball] = randShellVec(spaceRange, O.radius);
	}

	for (int Ball = 3500; Ball < 5000; Ball++)
	{
		O.R[Ball] = 150;
		O.m[Ball] = density * 4. / 3. * 3.14159 * pow(O.R[Ball], 3);
		O.moi[Ball] = .4 * O.m[Ball] * O.R[Ball] * O.R[Ball];
		O.w[Ball] = { 0, 0, 0 };
		O.pos[Ball] = randShellVec(spaceRange, O.radius);
	}

	int ballsInPhase2 = 3000;
	std::cout << "Balls in phase: " << ballsInPhase2 << "\n";

	// Generate non-overlapping spherical particle field:
	collisionDetected = 0;
	oldCollisions = 100000000;

	for (int failed = 0; failed < attempts; failed++)
	{
		for (int A = ballsInPhase1; A < ballsInPhase1 + ballsInPhase2; A++)
		{
			for (int B = A + 1; B < ballsInPhase1 + ballsInPhase2; B++)
			{
				// Check for Ball overlap.
				double dist = (O.pos[A] - O.pos[B]).norm();
				double sumRaRb = O.R[A] + O.R[B];
				double overlap = dist - sumRaRb;
				if (overlap < 0)
				{
					collisionDetected += 1;
					// Move the other ball:
					O.pos[B] = randShellVec(spaceRange, O.radius);
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
		if (failed == attempts - 1 || collisionDetected > int(1.5 * (double)ballsInPhase2)) // Added the second part to speed up spatial constraint increase when there are clearly too many collisions for the space to be feasable.
		{
			std::cout << "Failed " << spaceRange << ". Increasing range " << spaceRangeIncrement << "cm^3.\n";
			spaceRange += spaceRangeIncrement;
			failed = 0;
			for (int Ball = ballsInPhase1; Ball < ballsInPhase1 + ballsInPhase2; Ball++)
			{
				O.pos[Ball] = randShellVec(spaceRange, O.radius); // Each time we fail and increase range, redistribute all balls randomly so we don't end up with big balls near mid and small balls outside.
			}
		}
		collisionDetected = 0;
	}


	std::cout << "Initial Radius: " << O.radius << std::endl;
	std::cout << "Mass: " << O.mTotal << std::endl;

}

void testGen()
{
	// Make genBalls of 3 sizes in CGS with ratios such that the mass is distributed evenly among the 3 sizes (less large genBalls than small genBalls).
	int smalls = std::round((double)genBalls * 27 / 31.375); // Just here for reference. Whatever genBalls are left will be smalls.
	int mediums = std::round((double)genBalls * 27 / (8 * 31.375));
	int larges = std::round((double)genBalls * 1 / 31.375);


	for (int Ball = 0; Ball < larges; Ball++)
	{
		O.R[Ball] = 3. * scaleBalls;//pow(1. / (double)genBalls, 1. / 3.) * 3. * scaleBalls;
		O.m[Ball] = density * 4. / 3. * 3.14159 * pow(O.R[Ball], 3);
		O.moi[Ball] = .4 * O.m[Ball] * O.R[Ball] * O.R[Ball];
		O.w[Ball] = { 0, 0, 0 };
		O.pos[Ball] = randSphericalVec(spaceRange, spaceRange, spaceRange);
	}

	for (int Ball = larges; Ball < (larges + mediums); Ball++)
	{
		O.R[Ball] = 2. * scaleBalls;//pow(1. / (double)genBalls, 1. / 3.) * 2. * scaleBalls;
		O.m[Ball] = density * 4. / 3. * 3.14159 * pow(O.R[Ball], 3);
		O.moi[Ball] = .4 * O.m[Ball] * O.R[Ball] * O.R[Ball];
		O.w[Ball] = { 0, 0, 0 };
		O.pos[Ball] = randSphericalVec(spaceRange, spaceRange, spaceRange);
	}
	for (int Ball = (larges + mediums); Ball < genBalls; Ball++)
	{
		O.R[Ball] = 1. * scaleBalls;//pow(1. / (double)genBalls, 1. / 3.) * 1. * scaleBalls;
		O.m[Ball] = density * 4. / 3. * 3.14159 * pow(O.R[Ball], 3);
		O.moi[Ball] = .4 * O.m[Ball] * O.R[Ball] * O.R[Ball];
		O.w[Ball] = { 0, 0, 0 };
		O.pos[Ball] = randSphericalVec(spaceRange, spaceRange, spaceRange);
	}

	//simInitWrite();

	std::cout << "Smalls: " << smalls << " Mediums: " << mediums << " Larges: " << larges << std::endl;

	std::cout << "Final spacerange: " << spaceRange << std::endl;
	// Calculate approximate radius of imported cluster and center of mass:
	vector3d comNumerator;
	for (int Ball = 0; Ball < O.cNumBalls; Ball++)
	{
		O.mTotal += O.m[Ball];
		comNumerator += O.m[Ball] * O.pos[Ball];
	}
	O.com = comNumerator / O.mTotal;

	O.updateRadius();

	std::cout << "Initial Radius: " << O.radius << std::endl;
	std::cout << "Mass: " << O.mTotal << std::endl;
}

void threeSizeSphere()
{
	// Make genBalls of 3 sizes in CGS with ratios such that the mass is distributed evenly among the 3 sizes (less large genBalls than small genBalls).
	int smalls = std::round((double)genBalls * 27 / 31.375); // Just here for reference. Whatever genBalls are left will be smalls.
	int mediums = std::round((double)genBalls * 27 / (8 * 31.375));
	int larges = std::round((double)genBalls * 1 / 31.375);


	for (int Ball = 0; Ball < larges; Ball++)
	{
		O.R[Ball] = 3. * scaleBalls;//pow(1. / (double)genBalls, 1. / 3.) * 3. * scaleBalls;
		O.m[Ball] = density * 4. / 3. * 3.14159 * pow(O.R[Ball], 3);
		O.moi[Ball] = .4 * O.m[Ball] * O.R[Ball] * O.R[Ball];
		O.w[Ball] = { 0, 0, 0 };
		O.pos[Ball] = randSphericalVec(spaceRange, spaceRange, spaceRange);
	}

	for (int Ball = larges; Ball < (larges + mediums); Ball++)
	{
		O.R[Ball] = 2. * scaleBalls;//pow(1. / (double)genBalls, 1. / 3.) * 2. * scaleBalls;
		O.m[Ball] = density * 4. / 3. * 3.14159 * pow(O.R[Ball], 3);
		O.moi[Ball] = .4 * O.m[Ball] * O.R[Ball] * O.R[Ball];
		O.w[Ball] = { 0, 0, 0 };
		O.pos[Ball] = randSphericalVec(spaceRange, spaceRange, spaceRange);
	}
	for (int Ball = (larges + mediums); Ball < genBalls; Ball++)
	{
		O.R[Ball] = 1. * scaleBalls;//pow(1. / (double)genBalls, 1. / 3.) * 1. * scaleBalls;
		O.m[Ball] = density * 4. / 3. * 3.14159 * pow(O.R[Ball], 3);
		O.moi[Ball] = .4 * O.m[Ball] * O.R[Ball] * O.R[Ball];
		O.w[Ball] = { 0, 0, 0 };
		O.pos[Ball] = randSphericalVec(spaceRange, spaceRange, spaceRange);
	}

	std::cout << "Smalls: " << smalls << " Mediums: " << mediums << " Larges: " << larges << std::endl;

	// Generate non-overlapping spherical particle field:
	int collisionDetected = 0;
	int oldCollisions = genBalls;

	for (int failed = 0; failed < attempts; failed++)
	{
		for (int A = 0; A < genBalls; A++)
		{
			for (int B = A + 1; B < genBalls; B++)
			{
				// Check for Ball overlap.
				double dist = (O.pos[A] - O.pos[B]).norm();
				double sumRaRb = O.R[A] + O.R[B];
				double overlap = dist - sumRaRb;
				if (overlap < 0)
				{
					collisionDetected += 1;
					// Move the other ball:
					O.pos[B] = randSphericalVec(spaceRange, spaceRange, spaceRange);
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
				O.pos[Ball] = randSphericalVec(spaceRange, spaceRange, spaceRange); // Each time we fail and increase range, redistribute all balls randomly so we don't end up with big balls near mid and small balls outside.
			}
		}
		collisionDetected = 0;
	}

	std::cout << "Final spacerange: " << spaceRange << std::endl;
	// Calculate approximate radius of imported cluster and center of mass:
	vector3d comNumerator;
	for (int Ball = 0; Ball < O.cNumBalls; Ball++)
	{
		O.mTotal += O.m[Ball];
		comNumerator += O.m[Ball] * O.pos[Ball];
	}
	O.com = comNumerator / O.mTotal;

	O.updateRadius();

	std::cout << "Initial Radius: " << O.radius << std::endl;
	std::cout << "Mass: " << O.mTotal << std::endl;
}



void oneSizeSphere()
{

	for (int Ball = 0; Ball < genBalls; Ball++)
	{
		O.R[Ball] = scaleBalls;
		O.m[Ball] = density * 4. / 3. * 3.14159 * pow(O.R[Ball], 3);
		O.moi[Ball] = .4 * O.m[Ball] * O.R[Ball] * O.R[Ball];
		O.w[Ball] = { 0, 0, 0 };
		O.pos[Ball] = randSphericalVec(spaceRange, spaceRange, spaceRange);
	}

	// Generate non-overlapping spherical particle field:
	int collisionDetected = 0;
	int oldCollisions = genBalls;

	for (int failed = 0; failed < attempts; failed++)
	{
		for (int A = 0; A < genBalls; A++)
		{
			for (int B = A + 1; B < genBalls; B++)
			{
				// Check for Ball overlap.
				double dist = (O.pos[A] - O.pos[B]).norm();
				double sumRaRb = O.R[A] + O.R[B];
				double overlap = dist - sumRaRb;
				if (overlap < 0)
				{
					collisionDetected += 1;
					// Move the other ball:
					O.pos[B] = randSphericalVec(spaceRange, spaceRange, spaceRange);
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
				O.pos[Ball] = randSphericalVec(spaceRange, spaceRange, spaceRange); // Each time we fail and increase range, redistribute all balls randomly so we don't end up with big balls near mid and small balls outside.
			}
		}
		collisionDetected = 0;
	}

	std::cout << "Final spacerange: " << spaceRange << std::endl;
	// Calculate approximate radius of imported cluster and center of mass:
	vector3d comNumerator;
	for (int Ball = 0; Ball < O.cNumBalls; Ball++)
	{
		O.mTotal += O.m[Ball];
		comNumerator += O.m[Ball] * O.pos[Ball];
	}
	O.com = comNumerator / O.mTotal;

	O.updateRadius();

	std::cout << "Initial Radius: " << O.radius << std::endl;
	std::cout << "Mass: " << O.mTotal << std::endl;
}



void generateBallField()
{
	std::cout << "CLUSTER FORMATION\n";
	O.allocateGroup(genBalls);


	// Create new random number set.
	int seedSave = time(NULL);
	srand(seedSave);

	//twoSizeSphereShell5000();
	//oneSizeSphere();
	threeSizeSphere();
	//testGen();

	outputPrefix =
		std::to_string(genBalls) +
		"-R" + scientific(O.radius) +
		"-cor" + rounder(pow(cor, 2), 4) +
		"-mu" + rounder(mu, 3) +
		"-rho" + rounder(density, 4);
}



void safetyChecks()
{
	if (kin < 0)
	{
		printf("\nSPRING CONSTANT NOT SET\n");
		exit(EXIT_FAILURE);
	}

	if (dt <= 0)
	{
		printf("\nDT NOT SET\n");
		exit(EXIT_FAILURE);
	}

	if (steps < 0)
	{
		printf("\nSTEPS NOT SET\n");
		exit(EXIT_FAILURE);
	}
}


void calibrateDT(const int Step, const bool superSafe, bool doK)
{
	double dtOld = dt;

	// Calculate max velocity due to collapse
	O.updateRadius();
	O.updateComAndMass(); // Need to know mass for vCollapse calculation.
	double vCollapse = sqrt(2 * G * O.mTotal / O.radius);

	soc = 2 * O.radius; // sphere of consideration for max velocity, to avoid very unbound high vel balls.

	double vMax = O.getVelMax(false);

	// Check if the kick is going to be the most significant velocity basis, or if gravity will matter more.
	std::cout << std::endl;
	if (vMax > fabs(vCollapse))
	{
		std::cout << "vMax > binding. " << vCollapse << "<vCollapse | vMax>" << vMax;

		if (superSafe)
		{
			// Safe: dt based on fastest velocity
			setLazzDT(vMax);
			std::cout << " dt Calibrated: " << dt;
		}
		else
		{
			// Less safe: dt based on fastest velocity
			setGuidDT(vMax);
			std::cout << " dt Calibrated: " << dt;
		}

		if (doK)
		{
			if (superSafe)
			{
				// Safe: K based on fastest velocity
				setLazzK(vMax);
				std::cout << " K Calibrated: " << kin;
			}
			else
			{
				// Less safe: K based on fastest velocity
				setGuidK(vMax);
				std::cout << " K Calibrated: " << kin;
			}
		}
	}
	else
	{
		std::cout << "Binding > vMax. " << vCollapse << "<vCollapse | vMax>" << vMax;

		if (superSafe)
		{
			// Safe: dt based on fastest velocity
			setLazzDT(vCollapse);
			std::cout << " dt Calibrated: " << dt;
		}
		else
		{
			// Less safe: dt based on fastest velocity
			setGuidDT(vCollapse);
			std::cout << " dt Calibrated: " << dt;
		}

		if (doK)
		{
			if (superSafe)
			{
				// Safe: K based on fastest velocity
				setLazzK(vCollapse);
				std::cout << " K Calibrated: " << kin;
			}
			else
			{
				// Less safe: K based on fastest velocity
				setGuidK(vCollapse);
				std::cout << " K Calibrated: " << kin;
			}
		}
	}

	if (Step == 0)
	{
		steps = (size_t)(simTimeSeconds / dt);
		std::cout << " Step count: " << steps << std::endl;
	}
	else
	{
		steps = dt / dtOld * (steps - Step) + Step;
		std::cout << " New step count: " << steps << std::endl;
	}
}

void setGuidDT(double vel)
{
	// Guidos k and dt:
	dt = .01 * O.getRmin() / fabs(vel);
}

void setGuidK(double vel)
{
	kin = O.getMassMax() * vel * vel / (.1 * O.R[0] * .1 * O.R[0]);
	kout = cor * kin;
}

void setLazzDT(double vel)
{
	// Lazzati k and dt:
	double kTemp = 4 / 3 * M_PI * density * O.getMassMax() * vel * vel / (.1 * .1);
	dt = .01 * sqrt(4 / 3 * M_PI * density / kTemp * O.getRmin());
}

void setLazzK(double vel)
{
	kin = 4 / 3 * M_PI * density * O.getMassMax() * vel * vel / (.1 * .1);
	kout = cor * kin;
}

void simDataWrite()
{
	ballBuffer << std::endl; // Prepares a new line for incoming data.

	for (size_t Ball = 0; Ball < O.cNumBalls; Ball++)
	{
		// Send positions and rotations to buffer:
		if (Ball == 0)
		{
			ballBuffer << O.pos[Ball][0] << ',' << O.pos[Ball][1] << ',' << O.pos[Ball][2] << ',' << O.w[Ball][0] << ',' << O.w[Ball][1] << ',' << O.w[Ball][2] << ',' << O.w[Ball].norm() << ',' << O.vel[Ball].x << ',' << O.vel[Ball].y << ',' << O.vel[Ball].z << ',' << 0;
		}
		else
		{
			ballBuffer << ',' << O.pos[Ball][0] << ',' << O.pos[Ball][1] << ',' << O.pos[Ball][2] << ',' << O.w[Ball][0] << ',' << O.w[Ball][1] << ',' << O.w[Ball][2] << ',' << O.w[Ball].norm() << ',' << O.vel[Ball].x << ',' << O.vel[Ball].y << ',' << O.vel[Ball].z << ',' << 0;
		}
	}

	// Write simData to file and clear buffer.
	ballWrite.open(simDataName, myOpenMode);
	ballWrite << ballBuffer.rdbuf(); // Barf buffer to file.
	ballBuffer.str("");              // Resets the stream for that balls to blank.
	ballWrite.close();
}
