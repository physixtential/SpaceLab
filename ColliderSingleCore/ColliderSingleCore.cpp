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

// Function Prototypes
int countBalls(std::string initDataFileName);
cluster initFromFile(std::string initDataFileName, std::string initConstFileName, bool zeroMotion);

// Main function
int main(int argc, char const* argv[])
{
	// Runtime arguments:
	double spins[3] = { 0 };
	if (argc > 1)
	{
		//spins[0] = atof(argv[1]);
		//spins[1] = atof(argv[2]);
		//spins[2] = atof(argv[3]);
		//printf("Spin: %.2e %.2e %.2e\n", spins[0], spins[1], spins[2]);
		numThreads = atoi(argv[1]);
		printf("\nThread count set to %i.\n", numThreads);
		//y0Rot = atof(argv[5]);
		//z0Rot = atof(argv[6]);
		//printf("Rotate y and z: %1.3f\t%1.3f\n", y0Rot, z0Rot);
		//z1Rot = atof(argv[7]);
		//y1Rot = atof(argv[8]);
		clusterAName = argv[2];
		clusterBName = argv[3];
		KEfactor = atoi(argv[4]);
	}

	universe cosmos;

	// Two cluster sim:
	if (true)
	{
		// Count balls in files, reserve space, then load file data:
		int count = 0;
		std::cerr << "File 1: " << clusterAName << '\t' << "File 2: " << clusterBName << std::endl;
		count += countBalls(path + clusterAName + "simData.csv");
		count += countBalls(path + clusterBName + "simData.csv");
		cluster clusA = initFromFile(path + clusterAName + "simData.csv", path + clusterAName + "constants.csv", 0);
		cluster clusB = initFromFile(path + clusterBName + "simData.csv", path + clusterBName + "constants.csv", 0);

		clusA.offset(clusA.radius, clusB.radius + (clusA.balls[0].R * 1.), impactParameter); // Adding 3 times the radius of one ball gaurantees total separation between clusters.
		double PEsys = clusA.PE + clusB.PE + (-G * clusA.m * clusB.m / (clusA.com - clusB.com).norm());
		// Collision velocity calculation:
		double mSmall = clusA.m;
		double mBig = clusB.m;
		double mTot = mBig + mSmall;
		double vSmall = -sqrt(2 * KEfactor * fabs(PEsys) * (mBig / (mSmall * mTot))); // Negative because small offsets right.
		double vBig = -(mSmall / mBig) * vSmall; // Negative to be opposing projectile.
		fprintf(stdout, "Target Velocity: %.2e\nProjectile Velocity: %.2e\n", vBig, vSmall);
		if (isnan(vSmall) || isnan(vBig))
		{
			fprintf(stderr, "A VELOCITY WAS NAN!!!!!!!!!!!!!!!!!!!!!!\n\n");
			exit(EXIT_FAILURE);
		}
		clusA.kick(vSmall);
		clusB.kick(vBig);
		clusA.checkMomentum();
		clusB.checkMomentum();
		cosmos.balls.insert(cosmos.balls.end(), clusA.balls.begin(), clusA.balls.end());
		cosmos.balls.insert(cosmos.balls.end(), clusB.balls.begin(), clusB.balls.end());
	}

	// Future multicluster sim:
	for (std::string i : { "file1", "file2", "file3..." })
	{
		;// std::cout << i;
	}

	// One cluster sim:
	if (false)
	{
		// Count balls in files, reserve space, then load file data:
		int count = 0;
		count += countBalls(clusterAName + "simData.csv");
		cluster clusA = initFromFile(clusterAName + "simData.csv", clusterAName + "constants.csv", 0);
		// Rotate
		clusA.rotAll('z', z0Rot);
		clusA.rotAll('y', y0Rot);
		// Spin
		clusA.comSpinner(spins[0], spins[1], spins[2]);
		// Check and add to universe
		clusA.checkMomentum();
		cosmos.balls.insert(cosmos.balls.end(), clusA.balls.begin(), clusA.balls.end());
	}

	// Cosmos has been filled with balls. Size is known:
	int ballTotal = (int)cosmos.numBalls;
	std::vector<ball>& all = cosmos.balls;
	cosmos.checkMomentum(); // Is total momentum zero like it should be?

	cosmos.calcComAndMass();
	// Re-center universe mass to origin:
	for (int Ball = 0; Ball < ballTotal; Ball++)
	{
		cosmos.balls[Ball].pos -= cosmos.com;
	}
	// Compute physics between all balls. Distances, collision forces, energy totals, total mass:
	cosmos.initConditions();

	// Create string for file name identifying spin combination negative is 2, positive is 1 on each axis.
	std::string spinCombo = "";
	for (int i = 0; i < 3; i++)
	{
		if (spins[i] < 0) { spinCombo += "2"; }
		else if (spins[i] > 0) { spinCombo += "1"; }
		else { spinCombo += "0"; }
	}

	outputPrefix =
		std::to_string(ballTotal) +
		clusterAName + clusterBName +
		"-T" + rounder(KEfactor, 3) +
		"-IP" + rounder(impactParameter * 180 / 3.14159, 2) +
		"spin" + spinCombo +
		"-k" + scientific(kin) +
		"-cor" + rounder(pow(cor, 2), 4) +
		"-rho" + rounder(density, 4) +
		"-dt" + rounder(dt, 4) +
		"_";

	// Save file names:
	std::string simDataName = outputPrefix + "simData.csv",
		constantsName = outputPrefix + "constants.csv",
		energyName = outputPrefix + "energy.csv";

	std::ofstream::openmode myOpenMode = std::ofstream::app;

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
			<< ",bound" + thisBall;
	}

	std::cout << "\nSim data, energy, and constants file streams and headers created.";

	// Write constant data:
	for (int Ball = 0; Ball < ballTotal; Ball++)
	{

		constWrite
			<< all[Ball].R << ','
			<< all[Ball].m << ','
			<< all[Ball].moi
			<< std::endl;
	}

	// Write energy data to buffer:
	energyBuffer
		<< std::endl
		<< dt << ','
		<< cosmos.PE << ','
		<< cosmos.KE << ','
		<< cosmos.PE + cosmos.KE << ','
		<< cosmos.momentum.norm() << ','
		<< cosmos.angularMomentum.norm() << ','
		<< 0 << ',' //boundMass
		<< 0 << ',' //unboundMass
		<< cosmos.mTotal;
	energyWrite << energyBuffer.rdbuf();
	energyBuffer.str("");

	// Reinitialize energies for next step:
	cosmos.KE = 0;
	cosmos.PE = 0;
	cosmos.momentum = { 0, 0, 0 };
	cosmos.angularMomentum = { 0, 0, 0 };

	// Send position and rotation to buffer:
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
			<< ',' << all[Ball].pos.x << ',' // Needs comma start so the last bound doesn't have a dangling comma.
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
	std::cout << "Total mass: " << cosmos.mTotal << std::endl;
	std::cout << "\n===============================================================\n";

	//////////////////////////////////////////////////////////
	// Loop Start ///////////////////////////////////////////
	////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	//////////////////////////////////////////////////////
	/////////////////////////////////////////////////////
	time_t start = time(NULL);         // For end of program analysis
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
			float eta = ((time(NULL) - startProgress) / 500.0 * (steps - Step)) / 3600.; // In seconds.
			sizeof(int);
			float elapsed = (time(NULL) - start) / 3600.;
			float progress = ((float)Step / (float)steps * 100.f);
			printf("Step: %i\tProgress: %2.0f%%\tETA: %5.2lf\tElapsed: %5.2f\n", Step, progress, eta, elapsed);
			startProgress = time(NULL);
		}
		else
		{
			writeStep = false;
		}

		// FIRST PASS - Position, send to buffer, velocity half step:
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
		for (int A = 0; A < ballTotal; A++)
		{
			//#pragma omp parallel for shared(all,writeStep,A,ballTotal,dt,a) reduction(+:PEchange)
			for (int B = 0; B < ballTotal; B++)
			{
				if (B < A + 1)
				{
					continue;
				}
				double k;

				ball& a = all[A]; // THIS IS BAD. But necessary because collapse doesn't like
				ball& b = all[B];
				double sumRaRb = a.R + b.R;
				double dist = (a.pos - b.pos).norm();
				vector3d rVecab = b.pos - a.pos;
				vector3d rVecba = a.pos - b.pos;

				// Check for collision between Ball and otherBall:
				double overlap = sumRaRb - dist;
				vector3d totalForce = { 0, 0, 0 };
				vector3d aTorque = { 0, 0, 0 };
				vector3d bTorque = { 0, 0, 0 };

				// Check for collision between Ball and otherBall.
				if (overlap > 0)
				{
					// Apply coefficient of restitution to balls leaving collision.
					if (dist >= a.distances[B]) // <<< huge balls x balls array
					{
						k = kout;
						if (springTest)
						{
							if (a.distances[B] < 0.9 * a.R || a.distances[B] < 0.9 * b.R)
							{
								if (a.R >= b.R)
								{
									std::cout << "Warning: Ball compression is " << .5 * (sumRaRb - a.distances[B]) / b.R << " of radius = " << b.R << std::endl;
								}
								else
								{
									std::cout << "Warning: Ball compression is " << .5 * (sumRaRb - a.distances[B]) / a.R << " of radius = " << a.R << std::endl;
								}
								//int garbo;
								//std::cin >> garbo;
							}
						}
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

					if (writeStep)
					{
						// Calculate potential energy. Important to recognize that the factor of 1/2 is not in front of K because this is for the spring potential in each ball and they are the same potential.
						cosmos.PE += -G * all[A].m * all[B].m / dist + k * pow((all[A].R + all[B].R - dist) * .5, 2);
					}
				}
				else
				{
					// No collision: Include gravity only:
					vector3d gravForceOnA = (G * a.m * b.m / pow(dist, 2)) * (rVecab / dist);
					totalForce = gravForceOnA;
					if (writeStep)
					{
						cosmos.PE += -G * all[A].m * all[B].m / dist;
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
		if (writeStep)
		{
			ballBuffer << std::endl; // Prepares a new line for incoming data.
		}
		for (int Ball = 0; Ball < ballTotal; Ball++)
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
					ballBuffer << a.pos[0] << ',' << a.pos[1] << ',' << a.pos[2] << ',' << a.w[0] << ',' << a.w[1] << ',' << a.w[2] << ',' << a.w.norm() << ',' << a.vel[0] << ',' << a.vel[1] << ',' << a.vel[2] << ',' << 0; //bound[0];
				}
				else
				{
					ballBuffer << ',' << a.pos[0] << ',' << a.pos[1] << ',' << a.pos[2] << ',' << a.w[0] << ',' << a.w[1] << ',' << a.w[2] << ',' << a.w.norm() << ',' << a.vel[0] << ',' << a.vel[1] << ',' << a.vel[2] << ',' << 0; //bound[Ball];
				}

				cosmos.KE += .5 * a.m * a.vel.normsquared() + .5 * a.moi * a.w.normsquared(); // Now includes rotational kinetic energy.
				cosmos.momentum += a.m * a.vel;
				cosmos.angularMomentum += a.m * a.pos.cross(a.vel) + a.moi * a.w;
			}
		}
		if (writeStep)
		{
			// Write energy to stream:
			energyBuffer << std::endl
				<< dt * Step << ',' << cosmos.PE << ',' << cosmos.KE << ',' << cosmos.PE + cosmos.KE << ',' << cosmos.momentum.norm() << ',' << cosmos.angularMomentum.norm() << ',' << 0 << ',' << 0 << ',' << cosmos.mTotal; // the two zeros are bound and unbound mass

			// Reinitialize energies for next step:
			cosmos.KE = 0;
			cosmos.PE = 0;
			cosmos.momentum = { 0, 0, 0 };
			cosmos.angularMomentum = { 0, 0, 0 };
			// unboundMass = 0;
			// boundMass = cosmos.mTotal;

			////////////////////////////////////////////////////////////////////
			// Data Export /////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////
			if (time(NULL) - lastWrite > 1800 || Step / skip % 20 == 0 || Step == steps - 1)
			{
				std::cout << "\nData Write" << std::endl;

				// Write simData to file and clear buffer.
				ballWrite.open(simDataName, myOpenMode);
				ballWrite << ballBuffer.rdbuf(); // Barf buffer to file.
				ballBuffer.str("");              // Resets the stream for that balls to blank.
				ballWrite.close();

				// Write Energy data to file and clear buffer.
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
	exit(EXIT_SUCCESS);
} // end main



/////////////////////////////////////////////////////////////////////////////////////
// Sets ICs from file:
/////////////////////////////////////////////////////////////////////////////////////
int countBalls(std::string initDataFileName)
{
	// Get position and angular velocity data:
	std::ifstream initDataStream;
	std::string line, lineElement;
	initDataStream.open(initDataFileName, std::ifstream::in);
	if (initDataStream.is_open())
	{
		initDataStream.seekg(-1, std::ios_base::end); // go to one spot before the EOF

		bool keepLooping = true;
		while (keepLooping)
		{
			char ch;
			initDataStream.get(ch); // Get current byte's data

			if ((int)initDataStream.tellg() <= 1)
			{                            // If the data was at or before the 0th byte
				initDataStream.seekg(0); // The first line is the last line
				keepLooping = false;     // So stop there
			}
			else if (ch == '\n')
			{                        // If the data was a newline
				keepLooping = false; // Stop at the current position.
			}
			else
			{                                                 // If the data was neither a newline nor at the 0 byte
				initDataStream.seekg(-2, std::ios_base::cur); // Move to the front of that data, then to the front of the data before it
			}
		}

		std::getline(initDataStream, line); // Read the current line
	}
	else
	{
		std::cout << "File not found.\n";
		std::string garbo;
		std::cin >> garbo;
	}
	////////////////////////////////////////////////////
	//////////// check if we can use this line to count them cleaner. maybe has to do with error in mass and radius calc in first cluster
	//////////////////////////////////////
	int ballsInFile = std::count(line.begin(), line.end(), ',') / properties + 1; // Get number of balls in file
	return ballsInFile;
}

cluster initFromFile(std::string initDataFileName, std::string initConstFileName, bool zeroMotion)
{
	cluster tclus;
	// Get position and angular velocity data:
	if (auto simDataStream = std::ifstream(initDataFileName, std::ifstream::in))
	{
		std::string line, lineElement;
		std::cout << "Parsing last line of data.\n";

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
		tclus.balls.resize(std::count(line.begin(), line.end(), ',') / properties + 1); // Get number of balls in file

		std::stringstream chosenLine(line); // This is the last line of the read file, containing all data for all balls at last time step

		for (int A = 0; A < tclus.numBalls; A++)
		{
			ball& a = tclus.balls[A];

			for (int i = 0; i < 3; i++) // Position
			{
				std::getline(chosenLine, lineElement, ',');
				a.pos[i] = std::stod(lineElement);
				//std::cout << a->pos[i]<<',';
			}
			for (int i = 0; i < 3; i++) // Angular Velocity
			{
				std::getline(chosenLine, lineElement, ',');
				a.w[i] = std::stod(lineElement);
			}
			std::getline(chosenLine, lineElement, ','); // Angular velocity magnitude skipped
			for (int i = 0; i < 3; i++)                 // velocity
			{
				std::getline(chosenLine, lineElement, ',');
				a.vel[i] = std::stod(lineElement);
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
		for (int A = 0; A < tclus.numBalls; A++)
		{
			ball& a = tclus.balls[A];
			std::getline(ConstStream, line); // Ball line.
			std::stringstream chosenLine(line);
			std::getline(chosenLine, lineElement, ','); // Radius.
			a.R = std::stod(lineElement);
			std::getline(chosenLine, lineElement, ','); // Mass.
			a.m = std::stod(lineElement);
			std::getline(chosenLine, lineElement, ','); // Moment of inertia.
			a.moi = std::stod(lineElement);
		}
	}
	else
	{
		std::cerr << "Could not open constants file: " << initConstFileName << "... Existing program." << std::endl;
		exit(EXIT_FAILURE);
	}
	// Zero all angular momenta and velocity:
	if (zeroMotion)
	{
		for (int Ball = 0; Ball < tclus.numBalls; Ball++)
		{
			tclus.balls[Ball].w = { 0, 0, 0 };
			tclus.balls[Ball].vel = { 0, 0, 0 };
		}
	}

	// Calculate approximate radius of imported cluster and center mass at origin:
	vector3d comNumerator;
	for (int Ball = 0; Ball < tclus.numBalls; Ball++)
	{
		ball& a = tclus.balls[Ball];
		tclus.m += a.m;
		comNumerator += a.m * a.pos;
	}
	tclus.com = comNumerator / tclus.m;

	for (int Ball = 0; Ball < tclus.numBalls; Ball++)
	{
		double dist = (tclus.balls[Ball].pos - tclus.com).norm();
		if (dist > tclus.radius)
		{
			tclus.radius = dist;
		}
		// Center cluster mass at origin:
		tclus.balls[Ball].pos -= tclus.com;
	}

	tclus.com = { 0, 0, 0 }; // We just moved all balls to center the com.
	tclus.initConditions();

	std::cout << "Balls in current file: " << tclus.numBalls << std::endl;
	std::cout << "Mass in current file: " << tclus.m << std::endl;
	std::cout << "Approximate radius of current file: " << tclus.radius << " centimeters.\n";
	return tclus;
}

