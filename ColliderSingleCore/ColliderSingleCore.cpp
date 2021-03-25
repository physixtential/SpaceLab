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



// String buffers to hold data in memory until worth writing to file:
std::stringstream ballBuffer;
std::stringstream energyBuffer;

// These are used within simOneStep to keep track of time.
// They need to survive outside its scope, and I don't want to have to pass them all.
time_t start = time(NULL);        // For end of program analysis
time_t startProgress; // For progress reporting (gets reset)
time_t lastWrite;     // For write control (gets reset)
bool writeStep;       // This prevents writing to file every step (which is slow).

/// @brief The ballGroup run by the main sim looper.
ballGroup O;

// Prototypes
inline void simInitTwoCluster();
inline void simContinue();
inline void simInitCondAndCenter();
inline void simOneStep(const int& Step);
inline void simLooper();
inline void generateBallField();
inline void safetyChecks();
inline void calibrateDT(const int& Step, bool superSafe, bool doK);
inline void setGuidDT(const double& vel);
inline void setGuidK(const double& vel);
inline void setLazzDT(const double& vel);
inline void setLazzK(const double& vel);


//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
int main(int argc, char const* argv[])
{
	// Runtime arguments:
	if (argc > 1)
	{
		//numThreads = atoi(argv[1]);
		//printf("\nThread count set to %i.\n", numThreads);
		//projectileName = argv[2];
		//targetName = argv[3];
		//KEfactor = atof(argv[4]);
	}

	//simInitTwoCluster();
	simContinue();
	O.pushApart();
	//generateBallField();
	simInitCondAndCenter();
	safetyChecks();
	O.simInitWrite(outputPrefix);
	simLooper();

	return 0;
} // end main
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////



inline void simInitTwoCluster()
{
	// Load file data:
	std::cerr << "TWO CLUSTER SIM\nFile 1: " << projectileName << '\t' << "File 2: " << targetName << '\n';
	//ballGroup projectile(path + targetName);

	// DART PROBE
	ballGroup projectile(1);
	projectile.pos[0] = { 8800,0,0 };
	projectile.w[0] = { 0,0,0 };
	projectile.vel[0] = { 0,0,0 };
	projectile.R[0] = 78.5;
	projectile.m[0] = 560000;
	projectile.moi[0] = .4 * projectile.m[0] * projectile.R[0] * projectile.R[0];

	ballGroup target(path + targetName);

	// DO YOU WANT TO STOP EVERYTHING?
	projectile.zeroMotion();
	target.zeroMotion();

	// Calc info to determined cluster positioning and collisions velocity:
	projectile.updatePE();
	target.updatePE();

	//projectile.offset(projectile.radius, target.radius + (projectile.R[0]), impactParameter);

	double PEsys = projectile.PE + target.PE + (-G * projectile.getMass() * target.getMass() / (projectile.com - target.com).norm());

	// Collision velocity calculation:
	double mSmall = projectile.getMass();
	double mBig = target.getMass();
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

	std::cout << '\n';
	projectile.checkMomentum("Projectile");
	target.checkMomentum("Target");

	O.allocateGroup(projectile.cNumBalls + target.cNumBalls);

	O.addBallGroup(target);
	O.addBallGroup(projectile); // projectile second so smallest ball at end and largest ball at front for dt/k calcs.

	outputPrefix =
		projectileName + targetName +
		"T" + rounder(KEfactor, 4) +
		"-vBig" + scientific(vBig) +
		"-vSmall" + scientific(vSmall) +
		"-IP" + rounder(impactParameter * 180 / 3.14159, 2) +
		"-rho" + rounder(density, 4);
}


inline void simContinue()
{
	// Load file data:
	std::cerr << "Continuing Sim...\nFile: " << targetName << '\n';

	O.importDataFromFile(path + targetName);

	O.toOrigin();

	std::cout << '\n';
	O.checkMomentum("O");

	// Name the file based on info above:
	outputPrefix =
		projectileName + targetName +
		"T" + rounder(KEfactor, 4) +
		"-rho" + rounder(density, 4);
}


inline void simInitCondAndCenter()
{


	calibrateDT(0, true, true);

	// Hack temporarily manually setting k and dt
	// k and dt override to stabilize cluster.
	std::cout << "OVERRIDING K AND DT" << '\n';
	kin = 1.01787e16;
	kout = cor * kin;
	calibrateDT(0, true, false);
	steps = (size_t)(simTimeSeconds / dt);

	std::cout << "==================" << '\n';
	std::cout << "dt: " << dt << '\n';
	std::cout << "k: " << kin << '\n';
	std::cout << "Steps: " << steps << '\n';
	std::cout << "==================" << '\n';



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



inline void simOneStep(int& Step)
{
	// Check if this is a write step:
	if (Step % skip == 0)
	{
		writeStep = true;

		// Progress reporting:
		float eta = ((time(NULL) - startProgress) / 500.0 * (steps - Step)) / 3600.; // In seconds.
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

			// So last distance can be known for COR:
			O.distances[e] = dist;
		}
		// DONT DO ANYTHING HERE. A STARTS AT 1.
	}

	if (writeStep)
	{
		ballBuffer << '\n'; // Prepares a new line for incoming data.
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
		energyBuffer << '\n'
			<< simTimeElapsed << ',' << O.PE << ',' << O.KE << ',' << O.PE + O.KE << ',' << O.mom.norm() << ',' << O.angMom.norm(); // the two zeros are bound and unbound mass

		// Reinitialize energies for next step:
		O.KE = 0;
		O.PE = 0;
		O.mom = { 0, 0, 0 };
		O.angMom = { 0, 0, 0 };
		// unboundMass = 0;
		// boundMass = massTotal;

		////////////////////////////////////////////////////////////////////
		// Data Export /////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////
		if (time(NULL) - lastWrite > 1800 || Step / skip % 20 == 0 || Step == steps - 1)
		{
			std::cout << "\nData Write" << '\n';

			// Write simData to file and clear buffer.
			std::ofstream ballWrite;
			ballWrite.open(outputPrefix + "simData.csv", std::ofstream::app);
			ballWrite << ballBuffer.rdbuf(); // Barf buffer to file.
			ballBuffer.str("");              // Empty the stream for next filling.
			ballWrite.close();

			// Write Energy data to file and clear buffer.
			std::ofstream energyWrite;
			energyWrite.open(outputPrefix + "energy.csv", std::ofstream::app);
			energyWrite << energyBuffer.rdbuf();
			energyBuffer.str(""); // Empty the stream for next filling.
			energyWrite.close();

			lastWrite = time(NULL);
		} // Data export end
		calibrateDT(Step, true, false);
		simTimeElapsed += dt * skip;
	} // writestep end
} // Steps end


inline void simLooper()
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




inline void twoSizeSphereShell5000()
{
	double radius = O.getRadius();

	for (int Ball = 0; Ball < 1000; Ball++)
	{
		O.R[Ball] = 700;
		O.m[Ball] = density * 4. / 3. * 3.14159 * std::pow(O.R[Ball], 3);
		O.moi[Ball] = .4 * O.m[Ball] * O.R[Ball] * O.R[Ball];
		O.w[Ball] = { 0, 0, 0 };
		O.pos[Ball] = randShellVec(spaceRange, radius);
	}

	for (int Ball = 1000; Ball < 2000; Ball++)
	{
		O.R[Ball] = 400;
		O.m[Ball] = density * 4. / 3. * 3.14159 * std::pow(O.R[Ball], 3);
		O.moi[Ball] = .4 * O.m[Ball] * O.R[Ball] * O.R[Ball];
		O.w[Ball] = { 0, 0, 0 };
		O.pos[Ball] = randShellVec(spaceRange, radius);
	}

	int ballsInPhase1 = 2000;
	std::cout << "Balls in phase: " << ballsInPhase1 << "\n";

	// Generate non-overlapping spherical particle field:
	int collisionDetected = 0;
	int oldCollisions = (int)1e10;

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
					O.pos[B] = randShellVec(spaceRange, radius);
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
		if (failed == attempts - 1 || collisionDetected > int(1.5 * (double)ballsInPhase1)) // Added the second part to speed up spatial constraint increase when there are clearly too many collisions for the space to be feasible.
		{
			std::cout << "Failed " << spaceRange << ". Increasing range " << spaceRangeIncrement << "cm^3.\n";
			spaceRange += spaceRangeIncrement;
			failed = 0;
			for (int Ball = 0; Ball < ballsInPhase1; Ball++)
			{
				O.pos[Ball] = randShellVec(spaceRange, radius); // Each time we fail and increase range, redistribute all balls randomly so we don't end up with big balls near mid and small balls outside.
			}
		}
		collisionDetected = 0;
	}

	spaceRange += 2 * O.R[0] + 4 * 250;
	radius += O.R[0] + 250;
	std::cout << "Making shell between " << radius << " and " << spaceRange * .5 << '\n';

	// PHASE 2

	for (int Ball = 2000; Ball < 3500; Ball++)
	{
		O.R[Ball] = 250;
		O.m[Ball] = density * 4. / 3. * 3.14159 * std::pow(O.R[Ball], 3);
		O.moi[Ball] = .4 * O.m[Ball] * O.R[Ball] * O.R[Ball];
		O.w[Ball] = { 0, 0, 0 };
		O.pos[Ball] = randShellVec(spaceRange, radius);
	}

	for (int Ball = 3500; Ball < 5000; Ball++)
	{
		O.R[Ball] = 150;
		O.m[Ball] = density * 4. / 3. * 3.14159 * std::pow(O.R[Ball], 3);
		O.moi[Ball] = .4 * O.m[Ball] * O.R[Ball] * O.R[Ball];
		O.w[Ball] = { 0, 0, 0 };
		O.pos[Ball] = randShellVec(spaceRange, radius);
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
					O.pos[B] = randShellVec(spaceRange, radius);
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
		if (failed == attempts - 1 || collisionDetected > int(1.5 * (double)ballsInPhase2)) // Added the second part to speed up spatial constraint increase when there are clearly too many collisions for the space to be feasible.
		{
			std::cout << "Failed " << spaceRange << ". Increasing range " << spaceRangeIncrement << "cm^3.\n";
			spaceRange += spaceRangeIncrement;
			failed = 0;
			for (int Ball = ballsInPhase1; Ball < ballsInPhase1 + ballsInPhase2; Ball++)
			{
				O.pos[Ball] = randShellVec(spaceRange, radius); // Each time we fail and increase range, redistribute all balls randomly so we don't end up with big balls near mid and small balls outside.
			}
		}
		collisionDetected = 0;
	}

	std::cout << "Initial Radius: " << radius << '\n';
	std::cout << "Mass: " << O.getMass() << '\n';

}



inline void threeSizeSphere()
{
	// Make genBalls of 3 sizes in CGS with ratios such that the mass is distributed evenly among the 3 sizes (less large genBalls than small genBalls).
	int smalls = std::round((double)genBalls * 27 / 31.375); // Just here for reference. Whatever genBalls are left will be smalls.
	int mediums = std::round((double)genBalls * 27 / (8 * 31.375));
	int larges = std::round((double)genBalls * 1 / 31.375);


	for (int Ball = 0; Ball < larges; Ball++)
	{
		O.R[Ball] = 3. * scaleBalls;//std::pow(1. / (double)genBalls, 1. / 3.) * 3. * scaleBalls;
		O.m[Ball] = density * 4. / 3. * 3.14159 * std::pow(O.R[Ball], 3);
		O.moi[Ball] = .4 * O.m[Ball] * O.R[Ball] * O.R[Ball];
		O.w[Ball] = { 0, 0, 0 };
		O.pos[Ball] = randSphericalVec(spaceRange, spaceRange, spaceRange);
	}

	for (int Ball = larges; Ball < (larges + mediums); Ball++)
	{
		O.R[Ball] = 2. * scaleBalls;//std::pow(1. / (double)genBalls, 1. / 3.) * 2. * scaleBalls;
		O.m[Ball] = density * 4. / 3. * 3.14159 * std::pow(O.R[Ball], 3);
		O.moi[Ball] = .4 * O.m[Ball] * O.R[Ball] * O.R[Ball];
		O.w[Ball] = { 0, 0, 0 };
		O.pos[Ball] = randSphericalVec(spaceRange, spaceRange, spaceRange);
	}
	for (int Ball = (larges + mediums); Ball < genBalls; Ball++)
	{
		O.R[Ball] = 1. * scaleBalls;//std::pow(1. / (double)genBalls, 1. / 3.) * 1. * scaleBalls;
		O.m[Ball] = density * 4. / 3. * 3.14159 * std::pow(O.R[Ball], 3);
		O.moi[Ball] = .4 * O.m[Ball] * O.R[Ball] * O.R[Ball];
		O.w[Ball] = { 0, 0, 0 };
		O.pos[Ball] = randSphericalVec(spaceRange, spaceRange, spaceRange);
	}

	std::cout << "Smalls: " << smalls << " Mediums: " << mediums << " Larges: " << larges << '\n';

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
		if (failed == attempts - 1 || collisionDetected > int(1.5 * (double)genBalls)) // Added the second part to speed up spatial constraint increase when there are clearly too many collisions for the space to be feasible.
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

	std::cout << "Final spacerange: " << spaceRange << '\n';
	std::cout << "Initial Radius: " << O.getRadius() << '\n';
	std::cout << "Mass: " << O.getMass() << '\n';
}



inline void oneSizeSphere()
{

	for (int Ball = 0; Ball < genBalls; Ball++)
	{
		O.R[Ball] = scaleBalls;
		O.m[Ball] = density * 4. / 3. * 3.14159 * std::pow(O.R[Ball], 3);
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
		if (failed == attempts - 1 || collisionDetected > int(1.5 * (double)genBalls)) // Added the second part to speed up spatial constraint increase when there are clearly too many collisions for the space to be feasible.
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

	std::cout << "Final spacerange: " << spaceRange << '\n';
	std::cout << "Initial Radius: " << O.getRadius() << '\n';
	std::cout << "Mass: " << O.getMass() << '\n';
}



inline void generateBallField()
{
	std::cout << "CLUSTER FORMATION\n";
	O.allocateGroup(genBalls);


	// Create new random number set.
	int seedSave = time(NULL);
	srand(seedSave);

	//twoSizeSphereShell5000();
	//oneSizeSphere();
	//threeSizeSphere();
	testGen();

	outputPrefix =
		std::to_string(genBalls) +
		"-R" + scientific(O.getRadius()) +
		"-cor" + rounder(std::pow(cor, 2), 4) +
		"-mu" + rounder(mu, 3) +
		"-rho" + rounder(density, 4);
}



inline void safetyChecks()
{
	printf("\n//////////// SAFETY CHECKS ////////////\n");

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

	if (O.getRadius() == 0)
	{
		printf("\nRadius is 0\n");
		exit(EXIT_FAILURE);
	}

	for (size_t Ball = 0; Ball < O.cNumBalls; Ball++)
	{
		if (O.pos[Ball] == vector3d(0, 0, 0))
		{
			printf("\nA ball position is [0,0,0]. Possibly didn't initialize balls properly.\n");
			exit(EXIT_FAILURE);
		}

		if (O.R[Ball] == 0)
		{
			printf("\nA balls radius is 0.\n");
			exit(EXIT_FAILURE);
		}

		if (O.m[Ball] == 0)
		{
			printf("\nA balls mass is 0.\n");
			exit(EXIT_FAILURE);
		}
	}

	printf("\n//////////// SAFETY PASSED ////////////\n\n");
}


inline void calibrateDT(const int& Step, const bool superSafe, bool doK)
{
	double dtOld = dt;
	double radius = O.getRadius();
	double mass = O.getMass();

	// What velocity can a sphere have starting at 0 velocity after moving under gravity for 1/10 of the largest sphere radius.
	double position = 0;
	double vCollapse = 0;
	double rMin = O.getRmin();
	std::cout << "\nvCollapse:\n";
	while (position < rMin * .1)
	{
		std::cout << vCollapse << "                        \r";
		vCollapse += G * mass / (radius * radius) * 0.1;
		position += vCollapse * 0.1;
	}

	soc = 2 * radius; // sphere of consideration for max velocity, to avoid very unbound high vel balls.

	double vMax = O.getVelMax(false);

	// Check if the kick is going to be the most significant velocity basis, or if gravity will matter more.
	std::cout << '\n';
	if (vMax > fabs(vCollapse))
	{
		std::cout << "vMax > binding:\n" << vCollapse << " = vCollapse | vMax = " << vMax;

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
		std::cout << "Binding > vMax:\n" << vCollapse << " = vCollapse | vMax = " << vMax;

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

	if (Step == 0 or dtOld == -1)
	{
		steps = (size_t)(simTimeSeconds / dt);
		std::cout << " Step count: " << steps << '\n';
	}
	else
	{
		steps = dt / dtOld * (steps - Step) + Step;
		std::cout << " New step count: " << steps << '\n';
	}
}

void setGuidDT(const double& vel)
{
	// Guidos k and dt:
	dt = .01 * O.getRmin() / fabs(vel);
}

void setGuidK(const double& vel)
{
	kin = O.getMassMax() * vel * vel / (.1 * O.R[0] * .1 * O.R[0]);
	kout = cor * kin;
}

void setLazzDT(const double& vel)
{
	// Lazzati k and dt:
	// dt is ultimately depend on the velocities in the system, k is a part of this calculation because we derive dt with a dependence on k. Even if we don't choose to modify k, such as in the middle of a simulation (which would break conservation of energy), we maintain the concept of k for comprehension. One could just copy kTemp into the dt formula and ignore the k dependence.
	double rMin = O.getRmin();
	double kTemp = 4 / 3 * M_PI * density * O.getMassMax() * vel * vel / (.1 * .1);
	dt = .01 * sqrt(4 / 3 * M_PI * density / kTemp * rMin * rMin * rMin);
}

void setLazzK(const double& vel)
{
	kin = 4 / 3 * M_PI * density * O.getMassMax() * vel * vel / (.1 * .1);
	kout = cor * kin;
}
