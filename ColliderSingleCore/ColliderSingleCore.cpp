#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <fstream>
#include <ctime>
#include <sstream>
#include <iomanip>
#include "../vector3d.hpp"
#include "../initializations.hpp"
#include "../ballGroup.hpp"


// String buffers to hold data in memory until worth writing to file:
std::stringstream ballBuffer;
std::stringstream energyBuffer;

// These are used within simOneStep to keep track of time.
// They need to survive outside its scope, and I don't want to have to pass them all.
const time_t start = time(nullptr);        // For end of program analysis
time_t startProgress; // For progress reporting (gets reset)
time_t lastWrite;     // For write control (gets reset)
bool writeStep;       // This prevents writing to file every step (which is slow).

/// @brief The ballGroup run by the main sim looper.
ballGroup O;


// Prototypes
void simInitTwoCluster();
void simContinue();
void simInitCondAndCenter();
void simOneStep(const unsigned int& Step);
[[noreturn]] void simLooper();
void generateBallField();
void safetyChecks();
void calibrateDT(const unsigned int& Step, const double& customSpeed = -1.0);
void updateDTK(const double& vel);
void simType(const char simType);

void simType(const char simType)
{
	switch (simType)
	{
	case 'g':
		generateBallField();
		break;
	case 'c':
		simContinue();
		break;
	case 't':
		simInitTwoCluster();
		break;
	default:
		std::cout << "Did not choose a simulation type.";
		break;
	}
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
int main(const int argc, char const* argv[])
{
	energyBuffer.precision(12); // Need more precision on momentum.

	// Runtime arguments:
	if (argc > 1)
	{
		//numThreads = atoi(argv[1]);
		//printf("\nThread count set to %i.\n", numThreads);
		//projectileName = argv[2];
		//targetName = argv[3];
		//KEfactor = atof(argv[4]);
	}

	simType('c'); // c: continue old sim | t: two cluster collision | g: generate cluster
	O.zeroAngVel();
	//O.pushApart();
	calibrateDT(0, vTarget);
	simInitCondAndCenter();
	safetyChecks();
	O.simInitWrite(outputPrefix);
	simLooper();

} // end main
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////


// Set's up a two cluster collision.
void simInitTwoCluster()
{
	// Load file data:
	std::cerr << "TWO CLUSTER SIM\nFile 1: " << projectileName << '\t' << "File 2: " << targetName << '\n';
	//ballGroup projectile(path + targetName);

	// DART PROBE
	ballGroup projectile(1);
	projectile.pos[0] = { 8800, 0, 0 };
	projectile.w[0] = { 0, 0, 0 };
	projectile.vel[0] = { 0, 0, 0 };
	projectile.R[0] = 78.5;
	projectile.m[0] = 560000;
	projectile.moi[0] = .4 * projectile.m[0] * projectile.R[0] * projectile.R[0];

	ballGroup target(path + targetName);

	// DO YOU WANT TO STOP EVERYTHING?
	projectile.zeroAngVel();
	projectile.zeroVel();
	target.zeroAngVel();
	target.zeroVel();


	// Calc info to determined cluster positioning and collisions velocity:
	projectile.updatePE();
	target.updatePE();

	//projectile.offset(projectile.radius, target.radius + (projectile.R[0]), impactParameter);

	const double PEsys = projectile.PE + target.PE + (-G * projectile.getMass() * target.getMass() / (projectile.getCOM() - target.getCOM()).norm());

	// Collision velocity calculation:
	const double mSmall = projectile.getMass();
	const double mBig = target.getMass();
	const double mTot = mBig + mSmall;
	const double vSmall = -sqrt(2 * KEfactor * fabs(PEsys) * (mBig / (mSmall * mTot))); // Negative because small offsets right.
	//vSmall = -600000; // DART probe override.
	const double vBig = -(mSmall / mBig) * vSmall; // Negative to be opposing projectile.
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


void simContinue()
{
	// Load file data:
	std::cerr << "Continuing Sim...\nFile: " << targetName << '\n';

	O.loadSim(path + targetName);

	O.toOrigin();

	std::cout << '\n';
	O.checkMomentum("O");

	// Name the file based on info above:
	outputPrefix =
		O.cNumBalls +
		"-rho" + rounder(density, 4);
}


void simInitCondAndCenter()
{
	std::cout << "==================" << '\n';
	std::cout << "dt: " << dt << '\n';
	std::cout << "k: " << kin << '\n';
	std::cout << "Skip: " << skip << '\n';
	std::cout << "Steps: " << steps << '\n';
	std::cout << "==================" << '\n';

	O.checkMomentum("After Zeroing"); // Is total mom zero like it should be?

	// Compute physics between all balls. Distances, collision forces, energy totals, total mass:
	O.initConditions();

	// Name the file based on info above:
	outputPrefix +=
		"-k" + scientific(kin) +
		"-dt" + scientific(dt) +
		"_";
}



void simOneStep(const unsigned int& Step)
{
	// Check if this is a write step:
	if (Step % skip == 0)
	{
		writeStep = true;

		simTimeElapsed += dt * skip;

		// Progress reporting:
		float eta = ((time(nullptr) - startProgress) / 500.f * static_cast<float>(steps - Step)) / 3600.f; // In seconds.
		float elapsed = (time(nullptr) - start) / 3600.f;
		float progress = (Step / steps * 100.f);
		printf("Step: %u\tProgress: %2.0f%%\tETA: %5.2lf hr\tElapsed: %5.2f hr\n", Step, progress, eta, elapsed);
		startProgress = time(nullptr);
	}
	else
	{
		writeStep = false;
	}

	/// FIRST PASS - Update Kinematic Parameters:
	for (unsigned int Ball = 0; Ball < O.cNumBalls; Ball++)
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
	for (unsigned int A = 1; A < O.cNumBalls; A++) //cuda
	{
		/// DONT DO ANYTHING HERE. A STARTS AT 1.
		for (unsigned int B = 0; B < A; B++)
		{
			const double sumRaRb = O.R[A] + O.R[B];
			vector3d rVec = O.pos[B] - O.pos[A]; // Start with rVec from a to b.
			const double dist = (rVec).norm();
			vector3d totalForce;

			// Check for collision between Ball and otherBall:
			double overlap = sumRaRb - dist;

			// Distance array element: 1,0    2,0    2,1    3,0    3,1    3,2 ...
			unsigned int e = static_cast<unsigned>(A * (A - 1) * .5) + B;
			double oldDist = O.distances[e];

			// Check for collision between Ball and otherBall.
			if (overlap > 0)
			{
				// todo - Calibrate based on probe velocity ignore others.
				double k;
				// Apply coefficient of restitution to balls leaving collision.
				if (dist >= oldDist)
				{
					k = kout;
				}
				else
				{
					k = kin;
				}

				// todo - functionalize this shit and disable friction for the hardening of dymorphous.
				// Elastic a:
				vector3d elasticForce = -k * overlap * .5 * (rVec / dist);
				const double elasticMag = elasticForce.norm();

				// Friction a:
				//vector3d dVel = O.vel[B] - O.vel[A];
				//vector3d frictionForce;
				//const vector3d relativeVelOfA = dVel - dVel.dot(rVec) * (rVec / (dist * dist)) - O.w[A].cross(O.R[A] / sumRaRb * rVec) - O.w[B].cross(O.R[B] / sumRaRb * rVec);
				//double relativeVelMag = relativeVelOfA.norm();
				//if (relativeVelMag > 1e-10) // When relative velocity is very low, dividing its vector components by its magnitude below is unstable.
				//{
				//	frictionForce = mu * elasticMag * (relativeVelOfA / relativeVelMag);
				//}
				//const vector3d aTorque = (O.R[A] / sumRaRb) * rVec.cross(frictionForce);

				// Translational forces don't need to know about torque of b:
				const vector3d gravForceOnA = (G * O.m[A] * O.m[B] / (dist * dist)) * (rVec / dist);
				totalForce = gravForceOnA + elasticForce; // +frictionForce;

				// Elastic and Friction b:
				// Flip direction b -> a:
				//rVec = -rVec; 
				//dVel = -dVel;
				//elasticForce = -elasticForce;

				//const vector3d relativeVelOfB = dVel - dVel.dot(rVec) * (rVec / (dist * dist)) - O.w[B].cross(O.R[B] / sumRaRb * rVec) - O.w[A].cross(O.R[A] / sumRaRb * rVec);
				//relativeVelMag = relativeVelOfB.norm();
				//if (relativeVelMag > 1e-10)
				//{
				//	frictionForce = mu * elasticMag * (relativeVelOfB / relativeVelMag);
				//}
				//const vector3d bTorque = (O.R[B] / sumRaRb) * rVec.cross(frictionForce);
				//
				//O.aacc[A] += aTorque / O.moi[A];
				//O.aacc[B] += bTorque / O.moi[B];


				if (writeStep)
				{
					// Calculate potential energy. Important to recognize that the factor of 1/2 is not in front of K because this is for the spring potential in each ball and they are the same potential.
					const double x = (O.R[A] + O.R[B] - dist);
					O.PE += -G * O.m[A] * O.m[B] / dist + 0.5 * k * x * x;
				}
			}
			else
			{
				// No collision: Include gravity only:
				const vector3d gravForceOnA = (G * O.m[A] * O.m[B] / (dist * dist)) * (rVec / dist);
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
	for (unsigned int Ball = 0; Ball < O.cNumBalls; Ball++)
	{
		double vMax = 0;
		// Report vMax:
		if (O.vel[Ball].norm() > vMax)
		{
			vMax = O.vel[Ball].norm();
		}
		std::cout << "vMax = " << vMax << "Steps recorded: " << Step / skip << '\n';

		// Velocity for next step:
		O.vel[Ball] = O.velh[Ball] + .5 * O.acc[Ball] * dt;
		O.w[Ball] = O.wh[Ball] + .5 * O.aacc[Ball] * dt;

		if (writeStep)
		{

			// Send positions and rotations to buffer:
			if (Ball == 0)
			{
				ballBuffer
					<< O.pos[Ball][0] << ','
					<< O.pos[Ball][1] << ','
					<< O.pos[Ball][2] << ','
					<< O.w[Ball][0] << ','
					<< O.w[Ball][1] << ','
					<< O.w[Ball][2] << ','
					<< O.w[Ball].norm() << ','
					<< O.vel[Ball].x << ','
					<< O.vel[Ball].y << ','
					<< O.vel[Ball].z << ','
					<< 0;
			}
			else
			{
				ballBuffer
					<< ',' << O.pos[Ball][0] << ','
					<< O.pos[Ball][1] << ','
					<< O.pos[Ball][2] << ','
					<< O.w[Ball][0] << ','
					<< O.w[Ball][1] << ','
					<< O.w[Ball][2] << ','
					<< O.w[Ball].norm() << ','
					<< O.vel[Ball].x << ',' <<
					O.vel[Ball].y << ','
					<< O.vel[Ball].z << ','
					<< 0;
			}

			O.KE += .5 * O.m[Ball] * O.vel[Ball].normsquared() + .5 * O.moi[Ball] * O.w[Ball].normsquared(); // Now includes rotational kinetic energy.
			O.mom += O.m[Ball] * O.vel[Ball];
			O.angMom += O.m[Ball] * O.pos[Ball].cross(O.vel[Ball]) + O.moi[Ball] * O.w[Ball];
		}
	} // THIRD PASS END

	if (writeStep)
	{
		// Write energy to stream:
		energyBuffer << '\n'
			<< simTimeElapsed << ','
			<< O.PE << ','
			<< O.KE << ','
			<< O.PE + O.KE << ','
			<< O.mom.norm() << ','
			<< O.angMom.norm(); // the two zeros are bound and unbound mass

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
		if (time(nullptr) - lastWrite > 1800 || Step / skip % 10 == 0 || Step == steps - 1)
		{
			std::cout << "Data Write\n\n";

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

			lastWrite = time(nullptr);
		} // Data export end
		//calibrateDT(Step, false);
	} // writestep end
} // Steps end


[[noreturn]] void simLooper()
{
	std::cout << "Beginning simulation...\n";

	for (unsigned int Step = 1; Step < steps; Step++) // Steps start at 1 because the 0 step is initial conditions.
	{
		simOneStep(Step);
	}
	const time_t end = time(nullptr);

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




void twoSizeSphereShell5000()
{
	double radius = O.getRadius();

	for (unsigned int Ball = 0; Ball < 1000; Ball++)
	{
		O.R[Ball] = 700;
		O.m[Ball] = density * 4. / 3. * 3.14159 * std::pow(O.R[Ball], 3);
		O.moi[Ball] = .4 * O.m[Ball] * O.R[Ball] * O.R[Ball];
		O.w[Ball] = { 0, 0, 0 };
		O.pos[Ball] = randShellVec(spaceRange, radius);
	}

	for (unsigned int Ball = 1000; Ball < 2000; Ball++)
	{
		O.R[Ball] = 400;
		O.m[Ball] = density * 4. / 3. * 3.14159 * std::pow(O.R[Ball], 3);
		O.moi[Ball] = .4 * O.m[Ball] * O.R[Ball] * O.R[Ball];
		O.w[Ball] = { 0, 0, 0 };
		O.pos[Ball] = randShellVec(spaceRange, radius);
	}

	const unsigned int ballsInPhase1 = 2000;
	std::cout << "Balls in phase: " << ballsInPhase1 << "\n";

	// Generate non-overlapping spherical particle field:
	// Note that int can only handle 46340 spheres before potential int overflow.
	int collisionDetected = 0;
	int oldCollisions = INT_MAX;

	for (unsigned int failed = 0; failed < attempts; failed++)
	{
		for (unsigned int A = 0; A < ballsInPhase1; A++)
		{
			for (unsigned int B = A + 1; B < ballsInPhase1; B++)
			{
				// Check for Ball overlap.
				const double dist = (O.pos[A] - O.pos[B]).norm();
				const double sumRaRb = O.R[A] + O.R[B];
				const double overlap = dist - sumRaRb;
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
		if (failed == attempts - 1 || collisionDetected > static_cast<int>(1.5 * static_cast<double>(ballsInPhase1))) // Added the second part to speed up spatial constraint increase when there are clearly too many collisions for the space to be feasible.
		{
			std::cout << "Failed " << spaceRange << ". Increasing range " << spaceRangeIncrement << "cm^3.\n";
			spaceRange += spaceRangeIncrement;
			failed = 0;
			for (unsigned int Ball = 0; Ball < ballsInPhase1; Ball++)
			{
				O.pos[Ball] = randShellVec(spaceRange, radius); // Each time we fail and increase range, redistribute all balls randomly so we don't end up with big balls near mid and small balls outside.
			}
		}
		collisionDetected = 0;
	}

	spaceRange += 2. * O.R[0] + 4. * 250.;
	radius += O.R[0] + 250.;
	std::cout << "Making shell between " << radius << " and " << spaceRange * .5 << '\n';

	// PHASE 2

	for (unsigned int Ball = 2000; Ball < 3500; Ball++)
	{
		O.R[Ball] = 250;
		O.m[Ball] = density * 4. / 3. * 3.14159 * std::pow(O.R[Ball], 3);
		O.moi[Ball] = .4 * O.m[Ball] * O.R[Ball] * O.R[Ball];
		O.w[Ball] = { 0, 0, 0 };
		O.pos[Ball] = randShellVec(spaceRange, radius);
	}

	for (unsigned int Ball = 3500; Ball < 5000; Ball++)
	{
		O.R[Ball] = 150;
		O.m[Ball] = density * 4. / 3. * 3.14159 * std::pow(O.R[Ball], 3);
		O.moi[Ball] = .4 * O.m[Ball] * O.R[Ball] * O.R[Ball];
		O.w[Ball] = { 0, 0, 0 };
		O.pos[Ball] = randShellVec(spaceRange, radius);
	}

	const unsigned int ballsInPhase2 = 3000;
	std::cout << "Balls in phase: " << ballsInPhase2 << "\n";

	// Generate non-overlapping spherical particle field:
	collisionDetected = 0;
	oldCollisions = 100000000;

	for (unsigned int failed = 0; failed < attempts; failed++)
	{
		for (unsigned int A = ballsInPhase1; A < ballsInPhase1 + ballsInPhase2; A++)
		{
			for (unsigned int B = A + 1; B < ballsInPhase1 + ballsInPhase2; B++)
			{
				// Check for Ball overlap.
				const double dist = (O.pos[A] - O.pos[B]).norm();
				const double sumRaRb = O.R[A] + O.R[B];
				const double overlap = dist - sumRaRb;
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
		if (failed == attempts - 1 || collisionDetected > static_cast<int>(1.5 * static_cast<double>(ballsInPhase2))) // Added the second part to speed up spatial constraint increase when there are clearly too many collisions for the space to be feasible.
		{
			std::cout << "Failed " << spaceRange << ". Increasing range " << spaceRangeIncrement << "cm^3.\n";
			spaceRange += spaceRangeIncrement;
			failed = 0;
			for (unsigned int Ball = ballsInPhase1; Ball < ballsInPhase1 + ballsInPhase2; Ball++)
			{
				O.pos[Ball] = randShellVec(spaceRange, radius); // Each time we fail and increase range, redistribute all balls randomly so we don't end up with big balls near mid and small balls outside.
			}
		}
		collisionDetected = 0;
	}

	std::cout << "Initial Radius: " << radius << '\n';
	std::cout << "Mass: " << O.getMass() << '\n';

}



void threeSizeSphere()
{
	// Make genBalls of 3 sizes in CGS with ratios such that the mass is distributed evenly among the 3 sizes (less large genBalls than small genBalls).
	const unsigned int smalls = static_cast<unsigned>(std::round(static_cast<double>(genBalls) * 27. / 31.375)); // Just here for reference. Whatever genBalls are left will be smalls.
	const unsigned int mediums = static_cast<unsigned>(std::round(static_cast<double>(genBalls) * 27. / (8 * 31.375)));
	const unsigned int larges = static_cast<unsigned>(std::round(static_cast<double>(genBalls) * 1. / 31.375));


	for (unsigned int Ball = 0; Ball < larges; Ball++)
	{
		O.R[Ball] = 3. * scaleBalls;//std::pow(1. / (double)genBalls, 1. / 3.) * 3. * scaleBalls;
		O.m[Ball] = density * 4. / 3. * 3.14159 * std::pow(O.R[Ball], 3);
		O.moi[Ball] = .4 * O.m[Ball] * O.R[Ball] * O.R[Ball];
		O.w[Ball] = { 0, 0, 0 };
		O.pos[Ball] = randSphericalVec(spaceRange, spaceRange, spaceRange);
	}

	for (unsigned int Ball = larges; Ball < (larges + mediums); Ball++)
	{
		O.R[Ball] = 2. * scaleBalls;//std::pow(1. / (double)genBalls, 1. / 3.) * 2. * scaleBalls;
		O.m[Ball] = density * 4. / 3. * 3.14159 * std::pow(O.R[Ball], 3);
		O.moi[Ball] = .4 * O.m[Ball] * O.R[Ball] * O.R[Ball];
		O.w[Ball] = { 0, 0, 0 };
		O.pos[Ball] = randSphericalVec(spaceRange, spaceRange, spaceRange);
	}
	for (unsigned int Ball = (larges + mediums); Ball < genBalls; Ball++)
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

	for (unsigned int failed = 0; failed < attempts; failed++)
	{
		for (unsigned int A = 0; A < genBalls; A++)
		{
			for (unsigned int B = A + 1; B < genBalls; B++)
			{
				// Check for Ball overlap.
				const double dist = (O.pos[A] - O.pos[B]).norm();
				const double sumRaRb = O.R[A] + O.R[B];
				const double overlap = dist - sumRaRb;
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
		if (failed == attempts - 1 || collisionDetected > static_cast<int>(1.5 * static_cast<double>(genBalls))) // Added the second part to speed up spatial constraint increase when there are clearly too many collisions for the space to be feasible.
		{
			std::cout << "Failed " << spaceRange << ". Increasing range " << spaceRangeIncrement << "cm^3.\n";
			spaceRange += spaceRangeIncrement;
			failed = 0;
			for (unsigned int Ball = 0; Ball < genBalls; Ball++)
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



void oneSizeSphere()
{

	for (unsigned int Ball = 0; Ball < genBalls; Ball++)
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

	for (unsigned int failed = 0; failed < attempts; failed++)
	{
		for (unsigned int A = 0; A < genBalls; A++)
		{
			for (unsigned int B = A + 1; B < genBalls; B++)
			{
				// Check for Ball overlap.
				const double dist = (O.pos[A] - O.pos[B]).norm();
				const double sumRaRb = O.R[A] + O.R[B];
				const double overlap = dist - sumRaRb;
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
		if (failed == attempts - 1 || collisionDetected > static_cast<int>(1.5 * static_cast<double>(genBalls))) // Added the second part to speed up spatial constraint increase when there are clearly too many collisions for the space to be feasible.
		{
			std::cout << "Failed " << spaceRange << ". Increasing range " << spaceRangeIncrement << "cm^3.\n";
			spaceRange += spaceRangeIncrement;
			failed = 0;
			for (unsigned int Ball = 0; Ball < genBalls; Ball++)
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



void generateBallField()
{
	std::cout << "CLUSTER FORMATION\n";
	O.allocateGroup(genBalls);


	// Create new random number set.
	const unsigned int seedSave = static_cast<unsigned>(time(nullptr));
	srand(seedSave);

	//twoSizeSphereShell5000();
	//oneSizeSphere();
	threeSizeSphere();
	O.initialRadius = O.getRadius();
	O.mTotal = O.getMass();

	outputPrefix =
		std::to_string(genBalls) +
		"-R" + scientific(O.getRadius()) +
		"-cor" + rounder(std::pow(cor, 2), 4) +
		"-mu" + rounder(mu, 3) +
		"-rho" + rounder(density, 4);
}



void safetyChecks()
{
	titleBar("SAFETY CHECKS");

	if (skip == 0)
	{
		printf("\nSKIP NOT SET\n");
		exit(EXIT_FAILURE);
	}

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

	if (steps == 0)
	{
		printf("\nSTEPS NOT SET\n");
		exit(EXIT_FAILURE);
	}

	if (O.getRadius() <= 0)
	{
		printf("\nCluster radius <= 0\n");
		exit(EXIT_FAILURE);
	}

	for (unsigned int Ball = 0; Ball < O.cNumBalls; Ball++)
	{
		if (O.pos[Ball].norm() < vector3d(1e-10, 1e-10, 1e-10).norm())
		{
			printf("\nA ball position is [0,0,0]. Possibly didn't initialize balls properly.\n");
			exit(EXIT_FAILURE);
		}

		if (O.R[Ball] <= 0)
		{
			printf("\nA balls radius <= 0.\n");
			exit(EXIT_FAILURE);
		}

		if (O.m[Ball] <= 0)
		{
			printf("\nA balls mass <= 0.\n");
			exit(EXIT_FAILURE);
		}
	}
	titleBar("SAFETY PASSED");
}


void calibrateDT(const unsigned int& Step, const double& customSpeed)
{
	const double dtOld = dt;

	if (customSpeed > 0.)
	{
		updateDTK(customSpeed);
		std::cout << "CUSTOM SPEED: " << customSpeed;
	}
	else
	{
		// Sim fall velocity onto cluster:
		// vCollapse shrinks if a ball escapes but velMax should take over at that point, unless it is ignoring far balls.
		double position = 0;
		double vCollapse = 0;
		while (position < O.initialRadius)
		{
			vCollapse += G * O.mTotal / (O.initialRadius * O.initialRadius) * 0.1;
			position += vCollapse * 0.1;
		}
		vCollapse = fabs(vCollapse);

		std::cout << vCollapse << " <- vCollapse | Lazz Calc -> " << M_PI * M_PI * G * pow(density, 4. / 3.) * pow(O.mTotal, 2. / 3.) * O.rMax;
		system("pause");

		soc = 2 * O.initialRadius; // sphere of consideration for max velocity, to avoid very unbound high vel balls.

		double vMax = O.getVelMax(false);

		std::cout << '\n';

		// Take whichever velocity is greatest:
		if (vMax > fabs(vCollapse))
		{
			std::cout << "vMax > binding: " << vCollapse << " = vCollapse | vMax = " << vMax;
		}
		else
		{
			std::cout << "Binding > vMax: " << vCollapse << " = vCollapse | vMax = " << vMax;
			vMax = vCollapse;
		}

		updateDTK(vMax);
	}

	// todo - If current vMax greater than original, send warning and pause simulation.


	if (Step == 0 or dtOld < 0)
	{
		steps = static_cast<unsigned>(simTimeSeconds / dt);
		std::cout << " Step count: " << steps << '\n';
	}
	else
	{
		steps = static_cast<unsigned>(dt / dtOld * (steps - Step) + Step);
		std::cout << " New step count: " << steps << '\n';
	}

	if (timeResolution / dt > 1.)
	{
		skip = static_cast<unsigned>(floor(timeResolution / dt));
	}
	else
	{
		std::cout << "Desired time resolution is lower than dt.\n";
		system("pause");
	}
}

//void setGuidDT(const double& vel)
//{
//	// Guidos k and dt:
//	dt = .01 * O.getRmin() / fabs(vel);
//}
//
//void setGuidK(const double& vel)
//{
//	kin = O.getMassMax() * vel * vel / (.1 * O.R[0] * .1 * O.R[0]);
//	kout = cor * kin;
//}

void updateDTK(const double& vel)
{
	constexpr double kConsts = fourThirdsPiRho / (maxOverlap * maxOverlap);
	const double rMin = O.getRmin();
	const double rMax = O.getRmax();

	kin = kConsts * rMax * vel * vel;
	kout = cor * kin;
	dt = .01 * sqrt((fourThirdsPiRho / kin) * rMin * rMin * rMin);
}
