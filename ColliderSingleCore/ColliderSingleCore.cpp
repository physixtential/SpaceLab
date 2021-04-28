#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
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


// Prototypes
void simOneStep(const unsigned int& Step);
[[noreturn]] void simLooper();
void safetyChecks();


//ballGroup O(path + projectileName, path + targetName, 0); // Collision
ballGroup O(path + targetName, 0); // Continue
//ballGroup O(genBalls, true, 0); // Generate

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

	//O.zeroAngVel();
	//O.pushApart();
	safetyChecks();
	O.simInitWrite(outputPrefix);
	simLooper();

} // end main
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////




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
			unsigned int e = static_cast<unsigned>(A * (A - 1) * .5) + B; // a^2-a is always even, so this works.
			double oldDist = O.distances[e];

			// Check for collision between Ball and otherBall.
			if (overlap > 0)
			{
				// hack - temporary code for dymorphous collapse.
				/*const double rho = 4 / 3 * M_PI * O.initialRadius * O.initialRadius * O.initialRadius;
				const double dMax = M_PI * G * rho * O.initialRadius * O.mTotal / kTarget / 5.;
				if (overlap > dMax)
				{
					std::cout << dMax << "####### dMax Reached #######\n";
					writeStep = true;
					system("pause");
				}*/

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

				// Elastic a:
				vector3d elasticForce = -k * overlap * .5 * (rVec / dist);
				const double elasticMag = elasticForce.norm();

				// Friction a:
				vector3d dVel = O.vel[B] - O.vel[A];
				vector3d frictionForce;
				const vector3d relativeVelOfA = dVel - dVel.dot(rVec) * (rVec / (dist * dist)) - O.w[A].cross(O.R[A] / sumRaRb * rVec) - O.w[B].cross(O.R[B] / sumRaRb * rVec);
				double relativeVelMag = relativeVelOfA.norm();
				if (relativeVelMag > 1e-10) // When relative velocity is very low, dividing its vector components by its magnitude below is unstable.
				{
					frictionForce = mu * elasticMag * (relativeVelOfA / relativeVelMag);
				}
				const vector3d aTorque = (O.R[A] / sumRaRb) * rVec.cross(frictionForce);

				// Translational forces don't need to know about torque of b:
				const vector3d gravForceOnA = (G * O.m[A] * O.m[B] / (dist * dist)) * (rVec / dist);
				totalForce = gravForceOnA + elasticForce + frictionForce;

				// Elastic and Friction b:
				// Flip direction b -> a:
				rVec = -rVec;
				dVel = -dVel;
				elasticForce = -elasticForce;

				const vector3d relativeVelOfB = dVel - dVel.dot(rVec) * (rVec / (dist * dist)) - O.w[B].cross(O.R[B] / sumRaRb * rVec) - O.w[A].cross(O.R[A] / sumRaRb * rVec);
				relativeVelMag = relativeVelOfB.norm();
				if (relativeVelMag > 1e-10)
				{
					frictionForce = mu * elasticMag * (relativeVelOfB / relativeVelMag);
				}
				const vector3d bTorque = (O.R[B] / sumRaRb) * rVec.cross(frictionForce);

				O.aacc[A] += aTorque / O.moi[A];
				O.aacc[B] += bTorque / O.moi[B];


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
		/// Data Export ////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////
		if (time(nullptr) - lastWrite > 1800 || Step / skip % 10 == 0)
		{
			// Report vMax:
			std::cout << "vMax = " << O.vMax << " Steps recorded: " << Step / skip << '\n';
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
		O.calibrateDT(Step, false);
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












void safetyChecks()
{
	titleBar("SAFETY CHECKS");

	if (O.vCollapse <= 0)
	{
		printf("\nvCollapse NOT SET\n");
		exit(EXIT_FAILURE);
	}

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

	if (O.initialRadius <= 0)
	{
		printf("\nCluster initialRadius not set\n");
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


