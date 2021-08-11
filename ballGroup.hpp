#pragma once
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <limits.h>
#include <cstring>
#include "dust_const.hpp"
#include "vector3d.hpp"

/// @brief Facilitates the concept of a group of balls with physical properties.
class ballGroup
{
public:

	ballGroup() = default;

	/// @brief For creating a new ballGroup of size nBalls
	/// @param nBalls Number of balls to allocate.
	explicit ballGroup(const int nBalls)
	{
		allocateGroup(nBalls);
	}

	/// @brief For generating a new ballGroup of size nBalls
	/// @param nBalls Number of balls to allocate.
	/// @param generate Just here to get you to the right constructor. This is definitely wrong.
	/// @param customVel To condition for specific vMax.
	ballGroup(const int nBalls, const bool generate, const double& customVel)
	{
		generateBallField(nBalls);
		this->pos[0] = { 1e-4, 0, .4e-4 };
		this->pos[1] = { -1e-4, 0, 0 };
		this->vel[0] = { -0, 0, 0 };
		this->vel[1] = { 0, 0, 0 };
		this->m[1] = 10 * density * 4. / 3. * 3.14159 * std::pow(R[1], 3);
		this->moi[1] = .4 * m[1] * R[1] * R[1];
		mTotal = getMass();
		calc_v_collapse();
		calibrateDT(0, customVel);
		simInitCondAndCenter();
	}

	/// @brief For continuing a sim.
	/// @param fullpath is the filename and path excluding the suffix _simData.csv, _constants.csv, etc.
	/// @param customVel To condition for specific vMax.
	explicit ballGroup(const std::string& fullpath, const double& customVel)
	{
		simContinue(path, fullpath);
		calc_v_collapse();
		calibrateDT(0, customVel);
		simInitCondAndCenter();
	}

	/// @brief For two cluster sim.
	/// @param projectileName 
	/// @param targetName 
	/// @param customVel To condition for specific vMax.
	explicit ballGroup(const std::string& path, const std::string& projectileName, const std::string& targetName, const double& customVel)
	{
		simInitTwoCluster(path, projectileName, targetName);
		calc_v_collapse();
		calibrateDT(0, customVel);
		simInitCondAndCenter();
	}

	unsigned int cNumBalls = 0;
	unsigned int cNumBallsAdded = 0;

	// Useful values:
	double rMin = -1;
	double rMax = -1;
	double mTotal = -1;
	double initialRadius = -1;
	double vCollapse = 0;
	double vMax = -1;
	double vMaxPrev = HUGE_VAL;
	double soc = -1;

	vector3d mom = { 0, 0, 0 };
	vector3d angMom = { 0, 0, 0 }; // Can be vector3d because they only matter for writing out to file. Can process on host.

	double PE = 0, KE = 0;

	double* distances = nullptr;

	vector3d* pos = nullptr;
	vector3d* vel = nullptr;
	vector3d* velh = nullptr; ///< Velocity half step for integration purposes.
	vector3d* acc = nullptr;
	vector3d* w = nullptr;
	vector3d* wh = nullptr; ///< Angular velocity half step for integration purposes.
	vector3d* aacc = nullptr;
	double* R = nullptr; ///< Radius
	double* m = nullptr; ///< Mass
	double* moi = nullptr; ///< Moment of inertia

	void calibrateDT(const unsigned int& Step, const double& customSpeed = -1.)
	{
		const double dtOld = dt;

		if (customSpeed > 0.)
		{
			updateDTK(customSpeed);
			std::cerr << "CUSTOM SPEED: " << customSpeed;
		}
		else
		{
			//std::cerr << vCollapse << " <- vCollapse | Lazz Calc -> " << M_PI * M_PI * G * pow(density, 4. / 3.) * pow(mTotal, 2. / 3.) * rMax;

			vMax = getVelMax();

			std::cerr << '\n';

			// Take whichever velocity is greatest:
			std::cerr << vCollapse << " = vCollapse | vMax = " << vMax;
			if (vMax < vCollapse)
			{
				vMax = vCollapse;
			}

			if (vMax < vMaxPrev)
			{
				updateDTK(vMax);
				vMaxPrev = vMax;
				std::cerr << "\nk: " << kin << "\tdt: " << dt;
			}
		}

		if (Step == 0 or dtOld < 0)
		{
			steps = static_cast<unsigned>(simTimeSeconds / dt);
			std::cerr << "\tInitial Steps: " << steps;
		}
		else
		{
			steps = static_cast<unsigned>(dtOld / dt * (steps - Step) + Step);
			std::cerr << "\tSteps: " << steps;
		}

		if (timeResolution / dt > 1.)
		{
			skip = static_cast<unsigned>(floor(timeResolution / dt));
			std::cerr << "\tSkip: " << skip << '\n';
		}
		else
		{
			std::cerr << "Desired time resolution is lower than dt. Setting to 1 second per skip.\n";
			skip = static_cast<unsigned>(floor(1. / dt));
		}
	}

	// todo - make bigger balls favor the middle, or, smaller balls favor the outside.
	/// @brief Push balls apart until no overlaps
	void pushApart() const
	{
		std::cerr << "Separating spheres - Current max overlap:\n";
		/// Using acc array as storage for accumulated position change.
		int* counter = new int[cNumBalls];
		for (size_t Ball = 0; Ball < cNumBalls; Ball++)
		{
			acc[Ball] = { 0, 0, 0 };
			counter[Ball] = 0;
		}

		double overlapMax = -1;
		const double pseudoDT = rMin * .1;
		int step = 0;

		while (true)
		{
			//if (step % 10 == 0)
			//{
			//	simDataWrite("pushApart_");
			//}

			for (unsigned int A = 0; A < cNumBalls; A++)
			{
				for (unsigned int B = A + 1; B < cNumBalls; B++)
				{
					// Check for Ball overlap.
					vector3d rVecab = pos[B] - pos[A];
					vector3d rVecba = -1 * rVecab;
					const double dist = (rVecab).norm();
					const double sumRaRb = R[A] + R[B];
					const double overlap = sumRaRb - dist;

					if (overlapMax < overlap)
					{
						overlapMax = overlap;
					}

					if (overlap > 0)
					{
						acc[A] += overlap * (rVecba / dist);
						acc[B] += overlap * (rVecab / dist);
						counter[A] += 1;
						counter[B] += 1;
					}
				}
			}

			for (size_t Ball = 0; Ball < cNumBalls; Ball++)
			{
				if (counter[Ball] > 0)
				{
					pos[Ball] += acc[Ball].normalized() * pseudoDT;
					acc[Ball] = { 0, 0, 0 };
					counter[Ball] = 0;
				}
			}

			if (overlapMax > 0)
			{
				std::cerr << overlapMax << "                        \r";
			}
			else
			{
				std::cerr << "\nSuccess!\n";
				break;
			}
			overlapMax = -1;
			step++;
		}
		delete[] counter;
	}

	void calc_v_collapse()
	{
		// Sim fall velocity onto cluster:
		// vCollapse shrinks if a ball escapes but velMax should take over at that point, unless it is ignoring far balls.
		double position = 0;
		while (position < initialRadius)
		{
			// todo - include vdw!!!
			vCollapse += G * mTotal / (initialRadius * initialRadius) * 0.1;
			position += vCollapse * 0.1;
		}
		vCollapse = fabs(vCollapse);
	}

	/// get max velocity
	[[nodiscard]] double getVelMax()
	{
		vMax = 0;

		// todo - make this a manual set true or false to use soc so we know if it is being used or not.
		if (soc > 0)
		{
			int counter = 0;
			for (unsigned int Ball = 0; Ball < cNumBalls; Ball++)
			{
				// Only consider balls moving toward com and within 4x initial radius around it.
				const vector3d fromCOM = pos[Ball] - getCOM();
				if (acos(vel[Ball].normalized().dot(fromCOM.normalized())) > cone && fromCOM.norm() < soc)
				{
					if (vel[Ball].norm() > vMax)
					{
						vMax = vel[Ball].norm();
					}
				}
				else
				{
					counter++;
				}
			}
			std::cerr << '(' << counter << " spheres ignored" << ") ";
		}
		else
		{
			for (int Ball = 0; Ball < cNumBalls; Ball++)
			{
				if (vel[Ball].norm() > vMax)
				{
					vMax = vel[Ball].norm();
				}
			}

			// Is vMax for some reason unreasonably small? Don't proceed. Probably a finished sim.
			// This shouldn't apply to extremely destructive collisions because it is possible that no particles are considered, so it will keep pausing.
			if (vMax < 1e-10)
			{
				std::cerr << "\nMax velocity in system is less than 1e-10.\n";
				system("pause");
			}
		}

		return vMax;
	}

	// Kick ballGroup (give the whole thing a velocity)
	void kick(const double& vx, const double& vy, const double& vz) const
	{
		for (unsigned int Ball = 0; Ball < cNumBalls; Ball++)
		{
			vel[Ball] += {vx, vy, vz};
		}
	}


	void checkMomentum(const std::string& of) const
	{
		vector3d pTotal = { 0, 0, 0 };
		for (unsigned int Ball = 0; Ball < cNumBalls; Ball++)
		{
			pTotal += m[Ball] * vel[Ball];
		}
		fprintf(stderr, "%s Momentum Check: %.2e, %.2e, %.2e\n", of.c_str(), pTotal.x, pTotal.y, pTotal.z);
	}

	// offset cluster
	void offset(const double& rad1, const double& rad2, const double& impactParam) const
	{
		for (unsigned int Ball = 0; Ball < cNumBalls; Ball++)
		{
			pos[Ball].x += (rad1 + rad2) * cos(impactParam);
			pos[Ball].y += (rad1 + rad2) * sin(impactParam);
		}
	}

	/// Approximate the radius of the ballGroup.
	[[nodiscard]] double getRadius() const
	{
		double radius = 0;

		if (cNumBalls > 1)
		{
			for (unsigned int A = 0; A < cNumBalls; A++)
			{
				for (unsigned int B = A + 1; B < cNumBalls; B++)
				{
					// Identify two farthest balls from each other. That is diameter of cluster.
					const double diameter = (pos[A] - pos[B]).norm();
					if (diameter * .5 > radius)
					{
						radius = diameter * .5;
					}
				}
			}
		}
		else
		{
			radius = R[0];
		}

		return radius;
	}

	// Update Potential Energy:
	void updatePE()
	{
		PE = 0;

		if (cNumBalls > 1) // Code below only necessary for effects between balls.
		{
			for (unsigned int A = 1; A < cNumBalls; A++)
			{
				for (unsigned int B = 0; B < A; B++)
				{
					const double sumRaRb = R[A] + R[B];
					const double dist = (pos[A] - pos[B]).norm();
					const double overlap = sumRaRb - dist;

					// Check for collision between Ball and otherBall.
					if (overlap > 0)
					{
						PE += -G * m[A] * m[B] / dist + kin * ((sumRaRb - dist) * .5) * ((sumRaRb - dist) * .5);
					}
					else
					{
						PE += -G * m[A] * m[B] / dist;
					}
				}
			}
		}
		else // For the case of just one ball:
		{
			PE = 0;
		}
	}

	void simInitWrite(std::string& filename)
	{
		// Create string for file name identifying spin combination negative is 2, positive is 1 on each axis.
		//std::string spinCombo = "";
		//for (unsigned int i = 0; i < 3; i++)
		//{
		//	if (spins[i] < 0) { spinCombo += "2"; }
		//	else if (spins[i] > 0) { spinCombo += "1"; }
		//	else { spinCombo += "0"; }
		//}


		// Check if file name already exists.
		std::ifstream checkForFile;
		checkForFile.open(filename + "simData.csv", std::ifstream::in);
		// Add a counter to the file name until it isn't overwriting anything:
		int counter = 0;
		while (checkForFile.is_open())
		{
			counter++;
			checkForFile.close();
			checkForFile.open(std::to_string(counter) + '_' + filename + "simData.csv", std::ifstream::in);
		}

		if (counter > 0)
		{
			filename.insert(0, std::to_string(counter) + '_');
		}

		// Complete file names:
		std::string simDataFilename = filename + "simData.csv";
		std::string energyFilename = filename + "energy.csv";
		std::string constantsFilename = filename + "constants.csv";

		std::cerr << "New file tag: " << filename;

		// Open all file streams:
		std::ofstream energyWrite, ballWrite, constWrite;
		energyWrite.open(energyFilename, std::ofstream::app);
		ballWrite.open(simDataFilename, std::ofstream::app);
		constWrite.open(constantsFilename, std::ofstream::app);

		// Make column headers:
		energyWrite << "Time,PE,KE,E,p,L";
		ballWrite << "x0,y0,z0,wx0,wy0,wz0,wmag0,vx0,vy0,vz0,bound0";

		for (unsigned int Ball = 1; Ball < cNumBalls; Ball++) // Start at 2nd ball because first one was just written^.
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

		// Write constant data:
		for (unsigned int Ball = 0; Ball < cNumBalls; Ball++)
		{

			constWrite
				<< R[Ball] << ','
				<< m[Ball] << ','
				<< moi[Ball]
				<< '\n';
		}

		// Write energy data to buffer:
		energyBuffer
			<< '\n'
			<< simTimeElapsed << ','
			<< PE << ','
			<< KE << ','
			<< PE + KE << ','
			<< mom.norm() << ','
			<< angMom.norm();
		energyWrite << energyBuffer.rdbuf();
		energyBuffer.str("");

		// Reinitialize energies for next step:
		KE = 0;
		PE = 0;
		mom = { 0, 0, 0 };
		angMom = { 0, 0, 0 };

		// Send position and rotation to buffer:
		ballBuffer << '\n'; // Necessary new line after header.
		ballBuffer
			<< pos[0].x << ','
			<< pos[0].y << ','
			<< pos[0].z << ','
			<< w[0].x << ','
			<< w[0].y << ','
			<< w[0].z << ','
			<< w[0].norm() << ','
			<< vel[0].x << ','
			<< vel[0].y << ','
			<< vel[0].z << ','
			<< 0; //bound[0];
		for (unsigned int Ball = 1; Ball < cNumBalls; Ball++)
		{
			ballBuffer
				<< ',' << pos[Ball].x << ',' // Needs comma start so the last bound doesn't have a dangling comma.
				<< pos[Ball].y << ','
				<< pos[Ball].z << ','
				<< w[Ball].x << ','
				<< w[Ball].y << ','
				<< w[Ball].z << ','
				<< w[Ball].norm() << ','
				<< vel[Ball].x << ','
				<< vel[Ball].y << ','
				<< vel[Ball].z << ','
				<< 0; //bound[Ball];
		}
		// Write position and rotation data to file:
		ballWrite << ballBuffer.rdbuf();
		ballBuffer.str(""); // Resets the stream buffer to blank.

		// Close Streams for user viewing:
		energyWrite.close();
		ballWrite.close();
		constWrite.close();

		std::cerr << "\nSimulating " << steps * dt / 60 / 60 << " hours.\n";
		std::cerr << "Total mass: " << mTotal << '\n';
		std::cerr << "\n===============================================================\n";
	}


	[[nodiscard]] vector3d getCOM() const
	{
		if (mTotal > 0)
		{
			vector3d comNumerator;
			for (unsigned int Ball = 0; Ball < cNumBalls; Ball++)
			{
				comNumerator += m[Ball] * pos[Ball];
			}
			vector3d com = comNumerator / mTotal;
			return com;
		}
		else
		{
			std::cerr << "Mass of cluster is zero.\n";
			exit(EXIT_FAILURE);
		}
	}

	void zeroVel() const
	{
		for (unsigned int Ball = 0; Ball < cNumBalls; Ball++)
		{
			vel[Ball] = { 0, 0, 0 };
		}
	}

	void zeroAngVel() const
	{
		for (unsigned int Ball = 0; Ball < cNumBalls; Ball++)
		{
			w[Ball] = { 0, 0, 0 };
		}
	}

	void toOrigin() const
	{
		const vector3d com = getCOM();

		for (unsigned int Ball = 0; Ball < cNumBalls; Ball++)
		{
			pos[Ball] -= com;
		}
	}

	// Set velocity of all balls such that the cluster spins:
	void comSpinner(const double& spinX, const double& spinY, const double& spinZ) const
	{
		const vector3d comRot = { spinX, spinY, spinZ }; // Rotation axis and magnitude
		for (unsigned int Ball = 0; Ball < cNumBalls; Ball++)
		{
			vel[Ball] += comRot.cross(pos[Ball] - getCOM());
			w[Ball] += comRot;
		}
	}

	void rotAll(const char axis, const double angle) const
	{
		for (unsigned int Ball = 0; Ball < cNumBalls; Ball++)
		{
			pos[Ball] = pos[Ball].rot(axis, angle);
			vel[Ball] = vel[Ball].rot(axis, angle);
			w[Ball] = w[Ball].rot(axis, angle);
		}
	}

private:
	// String buffers to hold data in memory until worth writing to file:
	std::stringstream ballBuffer;
	std::stringstream energyBuffer;



	/// Allocate balls - MUST RUN updateUsefuls() after you have filled with data by whatever means.
	void allocateGroup(const unsigned int nBalls)
	{
		cNumBalls = nBalls;

		try
		{
			distances = new double[(cNumBalls * cNumBalls / 2) - (cNumBalls / 2)];

			pos = new vector3d[cNumBalls];
			vel = new vector3d[cNumBalls];
			velh = new vector3d[cNumBalls];
			acc = new vector3d[cNumBalls];
			w = new vector3d[cNumBalls];
			wh = new vector3d[cNumBalls];
			aacc = new vector3d[cNumBalls];
			R = new double[cNumBalls];
			m = new double[cNumBalls];
			moi = new double[cNumBalls];
		}
		catch (const std::exception& e)
		{
			std::cerr << "Failed trying to allocated group. " << e.what() << '\n';
		}
	}


	/// @brief Add another ballGroup into this one.
	/// @param src The ballGroup to be added.
	void addBallGroup(const ballGroup& src)
	{
		// Copy incoming data to the end of the currently loaded data.
		std::memcpy(&distances[cNumBallsAdded], src.distances, sizeof(src.distances[0]) * src.cNumBalls);
		std::memcpy(&pos[cNumBallsAdded], src.pos, sizeof(src.pos[0]) * src.cNumBalls);
		std::memcpy(&vel[cNumBallsAdded], src.vel, sizeof(src.vel[0]) * src.cNumBalls);
		std::memcpy(&velh[cNumBallsAdded], src.velh, sizeof(src.velh[0]) * src.cNumBalls);
		std::memcpy(&acc[cNumBallsAdded], src.acc, sizeof(src.acc[0]) * src.cNumBalls);
		std::memcpy(&w[cNumBallsAdded], src.w, sizeof(src.w[0]) * src.cNumBalls);
		std::memcpy(&wh[cNumBallsAdded], src.wh, sizeof(src.wh[0]) * src.cNumBalls);
		std::memcpy(&aacc[cNumBallsAdded], src.aacc, sizeof(src.aacc[0]) * src.cNumBalls);
		std::memcpy(&R[cNumBallsAdded], src.R, sizeof(src.R[0]) * src.cNumBalls);
		std::memcpy(&m[cNumBallsAdded], src.m, sizeof(src.m[0]) * src.cNumBalls);
		std::memcpy(&moi[cNumBallsAdded], src.moi, sizeof(src.moi[0]) * src.cNumBalls);

		// Keep track of now loaded ball set to start next set after it:
		cNumBallsAdded += src.cNumBalls;
		rMin = getRmin();
		rMax = getRmax();
		mTotal = getMass();
		initialRadius = getRadius();
		soc = 4 * rMax + initialRadius;

		// DON'T FORGET TO FREEMEMORY
	}

	/// @brief Deallocate arrays to recover memory.
	void freeMemory() const
	{
		delete[] distances;
		delete[] pos;
		delete[] vel;
		delete[] velh;
		delete[] acc;
		delete[] w;
		delete[] wh;
		delete[] aacc;
		delete[] R;
		delete[] m;
		delete[] moi;
	}




	// Initialize accelerations and energy calculations:
	void initConditions()
	{
		KE = 0;
		PE = 0;
		mom = { 0, 0, 0 };
		angMom = { 0, 0, 0 };
		if (cNumBalls > 1) // Code below only necessary for effects between balls.
		{
			// Because A starts at 1 below:
			KE += .5 * m[0] * vel[0].dot(vel[0]) + .5 * moi[0] * w[0].dot(w[0]);
			mom += m[0] * vel[0];
			angMom += m[0] * pos[0].cross(vel[0]) + moi[0] * w[0];
			for (unsigned int A = 1; A < cNumBalls; A++)
			{
				// Warning: "A" Starts at 1 not 0.

				for (unsigned int B = 0; B < A; B++)
				{
					const double sumRaRb = R[A] + R[B];
					const double dist = (pos[A] - pos[B]).norm();
					vector3d rVecab = pos[B] - pos[A];
					vector3d rVecba = -1 * rVecab;

					// Check for collision between Ball and otherBall:
					const double overlap = sumRaRb - dist;
					vector3d totalForce;
					vector3d aTorque;
					vector3d bTorque;

					// Check for collision between Ball and otherBall.
					if (overlap > 0)
					{
						// todo - refactor initConditions like main loop (verify this gives same values as other):
						// Calculate force and torque for a:
						vector3d dVel = vel[B] - vel[A];
						vector3d relativeVelOfA = dVel - dVel.dot(rVecab) * (rVecab / (dist * dist)) - w[A].cross(R[A] / sumRaRb * rVecab) - w[B].cross(R[B] / sumRaRb * rVecab);
						vector3d elasticForceOnA = -kin * overlap * .5 * (rVecab / dist);
						vector3d frictionForceOnA = { 0, 0, 0 };
						if (relativeVelOfA.norm() > 1e-10) // When relative velocity is very low, dividing its vector components by its magnitude below is unstable.
						{
							frictionForceOnA = u_s * elasticForceOnA.norm() * (relativeVelOfA / relativeVelOfA.norm());
						}
						aTorque = (R[A] / sumRaRb) * rVecab.cross(frictionForceOnA);

						// Calculate force and torque for b:
						dVel = vel[A] - vel[B];
						vector3d relativeVelOfB = dVel - dVel.dot(rVecba) * (rVecba / (dist * dist)) - w[B].cross(R[B] / sumRaRb * rVecba) - w[A].cross(R[A] / sumRaRb * rVecba);
						vector3d elasticForceOnB = -kin * overlap * .5 * (rVecba / dist);
						vector3d frictionForceOnB = { 0, 0, 0 };
						if (relativeVelOfB.norm() > 1e-10)
						{
							frictionForceOnB = u_s * elasticForceOnB.norm() * (relativeVelOfB / relativeVelOfB.norm());
						}
						bTorque = (R[B] / sumRaRb) * rVecba.cross(frictionForceOnB);

						vector3d gravForceOnA = (G * m[A] * m[B] / (dist * dist)) * (rVecab / dist);
						totalForce = gravForceOnA + elasticForceOnA + frictionForceOnA;
						if (isnan(totalForce.x) or isnan(totalForce.y) or isnan(totalForce.z))
						{
							std::cerr << "NAN";
						}
						aacc[A] += aTorque / moi[A];
						aacc[B] += bTorque / moi[B];
						PE += -G * m[A] * m[B] / dist + kin * ((sumRaRb - dist) * .5) * ((sumRaRb - dist) * .5);
					}
					else
					{
						// No collision: Include gravity only:
						const vector3d gravForceOnA = (G * m[A] * m[B] / (dist * dist)) * (rVecab / dist);
						totalForce = gravForceOnA;
						if (isnan(totalForce.x) or isnan(totalForce.y) or isnan(totalForce.z))
						{
							std::cerr << "NAN";
						}
						PE += -G * m[A] * m[B] / dist;
					}
					// Newton's equal and opposite forces applied to acceleration of each ball:
					acc[A] += totalForce / m[A];
					acc[B] -= totalForce / m[B];

					const unsigned int e = static_cast<unsigned>(A * (A - 1) * .5) + B; // Complex storage of n square over 2 distances.
					distances[e] = dist;
				}
				KE += .5 * m[A] * vel[A].dot(vel[A]) + .5 * moi[A] * w[A].dot(w[A]);
				mom += m[A] * vel[A];
				angMom += m[A] * pos[A].cross(vel[A]) + moi[A] * w[A];
			}
		}
		else // For the case of just one ball:
		{
			PE = 0;
			KE = .5 * m[0] * vel[0].dot(vel[0]) + .5 * moi[0] * w[0].dot(w[0]);
			mom = m[0] * vel[0];
			angMom = m[0] * pos[0].cross(vel[0]) + moi[0] * w[0];
		}
	}


	[[nodiscard]] double getRmin()
	{
		rMin = R[0];
		for (unsigned int Ball = 0; Ball < cNumBalls; Ball++)
		{
			if (R[Ball] < rMin)
			{
				rMin = R[Ball];
			}
		}
		return rMin;
	}

	[[nodiscard]] double getRmax()
	{
		rMax = R[0];
		for (unsigned int Ball = 0; Ball < cNumBalls; Ball++)
		{
			if (R[Ball] > rMax)
			{
				rMax = R[Ball];
			}
		}
		return rMax;
	}


	[[nodiscard]] double getMassMax() const
	{
		double mMax = m[0];
		for (unsigned int Ball = 0; Ball < cNumBalls; Ball++)
		{
			if (m[Ball] > mMax)
			{
				mMax = m[Ball];
			}
		}
		return mMax;
	}





	void parseSimData(std::string line)
	{
		std::string lineElement;

		// Get number of balls in file
		unsigned int count = std::count(line.begin(), line.end(), ',') / properties + 1;
		allocateGroup(count);

		std::stringstream chosenLine(line); // This is the last line of the read file, containing all data for all balls at last time step

		// Get position and angular velocity data:
		for (unsigned int A = 0; A < cNumBalls; A++)
		{

			for (unsigned int i = 0; i < 3; i++) // Position
			{
				std::getline(chosenLine, lineElement, ',');
				pos[A][i] = std::stod(lineElement);
				//std::cerr << tclus.pos[A][i]<<',';
			}
			for (unsigned int i = 0; i < 3; i++) // Angular Velocity
			{
				std::getline(chosenLine, lineElement, ',');
				w[A][i] = std::stod(lineElement);
			}
			std::getline(chosenLine, lineElement, ','); // Angular velocity magnitude skipped
			for (unsigned int i = 0; i < 3; i++)                 // velocity
			{
				std::getline(chosenLine, lineElement, ',');
				vel[A][i] = std::stod(lineElement);
			}
			for (unsigned int i = 0; i < properties - 10; i++) // We used 10 elements. This skips the rest.
			{
				std::getline(chosenLine, lineElement, ',');
			}
		}
	}


	/// Get previous sim constants by filename.
	void loadConsts(const std::string& path, const std::string& filename)
	{
		// Get radius, mass, moi:
		std::string constantsFilename = path + filename + "constants.csv";
		if (auto ConstStream = std::ifstream(constantsFilename, std::ifstream::in))
		{
			std::string line, lineElement;
			for (unsigned int A = 0; A < cNumBalls; A++)
			{
				std::getline(ConstStream, line); // Ball line.
				std::stringstream chosenLine(line);
				std::getline(chosenLine, lineElement, ','); // Radius.
				R[A] = std::stod(lineElement);
				std::getline(chosenLine, lineElement, ','); // Mass.
				m[A] = std::stod(lineElement);
				std::getline(chosenLine, lineElement, ','); // Moment of inertia.
				moi[A] = std::stod(lineElement);
			}
		}
		else
		{
			std::cerr << "Could not open constants file: " << constantsFilename << "... Existing program." << '\n';
			exit(EXIT_FAILURE);
		}
	}



	/// Get last line of previous simData by filename.
	[[nodiscard]] static std::string getLastLine(const std::string& path, const std::string& filename)
	{
		std::string simDataFilepath = path + filename + "simData.csv";

		if (auto simDataStream = std::ifstream(simDataFilepath, std::ifstream::in))
		{

			std::cerr << "\nParsing last line of data.\n";

			simDataStream.seekg(-1, std::ios_base::end); // go to one spot before the EOF

			bool keepLooping = true;
			while (keepLooping)
			{
				char ch = ' ';
				simDataStream.get(ch); // Get current byte's data

				if (static_cast<int>(simDataStream.tellg()) <= 1)
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
			std::string line;
			std::getline(simDataStream, line); // Read the current line

			return line;
		}
		else
		{
			std::cerr << "Could not open simData file: " << simDataFilepath << "... Existing program." << '\n';
			exit(EXIT_FAILURE);
		}

	}




	void simDataWrite(std::string& outFilename)
	{
		// todo - for some reason I need checkForFile instead of just using ballWrite. Need to work out why.
		// Check if file name already exists. If not, initialize
		std::ifstream checkForFile;
		checkForFile.open(outFilename + "simData.csv", std::ifstream::in);
		if (checkForFile.is_open() == false)
		{
			simInitWrite(outFilename);
		}
		else
		{
			ballBuffer << '\n'; // Prepares a new line for incoming data.

			for (size_t Ball = 0; Ball < cNumBalls; Ball++)
			{
				// Send positions and rotations to buffer:
				if (Ball == 0)
				{
					ballBuffer
						<< pos[Ball][0] << ','
						<< pos[Ball][1] << ','
						<< pos[Ball][2] << ','
						<< w[Ball][0] << ','
						<< w[Ball][1] << ','
						<< w[Ball][2] << ','
						<< w[Ball].norm() << ','
						<< vel[Ball].x << ','
						<< vel[Ball].y << ','
						<< vel[Ball].z << ','
						<< 0;
				}
				else
				{
					ballBuffer << ','
						<< pos[Ball][0] << ','
						<< pos[Ball][1] << ','
						<< pos[Ball][2] << ','
						<< w[Ball][0] << ','
						<< w[Ball][1] << ','
						<< w[Ball][2] << ','
						<< w[Ball].norm() << ','
						<< vel[Ball].x << ','
						<< vel[Ball].y << ','
						<< vel[Ball].z << ','
						<< 0;
				}
			}

			// Write simData to file and clear buffer.
			std::ofstream ballWrite;
			ballWrite.open(outFilename + "simData.csv", std::ofstream::app);
			ballWrite << ballBuffer.rdbuf(); // Barf buffer to file.
			ballBuffer.str("");              // Resets the stream for that balls to blank.
			ballWrite.close();
		}
		checkForFile.close();
	}



	[[nodiscard]] double getMass()
	{
		mTotal = 0;
		{
			for (int Ball = 0; Ball < cNumBalls; Ball++)
			{
				mTotal += m[Ball];
			}
		}
		return mTotal;
	}

	void threeSizeSphere(const int nBalls)
	{
		// Make nBalls of 3 sizes in CGS with ratios such that the mass is distributed evenly among the 3 sizes (less large nBalls than small nBalls).
		const unsigned int smalls = static_cast<unsigned>(std::round(static_cast<double>(nBalls) * 27. / 31.375)); // Just here for reference. Whatever nBalls are left will be smalls.
		const unsigned int mediums = static_cast<unsigned>(std::round(static_cast<double>(nBalls) * 27. / (8 * 31.375)));
		const unsigned int larges = static_cast<unsigned>(std::round(static_cast<double>(nBalls) * 1. / 31.375));


		for (int Ball = 0; Ball < larges; Ball++)
		{
			// Below comment maintains asteroid radius while increasing particle count.
			//std::pow(1. / (double)nBalls, 1. / 3.) * 3. * scaleBalls;

			R[Ball] = 3. * scaleBalls;
			m[Ball] = density * 4. / 3. * 3.14159 * std::pow(R[Ball], 3);
			moi[Ball] = .4 * m[Ball] * R[Ball] * R[Ball];
			w[Ball] = { 0, 0, 0 };
			pos[Ball] = rand_spherical_vec(spaceRange, spaceRange, spaceRange);
		}

		for (int Ball = larges; Ball < (larges + mediums); Ball++)
		{
			R[Ball] = 2. * scaleBalls;//std::pow(1. / (double)nBalls, 1. / 3.) * 2. * scaleBalls;
			m[Ball] = density * 4. / 3. * 3.14159 * std::pow(R[Ball], 3);
			moi[Ball] = .4 * m[Ball] * R[Ball] * R[Ball];
			w[Ball] = { 0, 0, 0 };
			pos[Ball] = rand_spherical_vec(spaceRange, spaceRange, spaceRange);
		}
		for (int Ball = (larges + mediums); Ball < nBalls; Ball++)
		{
			R[Ball] = 1. * scaleBalls;//std::pow(1. / (double)nBalls, 1. / 3.) * 1. * scaleBalls;
			m[Ball] = density * 4. / 3. * 3.14159 * std::pow(R[Ball], 3);
			moi[Ball] = .4 * m[Ball] * R[Ball] * R[Ball];
			w[Ball] = { 0, 0, 0 };
			pos[Ball] = rand_spherical_vec(spaceRange, spaceRange, spaceRange);
		}

		std::cerr << "Smalls: " << smalls << " Mediums: " << mediums << " Larges: " << larges << '\n';

		// Generate non-overlapping spherical particle field:
		int collisionDetected = 0;
		int oldCollisions = nBalls;

		for (unsigned int failed = 0; failed < attempts; failed++)
		{
			for (unsigned int A = 0; A < nBalls; A++)
			{
				for (unsigned int B = A + 1; B < nBalls; B++)
				{
					// Check for Ball overlap.
					const double dist = (pos[A] - pos[B]).norm();
					const double sumRaRb = R[A] + R[B];
					const double overlap = dist - sumRaRb;
					if (overlap < 0)
					{
						collisionDetected += 1;
						// Move the other ball:
						pos[B] = rand_spherical_vec(spaceRange, spaceRange, spaceRange);
					}
				}
			}
			if (collisionDetected < oldCollisions)
			{
				oldCollisions = collisionDetected;
				std::cerr << "Collisions: " << collisionDetected << "                        \r";
			}
			if (collisionDetected == 0)
			{
				std::cerr << "\nSuccess!\n";
				break;
			}
			if (failed == attempts - 1 || collisionDetected > static_cast<int>(1.5 * static_cast<double>(nBalls))) // Added the second part to speed up spatial constraint increase when there are clearly too many collisions for the space to be feasible.
			{
				std::cerr << "Failed " << spaceRange << ". Increasing range " << spaceRangeIncrement << "cm^3.\n";
				spaceRange += spaceRangeIncrement;
				failed = 0;
				for (unsigned int Ball = 0; Ball < nBalls; Ball++)
				{
					pos[Ball] = rand_spherical_vec(spaceRange, spaceRange, spaceRange); // Each time we fail and increase range, redistribute all balls randomly so we don't end up with big balls near mid and small balls outside.
				}
			}
			collisionDetected = 0;
		}

		std::cerr << "Final spacerange: " << spaceRange << '\n';
		std::cerr << "Initial Radius: " << getRadius() << '\n';
		std::cerr << "Mass: " << getMass() << '\n';
	}

	void generateBallField(const unsigned int nBalls)
	{
		std::cerr << "CLUSTER FORMATION\n";
		allocateGroup(nBalls);

		// Create new random number set.
		const int seedSave = static_cast<unsigned>(time(nullptr));
		srand(0);//srand(seedSave);

		threeSizeSphere(nBalls);

		rMin = getRmin();
		rMax = getRmax();
		mTotal = getMass();
		initialRadius = getRadius();
		soc = 4 * rMax + initialRadius;

		outputPrefix =
			std::to_string(nBalls) +
			"-R" + scientific(getRadius()) +
			"-v" + scientific(vCustom) +
			"-cor" + rounder(std::pow(cor, 2), 4) +
			"-mu" + rounder(u_s, 3) +
			"-rho" + rounder(density, 4);
	}

	/// Make ballGroup from file data.
	void loadSim(const std::string& path, const std::string& filename)
	{
		parseSimData(getLastLine(path, filename));

		loadConsts(path, filename);

		rMin = getRmin();
		rMax = getRmax();
		mTotal = getMass();
		initialRadius = getRadius();
		soc = 4 * rMax + initialRadius;

		std::cerr << "Balls: " << cNumBalls << '\n';
		std::cerr << "Mass: " << mTotal << '\n';
		std::cerr << "Approximate radius: " << initialRadius << " cm.\n";
	}



	void oneSizeSphere(const unsigned int nBalls)
	{

		for (unsigned int Ball = 0; Ball < nBalls; Ball++)
		{
			R[Ball] = scaleBalls;
			m[Ball] = density * 4. / 3. * 3.14159 * std::pow(R[Ball], 3);
			moi[Ball] = .4 * m[Ball] * R[Ball] * R[Ball];
			w[Ball] = { 0, 0, 0 };
			pos[Ball] = rand_spherical_vec(spaceRange, spaceRange, spaceRange);
		}

		// Generate non-overlapping spherical particle field:
		int collisionDetected = 0;
		int oldCollisions = nBalls;

		for (unsigned int failed = 0; failed < attempts; failed++)
		{
			for (unsigned int A = 0; A < nBalls; A++)
			{
				for (unsigned int B = A + 1; B < nBalls; B++)
				{
					// Check for Ball overlap.
					const double dist = (pos[A] - pos[B]).norm();
					const double sumRaRb = R[A] + R[B];
					const double overlap = dist - sumRaRb;
					if (overlap < 0)
					{
						collisionDetected += 1;
						// Move the other ball:
						pos[B] = rand_spherical_vec(spaceRange, spaceRange, spaceRange);
					}
				}
			}
			if (collisionDetected < oldCollisions)
			{
				oldCollisions = collisionDetected;
				std::cerr << "Collisions: " << collisionDetected << "                        \r";
			}
			if (collisionDetected == 0)
			{
				std::cerr << "\nSuccess!\n";
				break;
			}
			if (failed == attempts - 1 || collisionDetected > static_cast<int>(1.5 * static_cast<double>(nBalls))) // Added the second part to speed up spatial constraint increase when there are clearly too many collisions for the space to be feasible.
			{
				std::cerr << "Failed " << spaceRange << ". Increasing range " << spaceRangeIncrement << "cm^3.\n";
				spaceRange += spaceRangeIncrement;
				failed = 0;
				for (unsigned int Ball = 0; Ball < nBalls; Ball++)
				{
					pos[Ball] = rand_spherical_vec(spaceRange, spaceRange, spaceRange); // Each time we fail and increase range, redistribute all balls randomly so we don't end up with big balls near mid and small balls outside.
				}
			}
			collisionDetected = 0;
		}

		std::cerr << "Final spacerange: " << spaceRange << '\n';
		std::cerr << "Initial Radius: " << getRadius() << '\n';
		std::cerr << "Mass: " << mTotal << '\n';
	}



	void updateDTK(const double& vel)
	{
		kin = kConsts * rMax * vel * vel;
		kout = cor * kin;
		dt = .01 * sqrt((fourThirdsPiRho / kin) * rMin * rMin * rMin);
	}

	void simInitCondAndCenter()
	{
		std::cerr << "==================" << '\n';
		std::cerr << "dt: " << dt << '\n';
		std::cerr << "k: " << kin << '\n';
		std::cerr << "Skip: " << skip << '\n';
		std::cerr << "Steps: " << steps << '\n';
		std::cerr << "==================" << '\n';

		toOrigin();

		checkMomentum("After Zeroing"); // Is total mom zero like it should be?

		// Compute physics between all balls. Distances, collision forces, energy totals, total mass:
		initConditions();

		// Name the file based on info above:
		outputPrefix +=
			"_k" + scientific(kin) +
			"_dt" + scientific(dt) +
			"_";
	}


	void simContinue(const std::string& path, const std::string& filename)
	{
		// Load file data:
		std::cerr << "Continuing Sim...\nFile: " << filename << '\n';

		loadSim(path, filename);

		std::cerr << '\n';
		checkMomentum("O");

		// Name the file based on info above:
		outputPrefix =
			std::to_string(cNumBalls) +
			"_rho" + rounder(density, 4);
	}

	// Set's up a two cluster collision.
	void simInitTwoCluster(const std::string& path, const std::string& projectileName, const std::string& targetName)
	{
		// Load file data:
		std::cerr << "TWO CLUSTER SIM\nFile 1: " << projectileName << '\t' << "File 2: " << targetName << '\n';

		// DART PROBE
		//ballGroup projectile(1);
		//projectile.pos[0] = { 8814, 0, 0 };
		//projectile.w[0] = { 0, 0, 0 };
		//projectile.vel[0] = { 0, 0, 0 };
		//projectile.R[0] = 78.5;
		//projectile.m[0] = 560000;
		//projectile.moi[0] = .4 * projectile.m[0] * projectile.R[0] * projectile.R[0];

		ballGroup projectile;
		projectile.loadSim(path, projectileName);
		ballGroup target;
		target.loadSim(path, targetName);

		// DO YOU WANT TO STOP EVERYTHING?
		projectile.zeroAngVel();
		projectile.zeroVel();
		target.zeroAngVel();
		target.zeroVel();


		// Calc info to determined cluster positioning and collisions velocity:
		projectile.updatePE();
		target.updatePE();

		projectile.offset(projectile.initialRadius, target.initialRadius + target.getRmax() * 2, impactParameter);

		const double PEsys = projectile.PE + target.PE + (-G * projectile.mTotal * target.mTotal / (projectile.getCOM() - target.getCOM()).norm());

		// Collision velocity calculation:
		const double mSmall = projectile.mTotal;
		const double mBig = target.mTotal;
		const double mTot = mBig + mSmall;
		//const double vSmall = -sqrt(2 * KEfactor * fabs(PEsys) * (mBig / (mSmall * mTot))); // Negative because small offsets right.
		const double vSmall = -vCustom; // DART probe override.
		const double vBig = -(mSmall / mBig) * vSmall; // Negative to oppose projectile.
		//const double vBig = 0; // Dymorphous override.

		if (isnan(vSmall) || isnan(vBig))
		{
			std::cerr << "A VELOCITY WAS NAN!!!!!!!!!!!!!!!!!!!!!!\n\n";
			exit(EXIT_FAILURE);
		}

		projectile.kick(vSmall, 0, 0);
		target.kick(vBig, 0, 0);

		fprintf(stderr, "\nTarget Velocity: %.2e\nProjectile Velocity: %.2e\n", vBig, vSmall);

		std::cerr << '\n';
		projectile.checkMomentum("Projectile");
		target.checkMomentum("Target");

		allocateGroup(projectile.cNumBalls + target.cNumBalls);

		addBallGroup(target);
		addBallGroup(projectile); // projectile second so smallest ball at end and largest ball at front for dt/k calcs.

		outputPrefix =
			projectileName + targetName +
			"T" + rounder(KEfactor, 4) +
			"_vBig" + scientific(vBig) +
			"_vSmall" + scientific(vSmall) +
			"_IP" + rounder(impactParameter * 180 / 3.14159, 2) +
			"_rho" + rounder(density, 4);
	}
};