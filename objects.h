#pragma once

/// @brief Facilitates the concept of a group of balls with physical properties.
/// Recommended: Use ballGroup(int nBalls) constructor to allocate all the memory needed for your ballGroup size.
struct ballGroup
{
	ballGroup() = default;

	/// Constructor to allocate all the memory needed for your ballGroup size.
	ballGroup(int nBalls) /// Desired number of balls in group.
	{
		allocateGroup(nBalls);
	}

	int cNumBalls = 0;
	int cNumBallsAdded = 0;

	vector3d
		com = { 0, 0, 0 },
		mom = { 0, 0, 0 },
		angMom = { 0, 0, 0 }; // Can be vector3d because they only matter for writing out to file. Can process on host.

	double mTotal = 0, radius = 0;
	double PE = 0, KE = 0;

	double* distances = 0;

	vector3d* pos = 0;
	vector3d* vel = 0;
	vector3d* velh = 0; ///< Velocity half step for integration purposes.
	vector3d* acc = 0;
	vector3d* w = 0;
	vector3d* wh = 0; ///< Angular velocity half step for integration purposes.
	vector3d* aacc = 0;
	double* R = 0; ///< Radius
	double* m = 0; ///< Mass
	double* moi = 0; ///< Moment of inertia

	/// Allocate ball property arrays.
	void allocateGroup(int nBalls)
	{
		cNumBalls = nBalls;

		distances = new double[(cNumBalls * cNumBalls / 2.) - (cNumBalls / 2.)];

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


	/// Add another ballGroup into this one.
	void addBallGroup(ballGroup* src)
	{
		// Copy incoming data to the end of the currently loaded data.
		memcpy(&distances[cNumBallsAdded], src->distances, sizeof(src->distances[0]) * src->cNumBalls);
		memcpy(&pos[cNumBallsAdded], src->pos, sizeof(src->pos[0]) * src->cNumBalls);
		memcpy(&vel[cNumBallsAdded], src->vel, sizeof(src->vel[0]) * src->cNumBalls);
		memcpy(&velh[cNumBallsAdded], src->velh, sizeof(src->velh[0]) * src->cNumBalls);
		memcpy(&acc[cNumBallsAdded], src->acc, sizeof(src->acc[0]) * src->cNumBalls);
		memcpy(&w[cNumBallsAdded], src->w, sizeof(src->w[0]) * src->cNumBalls);
		memcpy(&wh[cNumBallsAdded], src->wh, sizeof(src->wh[0]) * src->cNumBalls);
		memcpy(&aacc[cNumBallsAdded], src->aacc, sizeof(src->aacc[0]) * src->cNumBalls);
		memcpy(&R[cNumBallsAdded], src->R, sizeof(src->R[0]) * src->cNumBalls);
		memcpy(&m[cNumBallsAdded], src->m, sizeof(src->m[0]) * src->cNumBalls);
		memcpy(&moi[cNumBallsAdded], src->moi, sizeof(src->moi[0]) * src->cNumBalls);

		if (cNumBallsAdded > 0)
		{
			radius = -1; // radius is meaningless if there is more than one cluster:
		}

		// Keep track of now loaded ball set to start next set after it:
		cNumBallsAdded += src->cNumBalls;

		// DON'T FORGET TO FREEMEMORY
	}

	/// Deallocate arrays to recover memory.
	void freeMemory()
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

	/// Approximate the radius of the ballGroup.
	void updateRadius()
	{
		radius = 0;
		if (cNumBalls > 1)
		{
			for (int A = 0; A < cNumBalls; A++)
			{
				for (int B = A + 1; B < cNumBalls; B++)
				{
					// Identify two farthest balls from eachother. That is diameter of cluster.
					double diameter = (pos[A] - pos[B]).norm();
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
	}

	vector3d updateComAndMass()
	{
		mTotal = 0;
		{
			for (int Ball = 0; Ball < cNumBalls; Ball++)
			{
				mTotal += m[Ball];
			}
		}

		if (mTotal > 0)
		{
			vector3d comNumerator = { 0, 0, 0 };
			for (int Ball = 0; Ball < cNumBalls; Ball++)
			{
				comNumerator += m[Ball] * pos[Ball];
			}
			com = comNumerator / mTotal;
			return com;
		}
		else
		{
			std::cout << "Mass of cluster is zero...\n";
			return { NULL, NULL, NULL };
		}
	}

	void zeroMotion()
	{
		for (int Ball = 0; Ball < cNumBalls; Ball++)
		{
			w[Ball] = { 0, 0, 0 };
			vel[Ball] = { 0, 0, 0 };
		}
	}

	void toOrigin()
	{
		updateComAndMass();

		for (int Ball = 0; Ball < cNumBalls; Ball++)
		{
			pos[Ball] -= com;
		}
		updateComAndMass();
	}

	// Set velocity of all balls such that the cluster spins:
	void comSpinner(double spinX, double spinY, double spinZ)
	{
		vector3d comRot = { spinX, spinY, spinZ }; // Rotation axis and magnitude
		for (int Ball = 0; Ball < cNumBalls; Ball++)
		{
			vel[Ball] += comRot.cross(pos[Ball] - com);
			w[Ball] += comRot;
		}
	}

	void rotAll(char axis, double angle)
	{
		for (int Ball = 0; Ball < cNumBalls; Ball++)
		{
			pos[Ball] = pos[Ball].rot(axis, angle);
			vel[Ball] = vel[Ball].rot(axis, angle);
			w[Ball] = w[Ball].rot(axis, angle);
		}
	}



	// Initialzie accelerations and energy calculations:
	void initConditions()
	{
		mTotal = 0;
		KE = 0;
		PE = 0;
		mom = { 0, 0, 0 };
		angMom = { 0, 0, 0 };
		if (cNumBalls > 1) // Code below only necessary for effects between balls.
		{
			vector3d comNumerator = { 0, 0, 0 };

			// Because A starts at 1 below:
			mTotal += m[0];
			KE += .5 * m[0] * vel[0].dot(vel[0]) + .5 * moi[0] * w[0].dot(w[0]);
			mom += m[0] * vel[0];
			angMom += m[0] * pos[0].cross(vel[0]) + moi[0] * w[0];
			for (int A = 1; A < cNumBalls; A++)
			{
				// Warning: "A" Starts at 1 not 0.
				mTotal += m[A];
				comNumerator += m[A] * pos[A];

				for (int B = 0; B < A; B++)
				{
					double sumRaRb = R[A] + R[B];
					double dist = (pos[A] - pos[B]).norm();
					vector3d rVecab = pos[B] - pos[A];
					vector3d rVecba = -1 * rVecab;

					// Check for collision between Ball and otherBall:
					double overlap = sumRaRb - dist;
					vector3d totalForce = { 0, 0, 0 };
					vector3d aTorque = { 0, 0, 0 };
					vector3d bTorque = { 0, 0, 0 };

					// Check for collision between Ball and otherBall.
					if (overlap > 0)
					{
						// Calculate force and torque for a:
						vector3d dVel = vel[B] - vel[A];
						vector3d relativeVelOfA = dVel - dVel.dot(rVecab) * (rVecab / (dist * dist)) - w[A].cross(R[A] / sumRaRb * rVecab) - w[B].cross(R[B] / sumRaRb * rVecab);
						vector3d elasticForceOnA = -kin * overlap * .5 * (rVecab / dist);
						vector3d frictionForceOnA = { 0,0,0 };
						if (relativeVelOfA.norm() > 1e-12) // When relative velocity is very low, dividing its vector components by its magnitude below is unstable.
						{
							frictionForceOnA = mu * elasticForceOnA.norm() * (relativeVelOfA / relativeVelOfA.norm());
						}
						aTorque = (R[A] / sumRaRb) * rVecab.cross(frictionForceOnA);

						// Calculate force and torque for b:
						dVel = vel[A] - vel[B];
						vector3d relativeVelOfB = dVel - dVel.dot(rVecba) * (rVecba / (dist * dist)) - w[B].cross(R[B] / sumRaRb * rVecba) - w[A].cross(R[A] / sumRaRb * rVecba);
						vector3d elasticForceOnB = -kin * overlap * .5 * (rVecba / dist);
						vector3d frictionForceOnB = { 0,0,0 };
						if (relativeVelOfB.norm() > 1e-12)
						{
							frictionForceOnB = mu * elasticForceOnB.norm() * (relativeVelOfB / relativeVelOfB.norm());
						}
						bTorque = (R[B] / sumRaRb) * rVecba.cross(frictionForceOnB);

						vector3d gravForceOnA = (G * m[A] * m[B] / (dist * dist)) * (rVecab / dist);
						totalForce = gravForceOnA + elasticForceOnA + frictionForceOnA;
						aacc[A] += aTorque / moi[A];
						aacc[B] += bTorque / moi[B];
						PE += -G * m[A] * m[B] / dist + kin * ((sumRaRb - dist) * .5) * ((sumRaRb - dist) * .5);
					}
					else
					{
						// No collision: Include gravity only:
						vector3d gravForceOnA = (G * m[A] * m[B] / (dist * dist)) * (rVecab / dist);
						totalForce = gravForceOnA;
						PE += -G * m[A] * m[B] / dist;
					}
					// Newton's equal and opposite forces applied to acceleration of each ball:
					acc[A] += totalForce / m[A];
					acc[B] -= totalForce / m[B];
					int e = (A * (A - 1) * .5) + B;
					distances[e] = dist;
				}
				KE += .5 * m[A] * vel[A].dot(vel[A]) + .5 * moi[A] * w[A].dot(w[A]);
				mom += m[A] * vel[A];
				angMom += m[A] * pos[A].cross(vel[A]) + moi[A] * w[A];
			}
			com = comNumerator / mTotal;
		}
		else // For the case of just one ball:
		{
			mTotal = m[0];
			PE = 0;
			KE = .5 * m[0] * vel[0].dot(vel[0]) + .5 * moi[0] * w[0].dot(w[0]);
			mom = m[0] * vel[0];
			angMom = m[0] * pos[0].cross(vel[0]) + moi[0] * w[0];
			radius = R[0];
		}
	}

	// Update Potential Energy:
	void updatePE()
	{
		PE = 0;

		if (cNumBalls > 1) // Code below only necessary for effects between balls.
		{
			for (int A = 1; A < cNumBalls; A++)
			{
				for (int B = 0; B < A; B++)
				{
					double sumRaRb = R[A] + R[B];
					double dist = (pos[A] - pos[B]).norm();
					double overlap = sumRaRb - dist;

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


	// Kick ballGroup (give the whole thing a velocity)
	void kick(double vx, double vy, double vz)
	{
		for (int Ball = 0; Ball < cNumBalls; Ball++)
		{
			vel[Ball] += {vx, vy, vz};
		}
	}


	void checkMomentum(std::string of)
	{
		vector3d pTotal = { 0,0,0 };
		for (int Ball = 0; Ball < cNumBalls; Ball++)
		{
			pTotal += m[Ball] * vel[Ball];
		}
		printf("%s Momentum Check: %.2e, %.2e, %.2e\n", of.c_str(), pTotal.x, pTotal.y, pTotal.z);
	}

	// offset cluster
	void offset(double rad1, double rad2, double impactParam)
	{
		for (int Ball = 0; Ball < cNumBalls; Ball++)
		{
			pos[Ball].x += (rad1 + rad2) * cos(impactParam);
			pos[Ball].y += (rad1 + rad2) * sin(impactParam);
		}
		updateComAndMass(); // Update com.
	}


	// get max velocity
	double getVelMax(bool useSoc)
	{
		double vMax = 0;
		updateComAndMass();

		if (useSoc)
		{
			for (int Ball = 0; Ball < cNumBalls; Ball++)
			{
				if ((pos[Ball] - com).norm() < soc && vel[Ball].norm() > vMax)
				{
					vMax = vel[Ball].norm();
				}
			}
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
		}

		// Is vMax for some reason unreasonably small? Don't proceed. Probably a finished sim.
		if (vMax < 1e-10)
		{
			printf("\nMax velocity in system is less than 1e-10.\n");
			system("pause");
		}

		return vMax;
	}


	int getRmin()
	{
		int rMin = R[0];
		for (int Ball = 0; Ball < cNumBalls; Ball++)
		{
			if (R[Ball] < rMin)
			{
				rMin = R[Ball];
			}
		}
		return rMin;
	}


	int getMassMax()
	{
		int mMax = m[0];
		for (int Ball = 0; Ball < cNumBalls; Ball++)
		{
			if (m[Ball] > mMax)
			{
				mMax = m[Ball];
			}
		}
		return mMax;
	}

	/// Push all balls apart until elastic force < gravitational force (equilibrium).
	bool pushApart()
	{
		int issuesDetected = 0;
		double worstDiff = 0;

		for (int A = 0; A < cNumBalls; A++)
		{
			for (int B = A + 1; B < cNumBalls; B++)
			{
				// Check for Ball overlap.
				vector3d rVecab = pos[B] - pos[A];
				vector3d rVecba = -1 * rVecab;
				double dist = (rVecab).norm();
				double sumRaRb = R[A] + R[B];
				double overlap = sumRaRb - dist;
				double elasticForce = (-kin * overlap * .5 * (rVecab / dist)).norm();
				double gravForce = ((G * m[A] * m[B] / (dist * dist)) * (rVecab / dist)).norm();

				if (worstDiff < elasticForce / gravForce)
				{
					worstDiff = elasticForce / gravForce;
				}

				if (overlap > 0)// && elasticForce > gravForce)
				{
					double move = 0;

					(overlap * .5 > sumRaRb) ? move = sumRaRb : move = overlap * .5;

					issuesDetected += 1;

					if (R[B] <= R[A])
					{
						pos[B] += move * (rVecab / dist);
					}
					else
					{
						pos[A] += move * (rVecba / dist);
					}
				}
			}

			//std::cout << "Overlap: " << totalOverlap << "                        \r";
			//if (failed == attempts - 1) // Added the second part to speed up spatial constraint increase when there are clearly too many collisions for the space to be feasable.
			//{
			//	std::cout << "Failed. Re-randomizing \n";// << spaceRange << ". Increasing range " << spaceRangeIncrement << "cm^3.\n";
			//	//spaceRange += spaceRangeIncrement;
			//	failed = 0;
			//	for (int Ball = 0; Ball < cNumBalls; Ball++)
			//	{
			//		pos[Ball] = randSphericalVec(spaceRange, spaceRange, spaceRange); // Each time we fail and increase range, redistribute all balls randomly so we don't end up with big balls near mid and small balls outside.
			//	}
			//}
		}

		if (issuesDetected > 0)
		{
			std::cout << worstDiff << "                        \r";
			return false;
		}
		else
		{
			std::cout << "\nSuccess!\n";
			return true;
		}
	}
};