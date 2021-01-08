// There are 4 important steps to creating a new random cluster:
// 1- populateGroup(size) the cluster
// 3- initConditions() to set correct first step physics
// 4- freeMemory() to clear the arrays from memory when done.

struct ballGroup
{
	int cNumBalls = 0;

	double3
		com = make_double3(0, 0, 0),
		mom = make_double3(0, 0, 0),
		angMom = make_double3(0, 0, 0); // Can be double3 because they only matter for writing out to file. Can process on host.
	double mTotal = 0, radius = 0;
	double PE = 0, KE = 0;

	double* distances = 0;

	double3* pos = 0;
	double3* vel = 0;
	double3* velh = 0;
	double3* acc = 0;
	double3* w = 0;
	double* R = 0;
	double* m = 0;
	double* moi = 0;

	// Allocate ball property arrays.
	void populateGroup(int nBalls)
	{
		cNumBalls = nBalls;

		distances = new double[(cNumBalls * cNumBalls / 2) - (cNumBalls / 2)];

		pos = new double3[cNumBalls];
		vel = new double3[cNumBalls];
		velh = new double3[cNumBalls];
		acc = new double3[cNumBalls];
		w = new double3[cNumBalls];
		R = new double[cNumBalls];
		m = new double[cNumBalls];
		moi = new double[cNumBalls];
	}

	// Deallocate heap memory.
	void freeMemory()
	{
		delete[] distances;
		delete[] pos;
		delete[] vel;
		delete[] velh;
		delete[] acc;
		delete[] w;
		delete[] R;
		delete[] m;
		delete[] moi;
	}

	double updateRadius()
	{
		radius = 0;
		for (int Ball = 0; Ball < cNumBalls; Ball++)
		{
			double dist = length(pos[Ball] - com);
			if (dist > radius)
			{
				radius = dist;
			}
		}
		return radius;
	}

	double3 updateCom()
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
			double3 comNumerator = { 0, 0, 0 };
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
		}
	}

	void clusToOrigin()
	{
		updateCom();

		for (int Ball = 0; Ball < cNumBalls; Ball++)
		{
			pos[Ball] -= com;
		}
		updateCom();
	}

	// Set velocity of all balls such that the cluster spins:
	void comSpinner(double spinX, double spinY, double spinZ)
	{
		double3 comRot = make_double3(spinX, spinY, spinZ); // Rotation axis and magnitude
		for (int Ball = 0; Ball < cNumBalls; Ball++)
		{
			vel[Ball] += cross(comRot, (pos[Ball] - com));
			w[Ball] += comRot;
		}
	}

	// Kick projectile at target
	void kick(double vx)
	{
		for (int Ball = 0; Ball < cNumBalls; Ball++)
		{
			vel[Ball].x += vx;
		}
		updateCom(); // Update com.
	}

	void rotAll(char axis, double angle)
	{
		for (int Ball = 0; Ball < cNumBalls; Ball++)
		{
			pos[Ball] = rot(axis, angle, pos[Ball]);
			vel[Ball] = rot(axis, angle, vel[Ball]);
			w[Ball] = rot(axis, angle, w[Ball]);
		}
	}



	// Initialzie accelerations and energy calculations:
	void initConditions(int cNumBalls)
	{
		mTotal = 0;
		KE = 0;
		PE = 0;
		mom = make_double3(0, 0, 0);
		angMom = make_double3(0, 0, 0);
		if (cNumBalls > 1) // Code below only necessary for effects between balls.
		{
			double3 comNumerator = { 0, 0, 0 };

			for (int A = 1; A < cNumBalls; A++)
			{
				mTotal += m[A];
				comNumerator += m[A] * pos[A];

				for (int B = 0; B < A; B++)
				{
					double sumRaRb = R[A] + R[B];
					double dist = length(pos[A] - pos[B]);
					double3 rVecab = pos[B] - pos[A];
					double3 rVecba = -1 * rVecab;

					// Check for collision between Ball and otherBall:
					double overlap = sumRaRb - dist;
					double3 totalForce = { 0, 0, 0 };
					double3 aTorque = { 0, 0, 0 };
					double3 bTorque = { 0, 0, 0 };

					// Check for collision between Ball and otherBall.
					if (overlap > 0)
					{
						// Calculate force and torque for a:
						double3 dVel = vel[B] - vel[A];
						double3 relativeVelOfA = dVel - dot(dVel, rVecab) * (rVecab / (dist * dist)) - cross(w[A], R[A] / sumRaRb * rVecab) - cross(w[B], R[B] / sumRaRb * rVecab);
						double3 elasticForceOnA = -kin * overlap * .5 * (rVecab / dist);
						double3 frictionForceOnA = { 0,0,0 };
						if (length(relativeVelOfA) > 1e-12) // When relative velocity is very low, dividing its vector components by its magnitude below is unstable.
						{
							frictionForceOnA = mu * length(elasticForceOnA) * (relativeVelOfA / length(relativeVelOfA));
						}
						aTorque = (R[A] / sumRaRb) * cross(rVecab, frictionForceOnA);

						// Calculate force and torque for b:
						dVel = vel[A] - vel[B];
						double3 relativeVelOfB = dVel - dot(dVel, rVecba) * (rVecba / (dist * dist)) - cross(w[B], R[B] / sumRaRb * rVecba) - cross(w[A], R[A] / sumRaRb * rVecba);
						double3 elasticForceOnB = -kin * overlap * .5 * (rVecba / dist);
						double3 frictionForceOnB = { 0,0,0 };
						if (length(relativeVelOfB) > 1e-12)
						{
							frictionForceOnB = mu * length(elasticForceOnB) * (relativeVelOfB / length(relativeVelOfB));
						}
						bTorque = (R[B] / sumRaRb) * cross(rVecba, frictionForceOnB);

						double3 gravForceOnA = (G * m[A] * m[B] / pow(dist, 2)) * (rVecab / dist);
						totalForce = gravForceOnA + elasticForceOnA + frictionForceOnA;
						w[A] += aTorque / moi[A] * dt;
						w[B] += bTorque / moi[B] * dt;
						PE += -G * m[A] * m[B] / dist + kin * pow((sumRaRb - dist) * .5, 2);
					}
					else
					{
						// No collision: Include gravity only:
						double3 gravForceOnA = (G * m[A] * m[B] / pow(dist, 2)) * (rVecab / dist);
						totalForce = gravForceOnA;
						PE += -G * m[A] * m[B] / dist;
					}
					// Newton's equal and opposite forces applied to acceleration of each ball:
					acc[A] += totalForce / m[A];
					acc[B] -= totalForce / m[B];
					int e = (A * (A - 1) * .5) + B;
					distances[e] = dist;
				}
				KE += .5 * m[A] * dot(vel[A], vel[A]) + .5 * moi[A] * dot(w[A], w[A]);
				mom += m[A] * vel[A];
				angMom += m[A] * cross(pos[A], vel[A]) + moi[A] * w[A];
			}
			com = comNumerator / mTotal;
		}
		else // For the case of just one ball:
		{
			mTotal = m[0];
			PE = 0;
			KE = .5 * m[0] * dot(vel[0], vel[0]) + .5 * moi[0] * dot(w[0], w[0]);
			mom = m[0] * vel[0];
			angMom = m[0] * cross(pos[0], vel[0]) + moi[0] * w[0];
			radius = R[0];
		}
	}

	// Kick projectile at target
	void kick(double vx, double vy, double vz)
	{
		for (int Ball = 0; Ball < cNumBalls; Ball++)
		{
			vel[Ball] += make_double3(vx, vy, vz);
		}
	}

	void checkMomentum()
	{
		double3 pTotal = { 0,0,0 };
		for (int Ball = 0; Ball < cNumBalls; Ball++)
		{
			pTotal += m[Ball] * vel[Ball];
		}
		printf("Cluster Momentum Check: %.2e, %.2e, %.2e\n", pTotal.x, pTotal.y, pTotal.z);
	}

	// offset cluster
	void offset(double rad1, double rad2, double impactParam)
	{
		for (int Ball = 0; Ball < cNumBalls; Ball++)
		{
			pos[Ball].x += (rad1 + rad2) * cos(impactParam);
			pos[Ball].y += (rad1 + rad2) * sin(impactParam);
		}
		updateCom(); // Update com.
	}
};