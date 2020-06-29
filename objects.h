#include "cuVectorMath.h"
#include "initializations.h"
#include <vector>


#pragma once

// Distance between all balls
//size_t dist[(numBalls * numBalls / 2) - (numBalls / 2)]; // This is the number ball comparisons actually done.

struct cluster
{
	int numBalls;
	double3 com, mom, angMom; // Can be double3 because they only matter for writing out to file. Can process on host.
	double mTotal = 0, radius = 0;
	double PE = 0, KE = 0;

	double3* pos = 0;
	double3* vel = 0;
	double3* velh = 0;
	double3* acc = 0;
	double3* w = 0;
	double* R = 0;
	double* m = 0;
	double* moi = 0;

	void calcCom()
	{
		if (m > 0)
		{
			double3 comNumerator = { 0, 0, 0 };
			for (int Ball = 0; Ball < numBalls; Ball++)
			{
				int idx = Ball * numProps;
				comNumerator += balls3[idx + *posVec;
			}
			com = comNumerator / m;
		}
		else
		{
			std::cout << "Mass of cluster is zero...\n";
		}
	}

	// Set velocity of all balls such that the cluster spins:
	void comSpinner(double spinX, double spinY, double spinZ)
	{
		double3 comRot = { spinX, spinY, spinZ }; // Rotation axis and magnitude
		for (int Ball = 0; Ball < numBalls; Ball++)
		{
			int idx = Ball * numProps;
			double3 pos = { balls[idx + x_], balls[idx + y_], balls[idx + z_] };
			double3 vel = { balls[idx + vx_], balls[idx + vy_], balls[idx + vz_] };
			double3 cross = comRot.cross(pos - com);

			balls[idx + vx_] += cross.x;
			balls[idx + vy_] += cross.y;
			balls[idx + vz_] += cross.z;

			balls[idx + wx_] += comRot.x;
			balls[idx + wx_] += comRot.y;
			balls[idx + wx_] += comRot.z;
		}
	}

	// offset cluster
	void offset(double rad1, double rad2, double impactParam)
	{
		for (int Ball = 0; Ball < numBalls; Ball++)
		{
			int idx = Ball * numProps;

			balls[idx + x_] += (rad1 + rad2) * cos(impactParam);
			balls[idx + y_] += (rad1 + rad2) * sin(impactParam);
		}
		calcCom(); // Update com.
	}

	void rotAll(char axis, double angle)
	{
		for (int Ball = 0; Ball < numBalls; Ball++)
		{
			pos = pos.rot(axis, angle);
			vel = vel.rot(axis, angle);
			w = w.rot(axis, angle);
		}
	}

	// Initialzie accelerations and energy calculations:
	void initConditions()
	{
		mTotal = 0;
		KE = 0;
		PE = 0;
		mom = make_double3(0, 0, 0);
		angMom = make_double3(0, 0, 0);
		if (numBalls > 1) // Code below only necessary for effects between balls.
		{
			double3 comNumerator = { 0, 0, 0 };

			for (int A = 0; A < numBalls; A++)
			{
				mTotal += m[A];
				comNumerator += m[A] * pos[A];

				for (int B = A + 1; B < numBalls; B++)
				{
					double sumRaRb = R[A] + R[B];
					double dist = (pos[A] - pos[B]).norm();
					double3 rVecab = pos[B] - pos[A];
					double3 rVecba = pos[A] - pos[B];

					// Check for collision between Ball and otherBall:
					double overlap = sumRaRb - dist;
					double3 totalForce = { 0, 0, 0 };
					double3 aTorque = { 0, 0, 0 };
					double3 bTorque = { 0, 0, 0 };

					// Check for collision between Ball and otherBall.
					if (overlap > 0)
					{
						// Calculate force and torque for a:
						double3 dVel = b.vel - a.vel;
						double3 relativeVelOfA = (dVel)-((dVel).dot(rVecab)) * (rVecab / (dist * dist)) - a.w.cross(R[A] / sumRaRb * rVecab) - b.w.cross(R[B] / sumRaRb * rVecab);
						double3 elasticForceOnA = -kin * overlap * .5 * (rVecab / dist);
						double3 frictionForceOnA = { 0,0,0 };
						if (relativeVelOfA.norm() > 1e-14) // When relative velocity is very low, dividing its vector components by its magnitude below is unstable.
						{
							frictionForceOnA = mu * elasticForceOnA.norm() * (relativeVelOfA / relativeVelOfA.norm());
						}
						aTorque = (R[A] / sumRaRb) * rVecab.cross(frictionForceOnA);

						// Calculate force and torque for b:
						dVel = a.vel - b.vel;
						double3 relativeVelOfB = (dVel)-((dVel).dot(rVecba)) * (rVecba / (dist * dist)) - b.w.cross(R[B] / sumRaRb * rVecba) - a.w.cross(R[A] / sumRaRb * rVecba);
						double3 elasticForceOnB = -kin * overlap * .5 * (rVecba / dist);
						double3 frictionForceOnB = { 0,0,0 };
						if (relativeVelOfB.norm() > 1e-14)
						{
							frictionForceOnB = mu * elasticForceOnB.norm() * (relativeVelOfB / relativeVelOfB.norm());
						}
						bTorque = (R[B] / sumRaRb) * rVecba.cross(frictionForceOnB);

						double3 gravForceOnA = (G * a.m * b.m / pow(dist, 2)) * (rVecab / dist);
						totalForce = gravForceOnA + elasticForceOnA + frictionForceOnA;
						a.w += aTorque / a.moi * dt;
						b.w += bTorque / b.moi * dt;
						PE += -G * a.m * b.m / dist + kin * pow((sumRaRb - dist) * .5, 2);
					}
					else
					{
						// No collision: Include gravity only:
						double3 gravForceOnA = (G * a.m * b.m / pow(dist, 2)) * (rVecab / dist);
						totalForce = gravForceOnA;
						PE += -G * a.m * b.m / dist;
					}
					// Newton's equal and opposite forces applied to acceleration of each ball:
					a.acc += totalForce / a.m;
					b.acc -= totalForce / b.m;
					a.distances[B] = b.distances[A] = dist;
				}
				KE += .5 * a.m * a.vel.normsquared() + .5 * a.moi * a.w.normsquared();
				momentum += a.m * a.vel;
				angularMomentum += a.m * pos[A].cross(a.vel) + a.moi * a.w;
			}
			com = comNumerator / m;
		}
		else // For the case of just one ball:
		{
			ball& a = balls[0];
			m = a.m;
			PE = 0;
			KE = .5 * a.m * a.vel.normsquared() + .5 * a.moi * a.w.normsquared();
			momentum = a.m * a.vel;
			angularMomentum = a.m * pos[A].cross(a.vel) + a.moi * a.w;
			radius = R[A];
		}
	}

	// Kick projectile at target
	void kick(double vx, double vy, double vz)
	{
		for (int Ball = 0; Ball < numBalls; Ball++)
		{
			balls[Ball].vel.x += vx;
			balls[Ball].vel.y += vy;
			balls[Ball].vel.z += vz;
		}
	}

	void checkMomentum()
	{
		double3 pTotal = { 0,0,0 };
		double mass = 0;
		for (int Ball = 0; Ball < numBalls; Ball++)
		{
			pTotal += balls[Ball].m * balls[Ball].vel;
			mass += balls[Ball].m;
		}
		printf("Cluster Momentum Check: %.2e, %.2e, %.2e\n", pTotal.x, pTotal.y, pTotal.z);
	}
};

struct universe
{
	double3 com, momentum, angularMomentum;
	double mTotal = 0, KE = 0, PE = 0, spaceRange = 0;
	std::vector<ball> balls;
	std::vector<cluster> clusters;

	// Initialzie accelerations and energy calculations:
	void initConditions()
	{
		mTotal = KE = PE = 0;
		momentum = angularMomentum = { 0,0,0 };
		double3 comNumerator = { 0, 0, 0 };
		for (int A = 0; A < numBalls; A++)
		{
			balls[A].distances.reszie(numBalls);
		}

		for (int A = 0; A < numBalls; A++)
		{
			ball& a = balls[A];
			mTotal += a.m;
			comNumerator += a.m * pos[A];

			for (int B = A + 1; B < numBalls; B++)
			{
				ball& b = balls[B];
				double sumRaRb = R[A] + R[B];
				double dist = (pos[A] - pos[B]).norm();
				double3 rVecab = pos[B] - pos[A];
				double3 rVecba = pos[A] - pos[B];

				// Check for collision between Ball and otherBall:
				double overlap = sumRaRb - dist;
				double3 totalForce = { 0, 0, 0 };
				double3 aTorque = { 0, 0, 0 };
				double3 bTorque = { 0, 0, 0 };
				if (overlap > 0)
				{
					// Calculate force and torque for a:
					double3 dVel = b.vel - a.vel;
					double3 relativeVelOfA = (dVel)-((dVel).dot(rVecab)) * (rVecab / (dist * dist)) - a.w.cross(R[A] / sumRaRb * rVecab) - b.w.cross(R[B] / sumRaRb * rVecab);
					double3 elasticForceOnA = -kin * overlap * .5 * (rVecab / dist);
					double3 frictionForceOnA = { 0,0,0 };
					if (relativeVelOfA.norm() > 1e-14) // When relative velocity is very low, dividing its vector components by its magnitude below is unstable.
					{
						frictionForceOnA = mu * elasticForceOnA.norm() * (relativeVelOfA / relativeVelOfA.norm());
					}
					aTorque = (R[A] / sumRaRb) * rVecab.cross(frictionForceOnA);

					// Calculate force and torque for b:
					dVel = a.vel - b.vel;
					double3 relativeVelOfB = (dVel)-((dVel).dot(rVecba)) * (rVecba / (dist * dist)) - b.w.cross(R[B] / sumRaRb * rVecba) - a.w.cross(R[A] / sumRaRb * rVecba);
					double3 elasticForceOnB = -kin * overlap * .5 * (rVecba / dist);
					double3 frictionForceOnB = { 0,0,0 };
					if (relativeVelOfB.norm() > 1e-14)
					{
						frictionForceOnB = mu * elasticForceOnB.norm() * (relativeVelOfB / relativeVelOfB.norm());
					}
					bTorque = (R[B] / sumRaRb) * rVecba.cross(frictionForceOnB);

					double3 gravForceOnA = (G * a.m * b.m / pow(dist, 2)) * (rVecab / dist);
					totalForce = gravForceOnA + elasticForceOnA + frictionForceOnA;
					a.w += aTorque / a.moi * dt;
					b.w += bTorque / b.moi * dt;
					PE += -G * a.m * b.m / dist + kin * pow((sumRaRb - dist) * .5, 2);
				}
				else
				{
					// No collision: Include gravity only:
					double3 gravForceOnA = (G * a.m * b.m / pow(dist, 2)) * (rVecab / dist);
					totalForce = gravForceOnA;
					PE += -G * a.m * b.m / dist;
				}
				// Newton's equal and opposite forces applied to acceleration of each ball:
				a.acc += totalForce / a.m;
				b.acc -= totalForce / b.m;
				a.distances[B] = b.distances[A] = dist;
			}
			KE += .5 * a.m * a.vel.normsquared() + .5 * a.moi * a.w.normsquared();
			momentum += a.m * a.vel;
			angularMomentum += a.m * pos[A].cross(a.vel) + a.moi * a.w;
		}
		com = comNumerator / mTotal;
	}

	void calcComAndMass()
	{
		double3 comNumerator = { 0, 0, 0 };
		mTotal = 0;
		for (int Ball = 0; Ball < numBalls; Ball++)
		{
			mTotal += balls[Ball].m;
			comNumerator += balls[Ball].m * balls[Ball].pos;
		}
		com = comNumerator / mTotal;
	}

	void checkMomentum()
	{
		double3 pTotal = { 0,0,0 };
		double mass = 0;
		for (int Ball = 0; Ball < numBalls; Ball++)
		{
			pTotal += balls[Ball].m * balls[Ball].vel;
			mass += balls[Ball].m;
		}
		printf("Universe Momentum Check: %.2e, %.2e, %.2e\n", pTotal.x, pTotal.y, pTotal.z);
	}

	void zeroMomentum()
	{
		// Something about this is wrong. It is not zeroing momentum.
		double3 pTotal = { 0,0,0 };
		double mass = 0;
		for (int Ball = 0; Ball < numBalls; Ball++)
		{
			pTotal += balls[Ball].m * balls[Ball].vel;
			mass += balls[Ball].m;
		}
		for (int Ball = 0; Ball < numBalls; Ball++)
		{
			balls[Ball].vel -= (pTotal / mass);
		}

		pTotal = { 0,0,0 };
		for (int Ball = 0; Ball < numBalls; Ball++)
		{
			pTotal += balls[Ball].m * balls[Ball].vel;
		}
		std::cout << "\nCorrected momentum = " << pTotal.tostr() << std::endl;
	}
};

