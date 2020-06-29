#include "cuVectorMath.h"
#include "initializations.h"
#include <vector>


#pragma once

// Distance between all balls
//size_t dist[(NumBalls * NumBalls / 2) - (NumBalls / 2)]; // This is the number ball comparisons actually done.

struct cluster
{
	int cNumBalls = 0;

	double3 com, mom, angMom; // Can be double3 because they only matter for writing out to file. Can process on host.
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

	void calcCom()
	{
		if (mTotal > 0)
		{
			double3 comNumerator = { 0, 0, 0 };
			for (int Ball = 0; Ball < cNumBalls; Ball++)
			{
				comNumerator += m[Ball] * pos[Ball];
			}
			com = comNumerator / mTotal;
		}
		else
		{
			std::cout << "Mass of cluster is zero...\n";
		}
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

	// offset cluster
	void offset(double rad1, double rad2, double impactParam)
	{
		for (int Ball = 0; Ball < cNumBalls; Ball++)
		{
			pos[Ball].x += (rad1 + rad2) * cos(impactParam);
			pos[Ball].y += (rad1 + rad2) * sin(impactParam);
		}
		calcCom(); // Update com.
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

	// Initialzie accelerations and energy calculations:
	void initConditions(int cNumBalls)
	{
		distances = new double[(cNumBalls * cNumBalls / 2) - (cNumBalls / 2)];

		pos = new double3[cNumBalls];
		vel = new double3[cNumBalls];
		velh = new double3[cNumBalls];
		acc = new double3[cNumBalls];
		w = new double3[cNumBalls];
		R = new double[cNumBalls];
		m = new double[cNumBalls];
		moi = new double[cNumBalls];

		mTotal = 0;
		KE = 0;
		PE = 0;
		mom = make_double3(0, 0, 0);
		angMom = make_double3(0, 0, 0);
		if (cNumBalls > 1) // Code below only necessary for effects between balls.
		{
			double3 comNumerator = { 0, 0, 0 };

			for (int A = 0; A < cNumBalls; A++)
			{
				mTotal += m[A];
				comNumerator += m[A] * pos[A];

				for (int B = A + 1; B < cNumBalls; B++)
				{
					double sumRaRb = R[A] + R[B];
					double dist = mag(pos[A] - pos[B]);
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
						double3 dVel = vel[B] - vel[A];
						double3 relativeVelOfA = dVel - dot(dVel, rVecab) * (rVecab / (dist * dist)) - cross(w[A], R[A] / sumRaRb * rVecab) - cross(w[B], R[B] / sumRaRb * rVecab);
						double3 elasticForceOnA = -kin * overlap * .5 * (rVecab / dist);
						double3 frictionForceOnA = { 0,0,0 };
						if (mag(relativeVelOfA) > 1e-12) // When relative velocity is very low, dividing its vector components by its magnitude below is unstable.
						{
							frictionForceOnA = mu * mag(elasticForceOnA) * (relativeVelOfA / mag(relativeVelOfA));
						}
						aTorque = (R[A] / sumRaRb) * cross(rVecab, frictionForceOnA);

						// Calculate force and torque for b:
						dVel = vel[A] - vel[B];
						double3 relativeVelOfB = dVel - dot(dVel, rVecba) * (rVecba / (dist * dist)) - cross(w[B], R[B] / sumRaRb * rVecba) - cross(w[A], R[A] / sumRaRb * rVecba);
						double3 elasticForceOnB = -kin * overlap * .5 * (rVecba / dist);
						double3 frictionForceOnB = { 0,0,0 };
						if (mag(relativeVelOfB) > 1e-12)
						{
							frictionForceOnB = mu * mag(elasticForceOnB) * (relativeVelOfB / mag(relativeVelOfB));
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
					a.distances[B] = b.distances[A] = dist;
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
};

//struct universe
//{
//	double3 com, momentum, angularMomentum;
//	double mTotal = 0, KE = 0, PE = 0, spaceRange = 0;
//	std::vector<ball> balls;
//	std::vector<cluster> clusters;
//
//	// Initialzie accelerations and energy calculations:
//	void initConditions()
//	{
//		mTotal = KE = PE = 0;
//		momentum = angularMomentum = { 0,0,0 };
//		double3 comNumerator = { 0, 0, 0 };
//		for (int A = 0; A < cNumBalls; A++)
//		{
//			balls[A].distances.reszie(cNumBalls);
//		}
//
//		for (int A = 0; A < cNumBalls; A++)
//		{
//			ball& a = balls[A];
//			mTotal += m[A];
//			comNumerator += m[A] * pos[A];
//
//			for (int B = A + 1; B < cNumBalls; B++)
//			{
//				ball& b = balls[B];
//				double sumRaRb = R[A] + R[B];
//				double dist = (pos[A] - pos[B]).norm();
//				double3 rVecab = pos[B] - pos[A];
//				double3 rVecba = pos[A] - pos[B];
//
//				// Check for collision between Ball and otherBall:
//				double overlap = sumRaRb - dist;
//				double3 totalForce = { 0, 0, 0 };
//				double3 aTorque = { 0, 0, 0 };
//				double3 bTorque = { 0, 0, 0 };
//				if (overlap > 0)
//				{
//					// Calculate force and torque for a:
//					double3 dVel = vel[B] - vel[A];
//					double3 relativeVelOfA = (dVel)-((dVel).dot(rVecab)) * (rVecab / (dist * dist)) - a.w.cross(R[A] / sumRaRb * rVecab) - b.w.cross(R[B] / sumRaRb * rVecab);
//					double3 elasticForceOnA = -kin * overlap * .5 * (rVecab / dist);
//					double3 frictionForceOnA = { 0,0,0 };
//					if (relativeVelOfA.norm() > 1e-14) // When relative velocity is very low, dividing its vector components by its magnitude below is unstable.
//					{
//						frictionForceOnA = mu * elasticForceOnA.norm() * (relativeVelOfA / relativeVelOfA.norm());
//					}
//					aTorque = (R[A] / sumRaRb) * rVecab.cross(frictionForceOnA);
//
//					// Calculate force and torque for b:
//					dVel = vel[A] - vel[B];
//					double3 relativeVelOfB = (dVel)-((dVel).dot(rVecba)) * (rVecba / (dist * dist)) - b.w.cross(R[B] / sumRaRb * rVecba) - a.w.cross(R[A] / sumRaRb * rVecba);
//					double3 elasticForceOnB = -kin * overlap * .5 * (rVecba / dist);
//					double3 frictionForceOnB = { 0,0,0 };
//					if (relativeVelOfB.norm() > 1e-14)
//					{
//						frictionForceOnB = mu * elasticForceOnB.norm() * (relativeVelOfB / relativeVelOfB.norm());
//					}
//					bTorque = (R[B] / sumRaRb) * rVecba.cross(frictionForceOnB);
//
//					double3 gravForceOnA = (G * m[A] * m[B] / pow(dist, 2)) * (rVecab / dist);
//					totalForce = gravForceOnA + elasticForceOnA + frictionForceOnA;
//					a.w += aTorque / moi[A] * dt;
//					b.w += bTorque / moi[B] * dt;
//					PE += -G * m[A] * m[B] / dist + kin * pow((sumRaRb - dist) * .5, 2);
//				}
//				else
//				{
//					// No collision: Include gravity only:
//					double3 gravForceOnA = (G * m[A] * m[B] / pow(dist, 2)) * (rVecab / dist);
//					totalForce = gravForceOnA;
//					PE += -G * m[A] * m[B] / dist;
//				}
//				// Newton's equal and opposite forces applied to acceleration of each ball:
//				a.acc += totalForce / m[A];
//				b.acc -= totalForce / m[B];
//				a.distances[B] = b.distances[A] = dist;
//			}
//			KE += .5 * m[A] * vel[A].normsquared() + .5 * moi[A] * a.w.normsquared();
//			momentum += m[A] * vel[A];
//			angularMomentum += m[A] * pos[A].cross(vel[A]) + moi[A] * a.w;
//		}
//		com = comNumerator / mTotal;
//	}
//
//	void calcComAndMass()
//	{
//		double3 comNumerator = { 0, 0, 0 };
//		mTotal = 0;
//		for (int Ball = 0; Ball < cNumBalls; Ball++)
//		{
//			mTotal += balls[Ball].m;
//			comNumerator += balls[Ball].m * balls[Ball].pos;
//		}
//		com = comNumerator / mTotal;
//	}
//
//	void checkMomentum()
//	{
//		double3 pTotal = { 0,0,0 };
//		double mass = 0;
//		for (int Ball = 0; Ball < cNumBalls; Ball++)
//		{
//			pTotal += balls[Ball].m * balls[Ball].vel;
//			mass += balls[Ball].m;
//		}
//		printf("Universe Momentum Check: %.2e, %.2e, %.2e\n", pTotal.x, pTotal.y, pTotal.z);
//	}
//
//	void zeroMomentum()
//	{
//		// Something about this is wrong. It is not zeroing momentum.
//		double3 pTotal = { 0,0,0 };
//		double mass = 0;
//		for (int Ball = 0; Ball < cNumBalls; Ball++)
//		{
//			pTotal += balls[Ball].m * balls[Ball].vel;
//			mass += balls[Ball].m;
//		}
//		for (int Ball = 0; Ball < cNumBalls; Ball++)
//		{
//			balls[Ball].vel -= (pTotal / mass);
//		}
//
//		pTotal = { 0,0,0 };
//		for (int Ball = 0; Ball < cNumBalls; Ball++)
//		{
//			pTotal += balls[Ball].m * balls[Ball].vel;
//		}
//		std::cout << "\nCorrected momentum = " << pTotal.tostr() << std::endl;
//	}
//};

