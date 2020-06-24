#include "vector3d.h"
#include "initializations.h"
#include <vector>


#pragma once

// Balls
double* balls;
// Easy motion component reference in array structure:
constexpr unsigned int x_ = 0;
constexpr unsigned int y_ = 1;
constexpr unsigned int z_ = 2;
constexpr unsigned int vx_ = 3;
constexpr unsigned int vy_ = 4;
constexpr unsigned int vz_ = 5;
constexpr unsigned int vhx_ = 6;
constexpr unsigned int vhy_ = 7;
constexpr unsigned int vhz_ = 8;
constexpr unsigned int ax_ = 9;
constexpr unsigned int ay_ = 10;
constexpr unsigned int az_ = 11;
constexpr unsigned int wx_ = 12;
constexpr unsigned int wy_ = 13;
constexpr unsigned int wz_ = 14;
constexpr unsigned int R_ = 15;
constexpr unsigned int m_ = 16;
constexpr unsigned int moi_ = 17;
// Therefore:
constexpr unsigned int numProps = 18;

// Distance between all balls
size_t dist[(numBalls * numBalls / 2) - (numBalls / 2)]; // This is the number ball comparisons actually done.

struct cluster
{
	vector3d com, mom, angMom; // Can be vector3d because they only matter for writing out to file. Can process on host.
	double m = 0, radius = 0;
	double PE = 0, KE = 0;
	int numBalls = 1;
	double* balls;

	void calcCom()
	{
		if (m > 0)
		{
			vector3d comNumerator = { 0, 0, 0 };
			for (int Ball = 0; Ball < numBalls; Ball++)
			{
				int idx = Ball * numProps;
				vector3d posVec = { balls[idx + x_],balls[idx + y_],balls[idx + z_] };
				comNumerator += balls[idx + m_] * posVec;
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
		vector3d comRot = { spinX, spinY, spinZ }; // Rotation axis and magnitude
		for (int Ball = 0; Ball < numBalls; Ball++)
		{
			int idx = Ball * numProps;
			vector3d pos = { balls[idx + x_], balls[idx + y_], balls[idx + z_] };
			vector3d vel = { balls[idx + vx_], balls[idx + vy_], balls[idx + vz_] };
			vector3d cross = comRot.cross(pos - com);

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
			int idx = Ball * numProps;

			vector3d pos = { balls[idx + x_], balls[idx + y_], balls[idx + z_] };
			vector3d vel = { balls[idx + vx_], balls[idx + vy_], balls[idx + vz_] };
			vector3d w = { balls[idx + vx_], balls[idx + vy_], balls[idx + vz_] };

			pos = pos.rot(axis, angle);
			vel = vel.rot(axis, angle);
			w = w.rot(axis, angle);
		}
	}

	// Initialzie accelerations and energy calculations:
	void initConditions()
	{
		m = 0;
		KE = 0;
		PE = 0;
		mom = { 0,0,0 };
		angMom = { 0,0,0 };
		if (numBalls > 1) // Code below only necessary for effects between balls.
		{
			vector3d comNumerator = { 0, 0, 0 };

			for (int A = 0; A < numBalls; A++)
			{
				int idx = A * numProps;
				double& a = balls[idx];
				m += (a + m_);
				comNumerator += a.m * a.pos;

				for (int B = A + 1; B < numBalls; B++)
				{
					ball& b = balls[B];
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
						// Calculate force and torque for a:
						vector3d dVel = b.vel - a.vel;
						vector3d relativeVelOfA = (dVel)-((dVel).dot(rVecab)) * (rVecab / (dist * dist)) - a.w.cross(a.R / sumRaRb * rVecab) - b.w.cross(b.R / sumRaRb * rVecab);
						vector3d elasticForceOnA = -kin * overlap * .5 * (rVecab / dist);
						vector3d frictionForceOnA = { 0,0,0 };
						if (relativeVelOfA.norm() > 1e-14) // When relative velocity is very low, dividing its vector components by its magnitude below is unstable.
						{
							frictionForceOnA = mu * elasticForceOnA.norm() * (relativeVelOfA / relativeVelOfA.norm());
						}
						aTorque = (a.R / sumRaRb) * rVecab.cross(frictionForceOnA);

						// Calculate force and torque for b:
						dVel = a.vel - b.vel;
						vector3d relativeVelOfB = (dVel)-((dVel).dot(rVecba)) * (rVecba / (dist * dist)) - b.w.cross(b.R / sumRaRb * rVecba) - a.w.cross(a.R / sumRaRb * rVecba);
						vector3d elasticForceOnB = -kin * overlap * .5 * (rVecba / dist);
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
						PE += -G * a.m * b.m / dist + kin * pow((sumRaRb - dist) * .5, 2);
					}
					else
					{
						// No collision: Include gravity only:
						vector3d gravForceOnA = (G * a.m * b.m / pow(dist, 2)) * (rVecab / dist);
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
				angularMomentum += a.m * a.pos.cross(a.vel) + a.moi * a.w;
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
			angularMomentum = a.m * a.pos.cross(a.vel) + a.moi * a.w;
			radius = a.R;
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
		vector3d pTotal = { 0,0,0 };
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
	vector3d com, momentum, angularMomentum;
	double mTotal = 0, KE = 0, PE = 0, spaceRange = 0;
	std::vector<ball> balls;
	std::vector<cluster> clusters;

	// Initialzie accelerations and energy calculations:
	void initConditions()
	{
		mTotal = KE = PE = 0;
		momentum = angularMomentum = { 0,0,0 };
		vector3d comNumerator = { 0, 0, 0 };
		for (int A = 0; A < numBalls; A++)
		{
			balls[A].distances.reszie(numBalls);
		}

		for (int A = 0; A < numBalls; A++)
		{
			ball& a = balls[A];
			mTotal += a.m;
			comNumerator += a.m * a.pos;

			for (int B = A + 1; B < numBalls; B++)
			{
				ball& b = balls[B];
				double sumRaRb = a.R + b.R;
				double dist = (a.pos - b.pos).norm();
				vector3d rVecab = b.pos - a.pos;
				vector3d rVecba = a.pos - b.pos;

				// Check for collision between Ball and otherBall:
				double overlap = sumRaRb - dist;
				vector3d totalForce = { 0, 0, 0 };
				vector3d aTorque = { 0, 0, 0 };
				vector3d bTorque = { 0, 0, 0 };
				if (overlap > 0)
				{
					// Calculate force and torque for a:
					vector3d dVel = b.vel - a.vel;
					vector3d relativeVelOfA = (dVel)-((dVel).dot(rVecab)) * (rVecab / (dist * dist)) - a.w.cross(a.R / sumRaRb * rVecab) - b.w.cross(b.R / sumRaRb * rVecab);
					vector3d elasticForceOnA = -kin * overlap * .5 * (rVecab / dist);
					vector3d frictionForceOnA = { 0,0,0 };
					if (relativeVelOfA.norm() > 1e-14) // When relative velocity is very low, dividing its vector components by its magnitude below is unstable.
					{
						frictionForceOnA = mu * elasticForceOnA.norm() * (relativeVelOfA / relativeVelOfA.norm());
					}
					aTorque = (a.R / sumRaRb) * rVecab.cross(frictionForceOnA);

					// Calculate force and torque for b:
					dVel = a.vel - b.vel;
					vector3d relativeVelOfB = (dVel)-((dVel).dot(rVecba)) * (rVecba / (dist * dist)) - b.w.cross(b.R / sumRaRb * rVecba) - a.w.cross(a.R / sumRaRb * rVecba);
					vector3d elasticForceOnB = -kin * overlap * .5 * (rVecba / dist);
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
					PE += -G * a.m * b.m / dist + kin * pow((sumRaRb - dist) * .5, 2);
				}
				else
				{
					// No collision: Include gravity only:
					vector3d gravForceOnA = (G * a.m * b.m / pow(dist, 2)) * (rVecab / dist);
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
			angularMomentum += a.m * a.pos.cross(a.vel) + a.moi * a.w;
		}
		com = comNumerator / mTotal;
	}

	void calcComAndMass()
	{
		vector3d comNumerator = { 0, 0, 0 };
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
		vector3d pTotal = { 0,0,0 };
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
		vector3d pTotal = { 0,0,0 };
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

