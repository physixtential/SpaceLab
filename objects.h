#include "vector3d.h"
#include "initializations.h"
#include <vector>
// new test commit 

#pragma once

// Balls
double* balls;
// Easy motion component reference in array structure:
constexpr unsigned int x = 0;
constexpr unsigned int y = 1;
constexpr unsigned int z = 2;
constexpr unsigned int vx = 3;
constexpr unsigned int vy = 4;
constexpr unsigned int vz = 5;
constexpr unsigned int ax = 6;
constexpr unsigned int ay = 7;
constexpr unsigned int az = 8;
constexpr unsigned int wx = 9;
constexpr unsigned int wy = 10;
constexpr unsigned int wz = 11;
constexpr unsigned int R = 12;
constexpr unsigned int m = 13;
constexpr unsigned int moi = 14;
// Therefore:
constexpr unsigned int numProps = 15;

// Distance between all balls
size_t

struct cluster
{
	vector3d com, momentum, angularMomentum;
	double m = 0, radius = 0, PE = 0, KE = 0;
	std::vector<ball> balls;

	void calcCom()
	{
		if (m > 0)
		{
			vector3d comNumerator = { 0, 0, 0 };
			for (int Ball = 0; Ball < balls.size(); Ball++)
			{
				comNumerator += balls[Ball].m * balls[Ball].pos;
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
		for (int Ball = 0; Ball < balls.size(); Ball++)
		{
			balls[Ball].vel += comRot.cross(balls[Ball].pos - com); // If I compute com of this cluster and subtract it from pos[Ball] I can do this without it being at origin.
			balls[Ball].w += comRot;
		}
	}

	// offset cluster
	void offset(double rad1, double rad2, double impactParam)
	{
		for (int Ball = 0; Ball < balls.size(); Ball++)
		{
			balls[Ball].pos.x += (rad1 + rad2) * cos(impactParam);
			balls[Ball].pos.y += (rad1 + rad2) * sin(impactParam);
		}
		calcCom(); // Update com.
	}

	void rotAll(char axis, double angle)
	{
		for (int Ball = 0; Ball < balls.size(); Ball++)
		{
			balls[Ball].pos = balls[Ball].pos.rot(axis, angle);
			balls[Ball].vel = balls[Ball].vel.rot(axis, angle);
			balls[Ball].w = balls[Ball].w.rot(axis, angle);
		}
	}

	// Initialize accelerations and energy calculations:
	void initConditions()
	{
		m = 0;
		KE = 0;
		PE = 0;
		momentum = { 0,0,0 };
		angularMomentum = { 0,0,0 };
		if (balls.size() > 1)
		{
			vector3d comNumerator = { 0, 0, 0 };
			for (int A = 0; A < balls.size(); A++)
			{
				balls[A].distances.resize(balls.size());
			}

			for (int A = 0; A < balls.size(); A++)
			{
				ball& a = balls[A];
				m += a.m;
				comNumerator += a.m * a.pos;

				for (int B = A + 1; B < balls.size(); B++)
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
		for (int Ball = 0; Ball < balls.size(); Ball++)
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
		for (int Ball = 0; Ball < balls.size(); Ball++)
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

	// Initialize accelerations and energy calculations:
	void initConditions()
	{
		mTotal = KE = PE = 0;
		momentum = angularMomentum = { 0,0,0 };
		vector3d comNumerator = { 0, 0, 0 };
		for (int A = 0; A < balls.size(); A++)
		{
			balls[A].distances.resize(balls.size());
		}

		for (int A = 0; A < balls.size(); A++)
		{
			ball& a = balls[A];
			mTotal += a.m;
			comNumerator += a.m * a.pos;

			for (int B = A + 1; B < balls.size(); B++)
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
		for (int Ball = 0; Ball < balls.size(); Ball++)
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
		for (int Ball = 0; Ball < balls.size(); Ball++)
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
		for (int Ball = 0; Ball < balls.size(); Ball++)
		{
			pTotal += balls[Ball].m * balls[Ball].vel;
			mass += balls[Ball].m;
		}
		for (int Ball = 0; Ball < balls.size(); Ball++)
		{
			balls[Ball].vel -= (pTotal / mass);
		}

		pTotal = { 0,0,0 };
		for (int Ball = 0; Ball < balls.size(); Ball++)
		{
			pTotal += balls[Ball].m * balls[Ball].vel;
		}
		std::cout << "\nCorrected momentum = " << pTotal.tostr() << std::endl;
	}
};

