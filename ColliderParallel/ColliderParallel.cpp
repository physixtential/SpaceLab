#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <sstream>
#include <vector>
#include <algorithm>
#include <execution>
#include <omp.h>

#include "../vector3d.hpp"
#include "../initializations.hpp"

// String buffers to hold data in memory until worth writing to file:
std::stringstream ballBuffer;
std::stringstream energyBuffer;

// These are used within simOneStep to keep track of time.
// They need to survive outside its scope, and I don't want to have to pass them all.
const time_t start = time(nullptr);        // For end of program analysis
time_t startProgress; // For progress reporting (gets reset)
time_t lastWrite;     // For write control (gets reset)
bool writeStep;       // This prevents writing to file every step (which is slow).


//ballGroup O(path, projectileName, targetName, vCustom); // Collision
//ballGroup O(path, targetName, 0); // Continue
//ballGroup O(genBalls, true, vCustom); // Generate


struct p
{
	vector3d
		pos,
		vel,
		velh,
		acc,
		w,
		wh,
		aacc;

	double
		R = 0,
		m = 0,
		moi = 0;
};

struct p_pair
{
	p* A;
	p* B;
	double dist;
	p_pair(p* a, p* b, double d) : A(a), B(b), dist(d) {} // Init all
	p_pair(p* a, p* b) : A(a), B(b), dist(-1.0) {} // init pairs and set D to illogical distance
	//p_pair(const p_pair& in_pair) : A(in_pair.A), B(in_pair.B), dist(in_pair.dist) {} // Copy constructor
	p_pair() = default;
};

struct p_group
{
	int n; // Number of particles in the group

	std::vector<p> p_group; // Group of particles

	double U; // Potential energy
	double T; // Kinetic Energy
};

void update_kinematics(p& p)
{
	// Update velocity half step:
	p.velh = p.vel + .5 * p.acc * dt;

	// Update angular velocity half step:
	p.wh = p.w + .5 * p.aacc * dt;

	// Update position:
	p.pos += p.velh * dt;

	// Reinitialize acceleration to be recalculated:
	p.acc = { 0, 0, 0 };

	// Reinitialize angular acceleration to be recalculated:
	p.aacc = { 0, 0, 0 };
}

void compute_acceleration(p_pair& pair)
{
	const double Ra = pair.A->R;
	const double Rb = pair.B->R;
	const double sumRaRb = Ra + Rb;
	vector3d rVec = pair.B->pos - pair.A->pos; // Start with rVec from a to b.
	const double dist = (rVec).norm();
	vector3d totalForce;

	// Check for collision between Ball and otherBall:
	double overlap = sumRaRb - dist;

	double oldDist = pair.dist;

	// Check for collision between Ball and otherBall.
	if (overlap > 0)
	{
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

		// Cohesion:
		// h is the "separation" of the particles at particle radius - maxOverlap.
		// This allows particles to be touching while under vdwForce.
		const double h = maxOverlap * 1.01 - overlap;
		const double h2 = h * h;
		const double twoRah = 2 * Ra * h;
		const double twoRbh = 2 * Rb * h;
		const vector3d vdwForce =
			Ha / 6 *
			64 * Ra * Ra * Ra * Rb * Rb * Rb *
			(h + Ra + Rb) /
			(
				(h2 + twoRah + twoRbh) *
				(h2 + twoRah + twoRbh) *
				(h2 + twoRah + twoRbh + 4 * Ra * Rb) *
				(h2 + twoRah + twoRbh + 4 * Ra * Rb)
				) *
			rVec.normalized();

		// Elastic a:
		vector3d elasticForce = -k * overlap * .5 * (rVec / dist);

		// Friction a:
		vector3d dVel = pair.B->vel - pair.A->vel;
		vector3d frictionForce = { 0, 0, 0 };
		const vector3d relativeVelOfA = dVel - dVel.dot(rVec) * (rVec / (dist * dist)) - pair.A->w.cross(pair.A->R / sumRaRb * rVec) - pair.B->w.cross(pair.B->R / sumRaRb * rVec);
		double relativeVelMag = relativeVelOfA.norm();
		if (relativeVelMag > 1e-10) // When relative velocity is very low, dividing its vector components by its magnitude below is unstable.
		{
			frictionForce = mu * (elasticForce.norm() + vdwForce.norm()) * (relativeVelOfA / relativeVelMag);
		}

		// Torque a:
		const vector3d aTorque = (pair.A->R / sumRaRb) * rVec.cross(frictionForce);

		// Gravity on a:
		const vector3d gravForceOnA = (G * pair.A->m * pair.B->m / (dist * dist)) * (rVec / dist);

		// Total forces on a:
		totalForce = gravForceOnA + elasticForce + frictionForce + vdwForce;

		// Elastic and Friction b:
		// Flip direction b -> a:
		rVec = -rVec;
		dVel = -dVel;
		elasticForce = -elasticForce;

		const vector3d relativeVelOfB = dVel - dVel.dot(rVec) * (rVec / (dist * dist)) - pair.B->w.cross(pair.B->R / sumRaRb * rVec) - pair.A->w.cross(pair.A->R / sumRaRb * rVec);
		relativeVelMag = relativeVelOfB.norm(); // todo - This should be the same as mag for A. Same speed different direction.
		if (relativeVelMag > 1e-10)
		{
			frictionForce = mu * (elasticForce.norm() + vdwForce.norm()) * (relativeVelOfB / relativeVelMag);
		}
		const vector3d bTorque = (pair.B->R / sumRaRb) * rVec.cross(frictionForce);

		pair.A->aacc += aTorque / pair.A->moi;
		pair.B->aacc += bTorque / pair.B->moi;


		if (writeStep)
		{
			// Calculate potential energy. Important to recognize that the factor of 1/2 is not in front of K because this is for the spring potential in each ball and they are the same potential.
			O.PE += -G * pair.A->m * pair.B->m / dist + 0.5 * k * overlap * overlap;
		}
	}
	else
	{
		// No collision: Include gravity only:
		const vector3d gravForceOnA = (G * pair.A->m * pair.B->m / (dist * dist)) * (rVec / dist);
		totalForce = gravForceOnA;
		if (writeStep)
		{
			O.PE += -G * pair.A->m * pair.B->m / dist;
		}

		// For expanding overlappers:
		//pair.A->vel = { 0,0,0 };
		//pair.B->vel = { 0,0,0 };
	}

	// Newton's equal and opposite forces applied to acceleration of each ball:
	O.acc[A] += totalForce / pair.A->m;
	O.acc[B] -= totalForce / pair.B->m;

	// So last distance can be known for COR:
	O.distances[e] = dist;
}

int main()
{
	int n = 10000; // Number of particles
	int n_pairs = n * (n - 1) / 2;

	std::vector<p> psys(n); // Particle system

	std::vector<p_pair> p_pairs(n_pairs); // All particle pairs

	for (size_t i = 0; i < n_pairs; i++)
	{
		// Pair Combinations [A,B] [B,C] [C,D]... [A,C] [B,D] [C,E]... ...
		int A = i % n;
		int stride = 1 + i / n; // Stride increases by 1 after each full set of pairs
		int B = (A + stride) % n;

		// Create particle* pair
		p_pairs[i] = { &psys[A], &psys[B] };
	}


	std::for_each(std::execution::par_unseq, psys.begin(), psys.end(), update_kinematics);


}