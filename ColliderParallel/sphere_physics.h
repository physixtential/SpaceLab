#pragma once
#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <algorithm>
#include "../vector3d.hpp"
#include "constants.hpp"

/// @brief Contains everything needed to describe the physical state of a sphere.
/// Radius, mass, and moment of inertia must be defined upon creation.
struct Sphere
{
	vector3d
		pos,
		vel,
		velh,
		w,
		wh,
		force,
		torque;

	const double
		R,
		m,
		moi;

	Sphere(const double R, const double density)
		:
		R(R),
		m(density * 4. / 3. * M_PI * R * R * R),
		moi(.4 * m * R * R)
	{}
};

struct Sphere_pair
{
private:
	Sphere& a_;
	Sphere& b_;
	double overlap_; // Positive in contact. Negative otherwise. Negative indicates distance between nearest surfaces.

public:
	// Getters
	Sphere& a() const { return a_; }
	Sphere& b() const { return b_; }
	double get_overlap() const { return overlap_; }

	// Setters
	void a_add_force(const vector3d& force) { a_.force += force; }
	void b_add_force(const vector3d& force) { b_.force += force; }
	void set_overlap() { overlap_ = (a_.R + b_.R) - distance(Sphere_pair(a_, b_)); }

	// Constructors
	Sphere_pair() = default;
	Sphere_pair(Sphere& a, Sphere& b, double overlap) : a_(a), b_(b), overlap_(overlap) {} // Init all
	Sphere_pair(Sphere& a, Sphere& b) : a_(a), b_(b) {}

};

class Cosmos
{
public:
	int n; // Particle count.
	std::vector<Sphere> spheres;

	// Useful values:
	double r_min = -1; // Smallest particle.
	double r_max = -1; // Largest particle.
	double m_total = -1; // Total mass of the cosmos.
	double initial_radius = -1; // Determined by the two most distance particles from each other.
	double v_collapse = 0; // Fastest particle at the moment the most distant matter collapses onto the rest.
	double v_max = -1; // Fastest particle.
	double v_max_prev = HUGE_VAL;

	vector3d mom = { 0, 0, 0 }; // Momentum of the cosmos.
	vector3d ang_mom = { 0, 0, 0 }; // Angular momentum of the cosmos.

	double U = 0; // Potential energy.
	double T = 0; // Kinetic energy.

	Cosmos() = default;
	Cosmos(std::vector<Sphere> spheres) : spheres(spheres) {}

	void operator()(int step)
	{
		// perform one timestep.

		// then in main all I do is for loop O(i)
	}
};

vector3d vec_atob(const Sphere_pair& pair)
{
	return pair.a().pos - pair.b().pos;
}

double distance(const Sphere_pair& pair)
{
	vector3d vec = pair.b().pos - pair.a().pos;
	return sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

void apply_grav_force(Sphere_pair& pair, double& dist)
{
	vector3d force = (G * pair.a().m * pair.b().m / (dist * dist)) * (vec_atob(pair) / dist);
	pair.a_add_force(force);
	pair.b_add_force(-force);
}

void apply_elastic_force(Sphere_pair& pair, double& k)
{
	a.force -= k * pair.get_overlap() * .5 * (vec_atob(pair) / dist);
}

void apply_friction_sliding(Sphere& a, Sphere& b)
{

}

void apply_friction_rolling(Sphere& a, Sphere& b)
{

}

void apply_cohesion_force(Sphere& a, Sphere& b)
{

}

void update_sphere(Sphere& sphere)
{
	sphere.velh = sphere.vel + .5 * sphere.force / sphere.m * dt;
	sphere.wh = sphere.w + .5 * sphere.torque / sphere.moi * dt;
	sphere.pos += sphere.velh * dt;
	sphere.force = { 0, 0, 0 };
	sphere.torque = { 0, 0, 0 };
}

std::vector<Sphere_pair> return_pairs(std::vector<Sphere> spheres)
{
	int n = spheres.size();
	int combinations = n * (n - 1) / 2;
	std::vector<Sphere_pair> pairs;
	for (size_t i = 0; i < combinations; i++)
	{
		// Pair Combinations [A,B] [B,C] [C,D]... [A,C] [B,D] [C,E]... ...
		int A = i % n;
		int stride = 1 + i / n; // Stride increases by 1 after each full set of pairs
		int B = (A + stride) % n;

		// Create particle* pair
		pairs[i] = { &spheres[A], &spheres[B] };
	}
	return pairs;
}

