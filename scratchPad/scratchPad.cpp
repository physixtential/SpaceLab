#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <sstream>
#include <vector>
#include <algorithm>
#include <execution>

#include "scratchPad.h"


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

void timestep(p_pair& pair)
{
	pair.A->pos += {.1, .2, .3};
	pair.B->pos += {1, 2, 3};
	// todo 
	pair.dist = (pair.A->pos - pair.B->pos).norm();
}

int main()
{
	int n = 4000; // Number of particles
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

	std::for_each(std::execution::par, p_pairs.begin(), p_pairs.end(), timestep);

}