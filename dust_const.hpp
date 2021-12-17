#pragma once
#include <cmath>
#include <string>
#include <numbers>

////////////////////////////////////
// Initialization (Units are CGS) //
////////////////////////////////////

constexpr bool dynamicTime = false;

//T(K)	v(cm / s)
//3		0.36301555459799423
//10	0.6627726932618989
//30	1.1479559785988718
//100	2.0958712816717324
//300	3.6301555459799424
//1000	6.627726932618988

using std::numbers::pi;
constexpr double
G = 6.67e-8, // Gravitational constant
density = 2.25,
u_s = 0.1, // Coeff of sliding friction
u_r = 1.0e-3, // Coeff of rolling friction
sigma = .29, // Poisson ratio for rolling friction.
Y = 2.0e12, // Young's modulus in erg/cm3 
cor = 0.4, // Coeff of restitution
simTimeSeconds = 0.5e-3, // Seconds
timeResolution = 1e-5, // Seconds - This is duration between exported steps.
fourThirdsPiRho = 4. / 3. * pi * density, // for fraction of smallest sphere radius.
scaleBalls = 1e-5, // base radius of ball.
maxOverlap = .1, // of scaleBalls
KEfactor = 0, // Determines collision velocity based on KE/PE
v_custom = 6.627726932618988, // Velocity cm/s
kConsts = 3e3 * fourThirdsPiRho / (maxOverlap * maxOverlap),
impactParameter = 0, // Impact angle radians
Ha = 4.7e-12, // Hamaker constant for vdw force
h_min = scaleBalls * .1,//1e8 * std::numeric_limits<double>::epsilon(), // 2.22045e-10 (epsilon is 2.22045e-16)
cone = pi / 2; // Cone of particles ignored moving away from center of mass. Larger angle ignores more.

// Simulation Structure
constexpr int
properties = 11, // Number of columns in simData file per ball
genBalls = 2,
attempts = 200; // How many times to try moving every ball touching another in generator.

int
skip = 0,     // Steps thrown away before recording a step to the buffer. 500*.04 is every 20 seconds in sim.
steps = 0;

double
dt = -1,
kin = -1,      // Spring constant
kout = -1,
spaceRange = 4 * std::pow((1. / .74 * scaleBalls * scaleBalls * scaleBalls * genBalls), 1. / 3.), // Rough minimum space required
spaceRangeIncrement = scaleBalls * 3,
z0Rot = 0,           // Cluster one z axis rotation
y0Rot = 0,           // Cluster one y axis rotation
z1Rot = 0,           // Cluster two z axis rotation
y1Rot = 0,           // Cluster two y axis rotation
simTimeElapsed = 0;

// File from which to proceed with further simulations
inline std::string
path = "C:/Users/milin/Desktop/VSOUT/",
projectileName = "",
targetName = "2_R2e-05_v1e+00_cor0.16_mu0.1_rho2.25_k3e+01_Ha5e-12_dt2e-10_",
output_prefix = "Unnamed_";
