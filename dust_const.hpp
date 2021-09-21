#pragma once
#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
////////////////////////////////////
// Initialization (Units are CGS) //
////////////////////////////////////

constexpr bool dynamicTime = false;

constexpr double
G = 6.67e-8, // Gravitational constant
density = 2.7,
u_s = 0.9, // Coeff of sliding friction
u_r = 0.01, // Coeff of rolling friction
sigma = .29, // Poisson ratio for rolling friction.
Y = 2.0e12, // Young's modulus in erg/cm3
cor = 0.4, // Coeff of restitution
simTimeSeconds = 1e-2, // Seconds
timeResolution = 1e-5, // Seconds - This is duration between exported steps. Must be greater than dt
fourThirdsPiRho = 4. / 3. * M_PI * density, // for fraction of smallest sphere radius.
scaleBalls = 1e-4, // base radius of ball.
maxOverlap = .1, // of scaleBalls
KEfactor = 0, // Determines collision velocity based on KE/PE
vCustom = 1, // Velocity cm/s
kConsts = fourThirdsPiRho / (maxOverlap * maxOverlap),
impactParameter = 0, // Impact angle radians
Ha = 21.1e-13, // Hamaker constant for vdw force
h_min = scaleBalls * .1,//1e8 * std::numeric_limits<double>::epsilon(), // 2.22045e-10 (epsilon is 2.22045e-16)
cone = M_PI_2; // Cone of particles ignored moving away from center of mass. Larger angle ignores more.

// Simulation Structure
constexpr int
properties = 11, // Number of columns in simData file per ball
genBalls = 50,
attempts = 200; // How many times to try moving every ball touching another in generator.

size_t
skip = 0,     // Steps thrown away before recording a step to the buffer. 500*.04 is every 20 seconds in sim.
steps = 0;

double
dt = -1,
kin = -1,      // Spring constant
kout = -1,
spaceRange = std::pow((1. / .74 * scaleBalls * scaleBalls * scaleBalls * genBalls), 1. / 3.), // Rough minimum space required
spaceRangeIncrement = scaleBalls * 3,
z0Rot = 0,           // Cluster one z axis rotation
y0Rot = 0,           // Cluster one y axis rotation
z1Rot = 0,           // Cluster two z axis rotation
y1Rot = 0,           // Cluster two y axis rotation
simTimeElapsed = 0;

// File from which to proceed with further simulations
inline std::string
path = "C:/Users/milin/Desktop/GoogleDrive/GradResearch/Development/SpaceLab/ColliderSingleCore/",
projectileName = "cohTest_",
targetName = "cohTest_",
outputPrefix = "Unnamed_";
