#pragma once
#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
////////////////////////////////////
// Initialization (Units are CGS) //
////////////////////////////////////

constexpr bool dynamicTime = true;

constexpr double
G = 6.67e-8,   // Gravitational constant
density = 2.7, //2.7, // Typically based on some rock density
mu = 0.9,      // Coeff of friction
cor = 0.8,     // Coeff of restitution
simTimeSeconds = 18000., // Seconds
timeResolution = 20., // Seconds - This is duration between exported steps
maxOverlap = .1,
fourThirdsPiRho = 4. / 3. * M_PI * density, // for fraction of smallest sphere radius.
scaleBalls = 100, // base radius of balls
KEfactor = 0,       // Determines collision velocity based on KE/PE
impactParameter = 0, // Impact angle radians
Ha = 21.1e-20; // Hamaker constant for vdw force

// Simulation Structure
inline const unsigned int
properties = 11, // Number of columns in simData file per ball
genBalls = 100,
attempts = 200; // How many times to try moving every ball touching another in generator.

inline size_t
skip = 0,     // Steps thrown away before recording a step to the buffer. 500*.04 is every 20 seconds in sim.
steps = 0;

inline double
dt = -1,
kin = -1,      // Spring constant
kout = -1,
vCustom = 50,
spaceRange = std::pow((1. / .74 * scaleBalls * scaleBalls * scaleBalls * genBalls), 1. / 3.), // Rough minimum space required
spaceRangeIncrement = scaleBalls * 3,
z0Rot = 0,           // Cluster one z axis rotation
y0Rot = 0,           // Cluster one y axis rotation
z1Rot = 0,           // Cluster two z axis rotation
y1Rot = 0,           // Cluster two y axis rotation
simTimeElapsed = 0;

// File from which to proceed with further simulations
inline std::string
path = "C:/Users/milin/Desktop/GoogleDrive/GradResearch/Development/SpaceLab/ColliderSingleCore",
projectileName = "100-R847.93-v50-cor0.64-mu0.9-rho2.70_k8.4823e+08_dt0.0011547_",
targetName = "100-R847.93-v50-cor0.64-mu0.9-rho2.70_k8.4823e+08_dt0.0011547_",
outputPrefix = "Unnamed_";
