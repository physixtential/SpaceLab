#pragma once
#include <cmath>
#include <string>
#include <numbers>

////////////////////////////////////
// Initialization (Units are CGS) //
////////////////////////////////////

constexpr bool dynamicTime = false;

// T(K)	v(cm / s)
// 3    0.36301555459799423
// 10	0.6627726932618989
// 30	1.1479559785988718
// 100	2.0958712816717324
// 300	3.6301555459799424
// 1000	6.627726932618988

using std::numbers::pi;
constexpr double G = 6.67e-8;  // Gravitational constant
constexpr double density = 2.25;
constexpr double u_s = 0.1;                // Coeff of sliding friction
constexpr double u_r = 1e-5;               // Coeff of rolling friction
constexpr double sigma = .29;              // Poisson ratio for rolling friction.
constexpr double Y = 2.0e12;               // Young's modulus in erg/cm3
constexpr double cor = 0.4;                // Coeff of restitution
constexpr double simTimeSeconds = 0.5e-3;  // Seconds
constexpr double timeResolution = 1e-5;    // Seconds - This is duration between exported steps.
constexpr double fourThirdsPiRho = 4. / 3. * pi * density;  // for fraction of smallest sphere radius.
constexpr double scaleBalls = 1e-5;                         // base radius of ball.
constexpr double maxOverlap = .1;                           // of scaleBalls
constexpr double KEfactor = 0;                              // Determines collision velocity based on KE/PE
constexpr double v_custom = 0.36301555459799423;            // Velocity cm/s
constexpr double kConsts = 3e3 * fourThirdsPiRho / (maxOverlap * maxOverlap);
constexpr double impactParameter = 0;  // Impact angle radians
constexpr double Ha = 4.7e-12;         // Hamaker constant for vdw force
constexpr double h_min =
    scaleBalls * .1;  // 1e8 * std::numeric_limits<double>::epsilon(), // 2.22045e-10 (epsilon is 2.22045e-16)
constexpr double cone =
    pi / 2;  // Cone of particles ignored moving away from center of mass. Larger angle ignores more.

// Simulation Structure
constexpr int properties = 11;  // Number of columns in simData file per ball
constexpr int genBalls = 2;
constexpr int attempts = 200;  // How many times to try moving every ball touching another in generator.

int skip = 0;  // Steps thrown away before recording a step to the buffer. 500*.04 is every 20 seconds in sim.
int steps = 0;

double dt = -1;
double kin = -1;  // Spring constant
double kout = -1;
double spaceRange = 4 * std::pow(
                            (1. / .74 * scaleBalls * scaleBalls * scaleBalls * genBalls),
                            1. / 3.);  // Rough minimum space required
double spaceRangeIncrement = scaleBalls * 3;
double z0Rot = 0;  // Cluster one z axis rotation
double y0Rot = 0;  // Cluster one y axis rotation
double z1Rot = 0;  // Cluster two z axis rotation
double y1Rot = 0;  // Cluster two y axis rotation
double simTimeElapsed = 0;

// File from which to proceed with further simulations
std::string path = "/home/lucas/Desktop/Research/SpaceLabTesting/SpaceLab/";
std::string output_folder = path + "output/";
// std::string projectileName = "1_10_R1e-04_v4e-01_cor0.63_mu0.1_rho2.25_k7e+00_Ha5e-12_dt4e-10_";
// std::string targetName = "10_R1e-04_v4e-01_cor0.63_mu0.1_rho2.25_k7e+00_Ha5e-12_dt4e-10_";
std::string output_prefix = "";
