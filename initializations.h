#pragma once
// Made change in master after branching
////////////////////////////////////
// Initialization (Units are CGS) //
////////////////////////////////////
double
dt = 4e-2;

const double
G = 6.67e-8,   // Gravitational constant
density = 2.7, // Typically based on some rock density
mu = 0.3,      // Coeff of friction
cor = 0.8,     // Coeff of restitution
kin = 1e18,      // Spring constant
kout = cor * kin;
//spin = 0.0006;

const int
steps = (int)(12000. / dt),
skip = 500,     // Steps thrown away before recording a step to the buffer. 500*.04 is every 20 seconds in sim.
properties = 11, // Number of columns in simData file per ball
genBalls = 1250,
attempts = 200; // How many times to try moving every ball touching another in generator.

int numThreads = 1; // omp parallel thread count.

double
scaleBalls = 750000, // scales ball radius
spaceRange = pow(scaleBalls * genBalls, 1. / 3.), // Rough minimum space required
spaceRangeIncrement = scaleBalls * 3,
KEfactor = 3.,      // Determines collision velocity based on KE/PE
impactParameter = 0, // Impact angle
z0Rot = 0,           // Cluster one z axis rotation
y0Rot = 0,           // Cluster one y axis rotation
z1Rot = 0,           // Cluster two z axis rotation
y1Rot = 0;           // Cluster two y axis rotation

const bool
springTest = false; // If true, spring compression is checked against ball radius. If compression > .1R, send warning to console. 

// File from which to proceed with further simulations
std::string
path = "C:/Users/milin/Desktop/GoogleDrive/GradResearch/Development/spins/clusters/",
clusterAName = "",
clusterBName = "",
outputPrefix = "Unnamed";
