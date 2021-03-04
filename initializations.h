////////////////////////////////////
// Initialization (Units are CGS) //
////////////////////////////////////

const double
G = 6.67e-8,   // Gravitational constant
density = 2.7, //2.7, // Typically based on some rock density
mu = 0.9,      // Coeff of friction
cor = 0.3,     // Coeff of restitution
simTimeSeconds = 18000;

// Simulation Structure
const int
skip = 500,     // Steps thrown away before recording a step to the buffer. 500*.04 is every 20 seconds in sim.
properties = 11, // Number of columns in simData file per ball
genBalls = 5120,
attempts = 200; // How many times to try moving every ball touching another in generator.

int
steps = -1;

double
dt = -1,
kin = -1,      // Spring constant
kout = -1,
soc,				// double the radius of the initial system. Any ball outside that isn't considered for dynamic dt calibration.
scaleBalls = 12.5, // base radius of balls
spaceRange = pow(scaleBalls * genBalls, 1. / 3.), // Rough minimum space required
spaceRangeIncrement = scaleBalls * 3,
KEfactor = 0,       // Determines collision velocity based on KE/PE
impactParameter = 0, // Impact angle radians
z0Rot = 0,           // Cluster one z axis rotation
y0Rot = 0,           // Cluster one y axis rotation
z1Rot = 0,           // Cluster two z axis rotation
y1Rot = 0;           // Cluster two y axis rotation

const bool
springTest = false; // If true, spring compression is checked against ball radius. If compression > .1R, send warning to console. 

// File from which to proceed with further simulations
std::string
path = "C:/Users/milin/Desktop/GoogleDrive/GradResearch/Development/SpaceLab/x64/Release/",
projectileName = "5120_",
targetName = "5730_",
outputPrefix = "Unnamed";