#include "../ball_group.hpp"

#include <cmath>
#include <iostream>
#include <fstream>
#include <ctime>
#include <sstream>
#include <iomanip>


// String buffers to hold data in memory until worth writing to file:
std::stringstream ballBuffer;
std::stringstream energyBuffer;

// These are used within simOneStep to keep track of time.
// They need to survive outside its scope, and I don't want to have to pass them all.
const time_t start = time(nullptr);  // For end of program analysis
time_t startProgress;                // For progress reporting (gets reset)
time_t lastWrite;                    // For write control (gets reset)
bool writeStep;                      // This prevents writing to file every step (which is slow).


// Prototypes
void
sim_one_step(const bool write_step);
void
sim_looper();
void
safetyChecks();

/// @brief The ballGroup run by the main sim looper.
Ball_group O(output_folder, projectileName, targetName, v_custom); // Collision
// Ball_group O(path, targetName, 0);  // Continue
// std::cout<<"Start Main"<<std::endl;
// std::cerr<<"genBalls: "<<genBalls<<std::endl;
// Ball_group O(20, true, v_custom); // Generate
// Ball_group O(genBalls, true, v_custom); // Generate

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
int
main(const int argc, char const* argv[])
{
    energyBuffer.precision(12);  // Need more precision on momentum.

    // Runtime arguments:
    if (argc > 1) {
        // numThreads = atoi(argv[1]);
        // fprintf(stderr,"\nThread count set to %i.\n", numThreads);
        // projectileName = argv[2];
        // targetName = argv[3];
        // KEfactor = atof(argv[4]);
    }

    // O.zeroAngVel();
    // O.pushApart();
    safetyChecks();

    // Normal sim:
    O.sim_init_write(output_prefix);
    sim_looper();


    // Add projectile: For dust formation BPCA
    // std::string ori_output_prefix = output_prefix;
    // for (int i = 0; i < 20; i++) {
    // // for (int i = 0; i < 250; i++) {
    //     O.zeroAngVel();
    //     O.zeroVel();
    //     O = O.add_projectile();
    //     O.sim_init_write(ori_output_prefix);
    //     sim_looper();
    //     simTimeElapsed = 0;
    // }
}  // end main
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////


void
sim_one_step(const bool write_step)
{
    /// FIRST PASS - Update Kinematic Parameters:
    for (int Ball = 0; Ball < O.num_particles; Ball++) {
        // Update velocity half step:
        O.velh[Ball] = O.vel[Ball] + .5 * O.acc[Ball] * dt;

        // Update angular velocity half step:
        O.wh[Ball] = O.w[Ball] + .5 * O.aacc[Ball] * dt;

        // Update position:
        O.pos[Ball] += O.velh[Ball] * dt;

        // Reinitialize acceleration to be recalculated:
        O.acc[Ball] = {0, 0, 0};

        // Reinitialize angular acceleration to be recalculated:
        O.aacc[Ball] = {0, 0, 0};
    }

    /// SECOND PASS - Check for collisions, apply forces and torques:
    for (int A = 1; A < O.num_particles; A++)  // cuda
    {
        /// DONT DO ANYTHING HERE. A STARTS AT 1.
        for (int B = 0; B < A; B++) {
            const double sumRaRb = O.R[A] + O.R[B];
            const vec3 rVecab = O.pos[B] - O.pos[A];  // Vector from a to b.
            const vec3 rVecba = -rVecab;
            const double dist = (rVecab).norm();

            // Check for collision between Ball and otherBall:
            double overlap = sumRaRb - dist;

            vec3 totalForceOnA{0, 0, 0};

            // Distance array element: 1,0    2,0    2,1    3,0    3,1    3,2 ...
            int e = static_cast<unsigned>(A * (A - 1) * .5) + B;  // a^2-a is always even, so this works.
            double oldDist = O.distances[e];

            // Check for collision between Ball and otherBall.
            if (overlap > 0) {
                double k;
                // Apply coefficient of restitution to balls leaving collision.
                if (dist >= oldDist) {
                    k = kout;
                } else {
                    k = kin;
                }

                // Cohesion (in contact) h must always be h_min:
                constexpr double h = h_min;
                const double Ra = O.R[A];
                const double Rb = O.R[B];
                constexpr double h2 = h * h;
                const double twoRah = 2 * Ra * h;
                const double twoRbh = 2 * Rb * h;
                const vec3 vdwForceOnA = Ha / 6 * 64 * Ra * Ra * Ra * Rb * Rb * Rb *
                                         ((h + Ra + Rb) / ((h2 + twoRah + twoRbh) * (h2 + twoRah + twoRbh) *
                                                           (h2 + twoRah + twoRbh + 4 * Ra * Rb) *
                                                           (h2 + twoRah + twoRbh + 4 * Ra * Rb))) *
                                         rVecab.normalized();

                // Elastic force:
                const vec3 elasticForceOnA = -k * overlap * .5 * (rVecab / dist);

                // Gravity force:
                const vec3 gravForceOnA = (G * O.m[A] * O.m[B] / (dist * dist)) * (rVecab / dist);

                // Sliding and Rolling Friction:
                vec3 slideForceOnA{0, 0, 0};
                vec3 rollForceA{0, 0, 0};
                vec3 torqueA{0, 0, 0};
                vec3 torqueB{0, 0, 0};

                // Shared terms:
                const double elastic_force_A_mag = elasticForceOnA.norm();
                const vec3 r_a = rVecab * O.R[A] / sumRaRb;  // Center to contact point
                const vec3 r_b = rVecba * O.R[B] / sumRaRb;
                const vec3 w_diff = O.w[A] - O.w[B];

                // Sliding friction terms:
                const vec3 d_vel = O.vel[B] - O.vel[A];
                const vec3 frame_A_vel_B = d_vel - d_vel.dot(rVecab) * (rVecab / (dist * dist)) -
                                           O.w[A].cross(r_a) - O.w[B].cross(r_a);

                // Compute sliding friction force:
                const double rel_vel_mag = frame_A_vel_B.norm();
                if (rel_vel_mag > 1e-13)  // Divide by zero protection.
                {
                    // In the frame of A, B applies force in the direction of B's velocity.
                    slideForceOnA = u_s * elastic_force_A_mag * (frame_A_vel_B / rel_vel_mag);
                }

                // Compute rolling friction force:
                const double w_diff_mag = w_diff.norm();
                if (w_diff_mag > 1e-13)  // Divide by zero protection.
                {
                    rollForceA =
                        -u_r * elastic_force_A_mag * (w_diff).cross(r_a) / (w_diff).cross(r_a).norm();
                }

                // Total forces on a:
                totalForceOnA = gravForceOnA + elasticForceOnA + slideForceOnA + vdwForceOnA;

                // Total torque a and b:
                torqueA = r_a.cross(slideForceOnA + rollForceA);
                torqueB = r_b.cross(-slideForceOnA + rollForceA);

                O.aacc[A] += torqueA / O.moi[A];
                O.aacc[B] += torqueB / O.moi[B];

                if (write_step) {
                    // No factor of 1/2. Includes both spheres:
                    // O.PE += -G * O.m[A] * O.m[B] / dist + 0.5 * k * overlap * overlap;

                    // Van Der Waals + elastic:
                    const double diffRaRb = O.R[A] - O.R[B];
                    const double z = sumRaRb + h;
                    const double two_RaRb = 2 * O.R[A] * O.R[B];
                    const double denom_sum = z * z - (sumRaRb * sumRaRb);
                    const double denom_diff = z * z - (diffRaRb * diffRaRb);
                    const double U_vdw =
                        -Ha / 6 *
                        (two_RaRb / denom_sum + two_RaRb / denom_diff + log(denom_sum / denom_diff));
                    O.PE += U_vdw + 0.5 * k * overlap * overlap;
                }
            } else  // Non-contact forces:
            {
                // No collision: Include gravity and vdw:
                // const vec3 gravForceOnA = (G * O.m[A] * O.m[B] / (dist * dist)) * (rVecab / dist);

                // Cohesion (non-contact) h must be positive or h + Ra + Rb becomes catastrophic cancellation:
                double h = std::fabs(overlap);
                if (h < h_min)  // If h is closer to 0 (almost touching), use hmin.
                {
                    h = h_min;
                }
                const double Ra = O.R[A];
                const double Rb = O.R[B];
                const double h2 = h * h;
                const double twoRah = 2 * Ra * h;
                const double twoRbh = 2 * Rb * h;
                const vec3 vdwForceOnA = Ha / 6 * 64 * Ra * Ra * Ra * Rb * Rb * Rb *
                                         ((h + Ra + Rb) / ((h2 + twoRah + twoRbh) * (h2 + twoRah + twoRbh) *
                                                           (h2 + twoRah + twoRbh + 4 * Ra * Rb) *
                                                           (h2 + twoRah + twoRbh + 4 * Ra * Rb))) *
                                         rVecab.normalized();

                totalForceOnA = vdwForceOnA;  // +gravForceOnA;
                if (write_step) {
                    // O.PE += -G * O.m[A] * O.m[B] / dist; // Gravitational

                    const double diffRaRb = O.R[A] - O.R[B];
                    const double z = sumRaRb + h;
                    const double two_RaRb = 2 * O.R[A] * O.R[B];
                    const double denom_sum = z * z - (sumRaRb * sumRaRb);
                    const double denom_diff = z * z - (diffRaRb * diffRaRb);
                    const double U_vdw =
                        -Ha / 6 *
                        (two_RaRb / denom_sum + two_RaRb / denom_diff + log(denom_sum / denom_diff));
                    O.PE += U_vdw;  // Van Der Waals
                }

                // todo this is part of push_apart. Not great like this.
                // For pushing apart overlappers:
                // O.vel[A] = { 0,0,0 };
                // O.vel[B] = { 0,0,0 };
            }

            // Newton's equal and opposite forces applied to acceleration of each ball:
            O.acc[A] += totalForceOnA / O.m[A];
            O.acc[B] -= totalForceOnA / O.m[B];

            // So last distance can be known for COR:
            O.distances[e] = dist;
        }
        // DONT DO ANYTHING HERE. A STARTS AT 1.
    }

    if (write_step) {
        ballBuffer << '\n';  // Prepares a new line for incoming data.
    }

    // THIRD PASS - Calculate velocity for next step:
    for (int Ball = 0; Ball < O.num_particles; Ball++) {
        // Velocity for next step:
        O.vel[Ball] = O.velh[Ball] + .5 * O.acc[Ball] * dt;
        O.w[Ball] = O.wh[Ball] + .5 * O.aacc[Ball] * dt;

        if (write_step) {
            // Send positions and rotations to buffer:
            if (Ball == 0) {
                ballBuffer << O.pos[Ball][0] << ',' << O.pos[Ball][1] << ',' << O.pos[Ball][2] << ','
                           << O.w[Ball][0] << ',' << O.w[Ball][1] << ',' << O.w[Ball][2] << ','
                           << O.w[Ball].norm() << ',' << O.vel[Ball].x << ',' << O.vel[Ball].y << ','
                           << O.vel[Ball].z << ',' << 0;
            } else {
                ballBuffer << ',' << O.pos[Ball][0] << ',' << O.pos[Ball][1] << ',' << O.pos[Ball][2] << ','
                           << O.w[Ball][0] << ',' << O.w[Ball][1] << ',' << O.w[Ball][2] << ','
                           << O.w[Ball].norm() << ',' << O.vel[Ball].x << ',' << O.vel[Ball].y << ','
                           << O.vel[Ball].z << ',' << 0;
            }

            O.KE += .5 * O.m[Ball] * O.vel[Ball].normsquared() +
                    .5 * O.moi[Ball] * O.w[Ball].normsquared();  // Now includes rotational kinetic energy.
            O.mom += O.m[Ball] * O.vel[Ball];
            O.ang_mom += O.m[Ball] * O.pos[Ball].cross(O.vel[Ball]) + O.moi[Ball] * O.w[Ball];
        }
    }  // THIRD PASS END
}  // one Step end


void
sim_looper()
{
    std::cerr << "Beginning simulation...\n";

    startProgress = time(nullptr);

    for (int Step = 1; Step < steps; Step++)  // Steps start at 1 because the 0 step is initial conditions.
    {
        // Check if this is a write step:
        if (Step % skip == 0) {
            writeStep = true;

            simTimeElapsed += dt * skip;

            // Progress reporting:
            float eta = ((time(nullptr) - startProgress) / static_cast<float>(skip) *
                         static_cast<float>(steps - Step)) /
                        3600.f;  // Hours.
            float real = (time(nullptr) - start) / 3600.f;
            float simmed = static_cast<float>(simTimeElapsed / 3600.f);
            float progress = (static_cast<float>(Step) / static_cast<float>(steps) * 100.f);
            fprintf(
                stderr,
                "%u\t%2.0f%%\tETA: %5.2lf\tReal: %5.2f\tSim: %5.2f hrs\tR/S: %5.2f\n",
                Step,
                progress,
                eta,
                real,
                simmed,
                real / simmed);
            // fprintf(stdout, "%u\t%2.0f%%\tETA: %5.2lf\tReal: %5.2f\tSim: %5.2f hrs\tR/S: %5.2f\n", Step,
            // progress, eta, real, simmed, real / simmed);
            fflush(stdout);
            startProgress = time(nullptr);
        } else {
            writeStep = false;
        }

        // Physics integration step:
        sim_one_step(writeStep);

        if (writeStep) {
            // Write energy to stream:
            energyBuffer << '\n'
                         << simTimeElapsed << ',' << O.PE << ',' << O.KE << ',' << O.PE + O.KE << ','
                         << O.mom.norm() << ','
                         << O.ang_mom.norm();  // the two zeros are bound and unbound mass

            // Reinitialize energies for next step:
            O.KE = 0;
            O.PE = 0;
            O.mom = {0, 0, 0};
            O.ang_mom = {0, 0, 0};
            // unboundMass = 0;
            // boundMass = massTotal;

            // Data Export. Exports every 10 writeSteps (10 new lines of data) and also if the last write was
            // a long time ago.
            if (time(nullptr) - lastWrite > 1800 || Step / skip % 10 == 0) {
                // Report vMax:
                std::cerr << "vMax = " << O.getVelMax() << " Steps recorded: " << Step / skip << '\n';
                std::cerr << "Data Write\n\n";

                // Write simData to file and clear buffer.
                std::ofstream ballWrite;
                ballWrite.open(output_folder + output_prefix + "simData.csv", std::ofstream::app);
                ballWrite << ballBuffer.rdbuf();  // Barf buffer to file.
                ballBuffer.str("");               // Empty the stream for next filling.
                ballWrite.close();

                // Write Energy data to file and clear buffer.
                std::ofstream energyWrite;
                energyWrite.open(output_folder + output_prefix + "energy.csv", std::ofstream::app);
                energyWrite << energyBuffer.rdbuf();
                energyBuffer.str("");  // Empty the stream for next filling.
                energyWrite.close();

                lastWrite = time(nullptr);
            }  // Data export end


            if (dynamicTime) { O.calibrate_dt(Step, false); }
        }  // writestep end
    }

    const time_t end = time(nullptr);

    std::cerr << "Simulation complete!\n"
              << O.num_particles << " Particles and " << steps << " Steps.\n"
              << "Simulated time: " << steps * dt << " seconds\n"
              << "Computation time: " << end - start << " seconds\n";
    std::cerr << "\n===============================================================\n";
}  // end simLooper


void
safetyChecks()
{
    titleBar("SAFETY CHECKS");

    if (O.soc <= 0) {
        fprintf(stderr, "\nSOC NOT SET\n");
        exit(EXIT_FAILURE);
    }

    if (O.v_collapse <= 0) {
        fprintf(stderr, "\nvCollapse NOT SET\n");
        exit(EXIT_FAILURE);
    }

    if (skip == 0) {
        fprintf(stderr, "\nSKIP NOT SET\n");
        exit(EXIT_FAILURE);
    }

    if (kin < 0) {
        fprintf(stderr, "\nSPRING CONSTANT NOT SET\n");
        exit(EXIT_FAILURE);
    }

    if (dt <= 0) {
        fprintf(stderr, "\nDT NOT SET\n");
        exit(EXIT_FAILURE);
    }

    if (steps == 0) {
        fprintf(stderr, "\nSTEPS NOT SET\n");
        exit(EXIT_FAILURE);
    }

    if (O.initial_radius <= 0) {
        fprintf(stderr, "\nCluster initialRadius not set\n");
        exit(EXIT_FAILURE);
    }

    for (int Ball = 0; Ball < O.num_particles; Ball++) {
        if (O.pos[Ball].norm() < vec3(1e-10, 1e-10, 1e-10).norm()) {
            fprintf(stderr, "\nA ball position is [0,0,0]. Possibly didn't initialize balls properly.\n");
            exit(EXIT_FAILURE);
        }

        if (O.R[Ball] <= 0) {
            fprintf(stderr, "\nA balls radius <= 0.\n");
            exit(EXIT_FAILURE);
        }

        if (O.m[Ball] <= 0) {
            fprintf(stderr, "\nA balls mass <= 0.\n");
            exit(EXIT_FAILURE);
        }
    }
    titleBar("SAFETY PASSED");
}


// void setGuidDT(const double& vel)
//{
//	// Guidos k and dt:
//	dt = .01 * O.getRmin() / fabs(vel);
//}
//
// void setGuidK(const double& vel)
//{
//	kin = O.getMassMax() * vel * vel / (.1 * O.R[0] * .1 * O.R[0]);
//	kout = cor * kin;
//}
