#include "scratchPad.h"

#include <ppl.h>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <unordered_map>
#include <windows.h>



vector3d* g_positions; // one position per particle
vector3d* g_accelerationTotals; // one acceleration total per particle

void calculate(int particleAidx, int particleBidx)
{
    vector3d particleAposition = g_positions[particleAidx];
    vector3d particleBposition = g_positions[particleBidx];
    //...
    vector3d acceleration = 0;// todo, acceleration between particle A and B

    // can't do this, not multi-thread safe
    //g_accelerationTotals[particleAidx] += acceleration;
    //g_accelerationTotals[particleBidx] -= acceleration;

    // add the acceleration of A and B to their totals, this IS multi-thread safe
    InterlockedAdd(&g_accelerationTotals[particleAidx].x, acceleration.x);
    InterlockedAdd(&g_accelerationTotals[particleAidx].y, acceleration.y);
    InterlockedAdd(&g_accelerationTotals[particleAidx].z, acceleration.z);
    InterlockedAdd(&g_accelerationTotals[particleAidx].x, -acceleration.x);
    InterlockedAdd(&g_accelerationTotals[particleAidx].y, -acceleration.y);
    InterlockedAdd(&g_accelerationTotals[particleAidx].z, -acceleration.z);
}


int main()
{
	
}