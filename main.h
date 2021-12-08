//
// Created by Ruurd on 16-Oct-21.
//

#include <curand_kernel.h>

struct Params {
    float burnRate;
    float heightEffectMultiplierUp;
    float heightEffectMultiplierDown;
    float windEffectMultiplier;
    float activityThreshold;
    float spreadSpeed;
    float deathRate;
    float areaEffectMultiplier;
    float fireDeathThreshold;
    float windMatrix[8];
    float cellArea;
};

struct Cell {
    float fireActivity;
    float fuel;
    float height;
    float landCoverSpreadRate;
    // bank conflicts??
};

__global__ void gpuTick(curandState *randState, const Cell *board, Cell *boardCopy, Params params,
                        unsigned int width, unsigned int height);

#ifndef CUDA_SIM_MAIN_H
#define CUDA_SIM_MAIN_H

#endif //CUDA_SIM_MAIN_H
