//
// Created by Ruurd on 16-Oct-21.
//

#include <curand_kernel.h>
#include <array>

const int paramCount = 10;
enum Params {
    burnRate = 0,
    heightEffectMultiplierUp = 1,
    heightEffectMultiplierDown = 2,
    windEffectMultiplier = 3,
    activityThreshold = 4,
    spreadSpeed = 5,
    deathRate = 6,
    areaEffectMultiplier = 7,
    fireDeathThreshold = 8,
    cellArea = 9,
};

struct Cell {
    double fireActivity;
    double fuel;
    short height;
    int landCoverSpreadIndex;
};

__global__ void gpuTick(curandState *randState, const Cell *board, Cell *boardCopy, Params params,
                        unsigned int width, unsigned int height);

#ifndef CUDA_SIM_MAIN_H
#define CUDA_SIM_MAIN_H

#endif //CUDA_SIM_MAIN_H
