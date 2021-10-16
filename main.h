//
// Created by Ruurd on 16-Oct-21.
//

struct Cell {
    float fireActivity;
    float fuel;
    float height;
    float landCoverSpreadRate;
    // bank conflicts??
};

__global__ void gpuTick(const Cell *board, Cell *boardCopy, unsigned int width, unsigned int height);

#ifndef CUDA_SIM_MAIN_H
#define CUDA_SIM_MAIN_H

#endif //CUDA_SIM_MAIN_H
