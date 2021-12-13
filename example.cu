#include <iostream>
#include <chrono>
#include "main.h"
#include <curand.h>
#include <curand_kernel.h>
#include <string>

__global__ void setup_kernel(curandState *state) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    curand_init(1234, idx, 0, &state[idx]);
}

__global__ void gpuTick(curandState *randState, const Cell *board, Cell *boardCopy, const Params params, const unsigned int width,
                        const unsigned int height) {
    const unsigned int size = width * height;
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= size) return;

    unsigned int x = id % width;
    unsigned int y = id / width;

    Cell cell = board[id];

    boardCopy[id] = {
            curand_uniform(randState + id),
            curand_uniform(randState + id) * x,
            cell.height,
            cell.landCoverSpreadRate
    };
}

bool handleError(const std::string &reason) {
    printf("Cuda error! %s\n", reason.c_str());
    return false;
}

int main() {
    int nThreads = 2000;
    int ticks = 2000;
    int repeats = 100;
    int w = 100;
    int h = 100;
    int size = w * h;

    for (int r = 0; r < repeats; r++) {
        Cell *d_board = nullptr;
        Cell *d_boardCopy = nullptr;
        curandState *d_randState = nullptr;
        auto board = new Cell[size];
        cudaError_t cudaStatus;
        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess)
            return handleError("set device");

        size_t free, total;
        cudaStatus = cudaMemGetInfo(&free, &total);
        if (cudaStatus != cudaSuccess)
            return handleError("get mem info");

        // Init random generator on GPU
        cudaStatus = cudaMalloc(&d_randState, sizeof(curandState));
        if (cudaStatus != cudaSuccess)
            return handleError("malloc random");
        setup_kernel<<<size / nThreads + 1, nThreads>>>(d_randState);

        printf("Checking GPU MemInfo: free: %zu, total: %zu\n", free, total);

        // allocate gpu buffers for board and copy
        cudaStatus = cudaMalloc((void **) &d_board, size * sizeof(Cell));
        if (cudaStatus != cudaSuccess)
            return handleError("malloc d_board");

        // allocate gpu buffers for board and copy
        cudaStatus = cudaMalloc((void **) &d_boardCopy, size * sizeof(Cell));
        if (cudaStatus != cudaSuccess)
            return handleError("malloc d_board");

        cudaStatus = cudaMemcpy(d_board, board, size * sizeof(Cell), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
            return handleError("memCpy to GPU");

        for (int i = 0; i < ticks; i++) {
            // Execute on GPU
            gpuTick<<<size / nThreads + 1, nThreads>>>(
                    d_randState, d_board, d_boardCopy,
                    w, h
            );

            std::swap(d_board, d_boardCopy);
        }

        cudaFree(d_board);
        cudaFree(d_boardCopy);
        cudaFree(d_randState);
    }


    printf("SUCCESS");
}