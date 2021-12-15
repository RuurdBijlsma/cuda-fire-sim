#include <iostream>
#include <chrono>
#include "main.h"
#include <curand.h>
#include <curand_kernel.h>
#include <string>

//TODO:
//Fix cuda errors when size is large

#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void setupKernel(curandState *states) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(1234, id, 0, &states[id]);  // 	Initialize CURAND
}

__global__ void gpuTick(Cell *boardCopy,
                        const unsigned int width, const unsigned int height) {
    const unsigned int size = width * height;
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= size) return;
    boardCopy[id] = {
            1,
            2,
            3,
            4
    };
}

class Simulation {
private:
    unsigned int width;
    unsigned int height;
    unsigned int size;
    curandState *d_randState = nullptr;
    Cell *board;
    Cell *d_board = nullptr;
    Cell *d_boardCopy = nullptr;
    Params *d_params = nullptr;
    Params *params = (Params *) malloc(sizeof(Params));
    int nThreads;

public:
    Simulation(unsigned int w, unsigned int h, int threads = 1024) {
        nThreads = threads;
        width = w;
        height = h;
        size = w * h;
        board = new Cell[size];

        params->burnRate = .1;
        params->heightEffectMultiplierUp = 2;
        params->heightEffectMultiplierDown = 1;
        params->windEffectMultiplier = 1;
        params->activityThreshold = .2;
        params->spreadSpeed = 1.5;
        params->deathRate = .2;
        params->areaEffectMultiplier = 1;
        params->fireDeathThreshold = .1;
        //           nw     w     sw     s      n     ne     e     se
        float wm[8] = {1, 2, 3, 5, 0, 1, 2, 5};
        for (int i = 0; i < 8; i++)
            params->windMatrix[i] = wm[i];
        params->cellArea = 1;
        initBoard();
        initCuda();
    }

    ~Simulation() {
        freeCuda();
    }

    [[nodiscard]] unsigned int gridDim() const {
        return size / nThreads + 1;
    }

    void tick() {
        gpuTick<<<gridDim(), nThreads>>>(
                d_boardCopy, width, height
        );
        std::swap(d_board, d_boardCopy);
    }

    void initBoard() {
        for (int i = 0; i < size; i++)
            board[i] = {
                    1,
                    1,
                    1,
                    1,
            };
    }

    void initCuda() {
        size_t free, total;
        cudaCheck(cudaSetDevice(0));
        cudaCheck(cudaMemGetInfo(&free, &total));

        printf("Checking GPU MemInfo: free: %zu, total: %zu\n", free, total);

        // Init [size] random generators on GPU for each thread
        cudaCheck(cudaMalloc(&d_randState, gridDim() * nThreads * sizeof(curandState)));
        setupKernel<<<gridDim(), nThreads>>>(d_randState);

        // allocate gpu buffers for board and copy
        cudaCheck(cudaMalloc((void **) &d_board, size * sizeof(Cell)));
        cudaCheck(cudaMalloc((void **) &d_boardCopy, size * sizeof(Cell)));
        // copy board from CPU to GPU
        cudaCheck(cudaMemcpy(d_board, board, size * sizeof(Cell), cudaMemcpyHostToDevice));

        // allocate gpu buffers for params
        cudaCheck(cudaMalloc((void **) &d_params, sizeof(Params)));
        // copy params from CPU to GPU
        cudaCheck(cudaMemcpy(d_params, &params, sizeof(Params), cudaMemcpyHostToDevice));
    }

    void freeCuda() {
        cudaFree(d_board);
        cudaFree(d_boardCopy);
        cudaFree(d_randState);
    }
};

int findBestThreadCount(int W, int H) {
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;
    duration<double, std::milli> best = std::chrono::system_clock::duration::max();
    int bestN = -1;
    // 1st one: correct
    auto sim = Simulation(W, H, 32);
    sim.tick();

    // 2nd one: wrong call to cudaFree
    sim = Simulation(W, H, 32);
    sim.tick();

    // 3rd one: invalid memory read/write
    sim = Simulation(W, H, 32);
    sim.tick();

    return bestN;
}

int main() {
    findBestThreadCount(50, 50);

    return 0;
}