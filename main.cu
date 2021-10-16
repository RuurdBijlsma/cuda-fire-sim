#include <iostream>
#include <random>
#include <chrono>
#include "main.h"

__global__ void gpuTick(const Cell *board, Cell *boardCopy,
                        const unsigned int width, const unsigned int height) {
    const unsigned int size = width * height;
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= size) return;

    unsigned int x = id % width;
    unsigned int y = id / width;
    int ns[3] = {-1, 0, 1};
    for (int xOffset: ns) {
        for (int yOffset: ns) {
            if (xOffset == 0 && yOffset == 0)
                continue;
            unsigned int nX = x + width + xOffset;
            while (nX >= width)
                nX -= width;
            unsigned int nY = y + height + yOffset;
            while (nY >= height)
                nY -= height;
            unsigned int nI = nY * width + nX;
            Cell neighbour = board[nI];
        }
    }
}

class Simulation {
private:
    unsigned int width;
    unsigned int height;
    unsigned int size;
    Cell *board;
    Cell *d_board = nullptr;
    Cell *d_boardCopy = nullptr;
    cudaError_t cudaStatus = cudaError_t();
    int nThreads;

public:
    Simulation(unsigned int w, unsigned int h, int threads = 1024) {
        nThreads = threads;
        width = w;
        height = h;
        size = w * h;
        board = new Cell[size];
        initBoard();
        if (!initCuda())
            return;
    }

    bool tick(bool print = true) {
        // Execute on GPU
        gpuTick<<<size / nThreads + 1, nThreads>>>(d_board, d_boardCopy, width, height);

        if (print) {
            // Copy data back to CPU
            cudaStatus = cudaMemcpy(board, d_boardCopy, size * sizeof(int), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess)
                return handleError("memCpy to CPU");
            printBoard();
        }

        std::swap(d_board, d_boardCopy);
        return true;
    }

    void initBoard() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dist(0, 1);
        for (int i = 0; i < size; i++)
            board[i] = {
                    1,
                    1,
                    1,
                    1,
            };
    }

    bool initCuda() {
        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess)
            return handleError("set device");

        // allocate gpu buffers for board and copy
        cudaStatus = cudaMalloc((void **) &d_board, size * sizeof(int));
        if (cudaStatus != cudaSuccess)
            return handleError("malloc devBoard");

        cudaStatus = cudaMalloc((void **) &d_boardCopy, size * sizeof(int));
        if (cudaStatus != cudaSuccess)
            return handleError("malloc devBoardCopy");

        // copy board from CPU to GPU
        cudaStatus = cudaMemcpy(d_board, board, size * sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
            return handleError("memCpy to GPU");

        return true;
    }

    void printBoard() {
        for (int j = 0; j < width * height; j++) {
            if (board[j].fireActivity > .5)
                printf("O ");
            else
                printf("_ ");
            if (j % width == width - 1)
                printf("\n");
        }
    }

    void freeCuda() {
        cudaFree(d_board);
        cudaFree(d_boardCopy);
    }

    bool handleError(const std::string &reason) {
        freeCuda();
        printf("Cuda error! %s", reason.c_str());
        return false;
    }
};

int findBestThreadCount() {
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;
    duration<double, std::milli> best = std::chrono::system_clock::duration::max();
    int bestN = -1;
    // warm up cuda boy
    auto sim = Simulation(20, 20, 1);
    sim.tick(false);

    for (int n = 32; n <= 1024; n += 32) {
        auto t1 = high_resolution_clock::now();

        sim = Simulation(20, 20, n);
        for (int i = 0; i < 1000; i++) {
            sim.tick(false);
        }

        auto t2 = high_resolution_clock::now();
        /* Getting number of milliseconds as a double. */
        duration<double, std::milli> dur = t2 - t1;
        std::cout << dur.count() << "ms - " << n << " threads" << std::endl;
        if (dur < best) {
            best = dur;
            bestN = n;
        }
    }
    printf("Best thread count is %i with duration %fms\n", bestN, best.count());
    return bestN;
}

int main() {
    int nThreads = findBestThreadCount();

    auto sim = Simulation(20, 20, nThreads);
    for (int i = 0; i < 50; i++) {
        printf("----------- ITERATION %i -----------\n", i + 1);
        sim.tick(true);
    }
    return 0;
}
