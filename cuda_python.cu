#include <iostream>
#include <chrono>
#include "main.h"
#include "pythonHelpers.h"
#include <curand.h>
#include <curand_kernel.h>
#include <string>

//TODO:
// Remove malloc on repeated Simulations

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

__global__ void gpuTick(curandState *randStates,
                        const Cell *board, Cell *boardCopy,
                        const NDimArray<double> *landCoverRates,
                        const NDimArray<double> *psoConfigs,
                        const unsigned int width,
                        const unsigned int height,
                        const unsigned int batchIndex
) {
    const auto size = width * height;
    auto id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= size) return;
    auto localRandState = randStates[id];

    auto x = id % width;
    auto y = id / width;

    Cell cell = board[id];
    auto newFuel =
            cell.fuel - cell.fireActivity * psoConfigs->array[Params::burnRate + batchIndex * psoConfigs->shape[0]];

    const int ns[3] = {-1, 0, 1};
    double activityGrid[8];
    auto dirIndex = -1;
    for (auto xOffset: ns) {
        for (auto yOffset: ns) {
            dirIndex++;
            if (xOffset == 0 && yOffset == 0)
                continue; // don't count same cell as neighbour
            // calculate neighbour coordinate
            auto nX = (int) x + xOffset;
            auto nY = (int) y + yOffset;
            if (nX >= width || nY >= height || nX < 0 || nY < 0) {
                activityGrid[dirIndex] = cell.fireActivity;
                continue;
            }
            auto nI = nY * width + nX;
            // ------ WIND ------
            // Fire activity from neighbour cell counts more if wind comes from there
            // todo WIND
            // ------ HEIGHT ------
            // Same but for height, going down decreases activity spread, going up increases it
            double heightDifference = cell.height - board[nI].height;
//             hD > 0 when neighbouring cell is higher than neighbour (fire would spread up)
//             hD < 0 when neighbouring cell is lower than neighbour (fire would spread down)
            heightDifference *= heightDifference > 0 ?
                                psoConfigs->array[Params::heightEffectMultiplierUp + batchIndex * paramCount] :
                                psoConfigs->array[Params::heightEffectMultiplierDown + batchIndex * paramCount];
            activityGrid[dirIndex] = activityGrid[dirIndex] * (heightDifference + 1);
        }
    }
    double activitySum = 0;
    for (auto activity: activityGrid)
        activitySum += activity;
    auto activity = (activitySum / 8) *
                    landCoverRates->array[cell.landCoverSpreadIndex + batchIndex * landCoverRates->shape[0]];
    auto newActivity = cell.fireActivity;
    auto randomNum = curand_uniform(&localRandState);
    auto cellArea = psoConfigs->array[Params::cellArea + batchIndex * psoConfigs->shape[0]];
    auto activityThreshold = psoConfigs->array[Params::activityThreshold + batchIndex * psoConfigs->shape[0]];
    auto areaEffectMultiplier = psoConfigs->array[Params::areaEffectMultiplier + batchIndex * psoConfigs->shape[0]];
    auto fireDeathThreshold = psoConfigs->array[Params::fireDeathThreshold + batchIndex * psoConfigs->shape[0]];
    if (activity > activityThreshold + randomNum / 5) {
        auto spreadSpeed = psoConfigs->array[Params::spreadSpeed + batchIndex * psoConfigs->shape[0]];
//        // Increase fire activity in current cell
        newActivity = cell.fuel * activity /
                      (cellArea / spreadSpeed * areaEffectMultiplier);
    } else if (activity <= fireDeathThreshold) {
        auto deathRate = psoConfigs->array[Params::deathRate + batchIndex * psoConfigs->shape[0]];
//        // Reduce fire activity in current cell
        newActivity /= 1 + (deathRate / (cellArea * areaEffectMultiplier));
    }

    boardCopy[id] = {
            newActivity,
            newFuel,
            cell.height,
            cell.landCoverSpreadIndex
    };
}

class Simulation {
private:
    unsigned int width;
    unsigned int height;
    unsigned int size;
    unsigned int batchIndex;
    Cell *board;
    NDimArray<short> *landCoverGrid{};
    NDimArray<short> *elevation{};
    NDimArray<bool> *fire{};
    NDimArray<double> *weather{};
    NDimArray<double> *psoConfigs{};
    NDimArray<double> *landCoverRates{};

    curandState *d_randState = nullptr;
    Cell *d_board = nullptr;
    Cell *d_boardCopy = nullptr;
    NDimArray<double> *d_weather = nullptr;
    NDimArray<double> *d_landCoverRates = nullptr;
    NDimArray<double> *d_psoConfigs = nullptr;
    int nThreads;

public:
    Simulation(unsigned int w, unsigned int h, unsigned int batchIndex, int threads,
               const NDimArray<short> &landCoverGrid,
               const NDimArray<short> &elevation,
               const NDimArray<bool> &fire,
               const NDimArray<double> &weather,
               const NDimArray<double> &psoConfigs,
               const NDimArray<double> &landCoverRates) {
        *this->landCoverGrid = landCoverGrid;
        *this->elevation = elevation;
        *this->fire = fire;
        *this->weather = weather;
        *this->psoConfigs = psoConfigs;
        *this->landCoverRates = landCoverRates;

        width = w;
        height = h;
        size = w * h;
        nThreads = threads;
        this->batchIndex = batchIndex;
        board = new Cell[size];

        initBoard();
        initCuda();
    }

    [[nodiscard]] unsigned int gridDim() const {
        return size / nThreads + 1;
    }

    void tick(bool print = true) {
        // Execute on GPU
        gpuTick<<<gridDim(), nThreads>>>(
                d_randState, d_board, d_boardCopy,
                d_landCoverRates, d_psoConfigs,
                width, height, batchIndex
        );

        if (print) {
            // Copy data back to CPU
            cudaCheck(cudaMemcpy(board, d_boardCopy, size * sizeof(Cell), cudaMemcpyDeviceToHost))
            printBoard();
        }

        std::swap(d_board, d_boardCopy);
    }

    void initBoard() {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                auto index = y * width + x;
                board[index] = {
                        fire->array[index] ? 1. : 0.,
                        1,
                        elevation->array[index],
                        landCoverGrid->array[index],
                };
            }
        }
    }

    void initCuda() {
        cudaCheck(cudaSetDevice(0));

//        size_t free, total;
//        cudaCheck(cudaMemGetInfo(&free, &total));
//        printf("Checking GPU MemInfo: free: %zu, total: %zu\n", free, total);

        // Init [size] random generators on GPU for each thread
        cudaCheck(cudaMalloc(&d_randState, gridDim() * nThreads * sizeof(curandState)))
        setupKernel<<<gridDim(), nThreads>>>(d_randState);

        // allocate gpu buffers for board and copy
        cudaCheck(cudaMalloc((void **) &d_board, size * sizeof(Cell)))
        cudaCheck(cudaMalloc((void **) &d_boardCopy, size * sizeof(Cell)))
        cudaCheck(cudaMalloc((void **) &d_weather, sizeof(weather)))
        cudaCheck(cudaMalloc((void **) &d_psoConfigs, sizeof(psoConfigs)))
        cudaCheck(cudaMalloc((void **) &d_landCoverRates, sizeof(landCoverRates)))
        // copy board from CPU to GPU
        cudaCheck(cudaMemcpy(d_board, board, size * sizeof(Cell), cudaMemcpyHostToDevice))
        cudaCheck(cudaMemcpy(d_weather, weather, sizeof(weather), cudaMemcpyHostToDevice))
    }

    void printBoard() {
        for (int j = 0; j < width * height; j++) {
            auto cell = board[j];
            if (cell.fireActivity > .5)
                printf("O ");
            else
                printf("_ ");
            if (j % width == width - 1)
                printf("\n");
        }
    }

    static void freeCuda() {
        printf("Free CUDA\n");
        cudaCheck(cudaDeviceReset());
    }
};

int findBestThreadCount(int W = 100, int H = 100) {
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;
    duration<double, std::milli> best = std::chrono::system_clock::duration::max();
    int bestN = -1;
    // warm up cuda boy
    auto sim = Simulation(W, H, 0, 32,);
    sim.tick(false);

    for (int n = 32; n <= 2048; n += 32) {
        auto t1 = high_resolution_clock::now();

        Simulation::freeCuda();
        sim = Simulation(W, H, n);
        for (int i = 0; i < 100; i++) {
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


int batchSimulate(NDimArray<short> landCoverGrid,
                  NDimArray<short> elevation,
                  NDimArray<bool> fire,
                  NDimArray<double> weather,
                  NDimArray<double> psoConfigs,
                  NDimArray<double> landCoverRates,
                  double output[]) {

    return 12;
}

int main() {
    return 0;
}

np::ndarray wrapBatchSimulate(np::ndarray const &npLandCoverGrid,
                              np::ndarray const &npElevation,
                              np::ndarray const &npFire,
                              np::ndarray const &npWeather,
                              np::ndarray const &npPsoConfigs,
                              np::ndarray const &npLandCoverRates) {
    // Make sure we get right types
    // 2D WxH array, each cell value is index for landCoverRates array
    // 2D WxH array
    // 3D WxHxC array C is fire checkpoint steps count
    // 4D WxHxTxE array T is time steps. E is elements size (wind X, wind Y)
    // 2D PxN array P is params count, N is batch size
    // 2D LxN array L is land cover type count, N is batch size

    printf("STARTING C++ ENGINES\n");

    auto landCoverGrid = npToArray<short>(npLandCoverGrid);
    auto elevation = npToArray<short>(npElevation);
    auto fire = npToArray<bool>(npFire);
    auto weather = npToArray<double>(npWeather);
    auto psoConfigs = npToArray<double>(npPsoConfigs);
    auto landCoverRates = npToArray<double>(npLandCoverRates);

    static double output[2];

    auto temp = batchSimulate(landCoverGrid, elevation, fire, weather, psoConfigs, landCoverRates, output);
//    auto temp = psoConfigs.array[0];
//    auto temp = weather.array[0];

    output[0] = temp;
    output[1] = temp * 12;
    np::dtype dt = np::dtype::get_builtin<double>();
    p::tuple shape = p::make_tuple(3); // It has shape (2,)
    p::tuple stride = p::make_tuple(sizeof(double)); // 1D array, so its just size of double
    np::ndarray result = np::from_data(output, dt, shape, stride, p::object());
    printf("FINITO\n");
    return result;
}

BOOST_PYTHON_MODULE (cuda_python) {  // Thing in brackets should match output library name
    Py_Initialize();
    np::initialize();
    p::def("batch_simulate", wrapBatchSimulate);
    p::def("find_best_thread_count", findBestThreadCount);
}