#include <iostream>
#include <chrono>
#include "main.h"
#include <curand.h>
#include <curand_kernel.h>
#include <string>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

//TODO:
//Fix cuda errors when size is large

__global__ void setup_kernel(curandState *state) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    curand_init(1234, idx, 0, &state[idx]);
}

__global__ void gpuTick(curandState *randState, const Cell *board, Cell *boardCopy, const Params *params,
                        const unsigned int width, const unsigned int height) {
    const unsigned int size = width * height;
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= size) return;

    unsigned int x = id % width;
    unsigned int y = id / width;

    Cell cell = board[id];
    float newFuel = cell.fuel - cell.fireActivity * params->burnRate;

    const int ns[3] = {-1, 0, 1};
    float activityGrid[8];
    int dirIndex = -1;
    for (int xOffset: ns) {
        for (int yOffset: ns) {
            dirIndex++;
            if (xOffset == 0 && yOffset == 0)
                continue; // don't count same cell as neighbour
            // calculate neighbour coordinate
            int nX = (int) x + xOffset;
            int nY = (int) y + yOffset;
            if (nX >= width || nY >= height || nX < 0 || nY < 0) {
                activityGrid[dirIndex] = cell.fireActivity;
                continue;
            }
            unsigned int nI = nY * width + nX;
            // ------ WIND ------
            // Fire activity from neighbour cell counts more if wind comes from there
            activityGrid[dirIndex] =
                    board[nI].fireActivity * (1 + params->windMatrix[dirIndex] * params->windEffectMultiplier);
            // ------ HEIGHT ------
            // Same but for height, going down decreases activity spread, going up increases it
            float heightDifference = cell.height - board[nI].height;
            // hD > 0 when neighbouring cell is higher than neighbour (fire would spread up)
            // hd < 0 when neighbouring cell is lower than neighbour (fire would spread down)
            heightDifference *= heightDifference > 0 ?
                                params->heightEffectMultiplierUp :
                                params->heightEffectMultiplierDown;
            activityGrid[dirIndex] = activityGrid[dirIndex] * (heightDifference + 1);
        }
    }
    float activitySum = 0;
    for (float activity: activityGrid)
        activitySum += activity;
    float activity = (activitySum / 8) * cell.landCoverSpreadRate;
    float newActivity = cell.fireActivity;
    if (activity > params->activityThreshold + curand_uniform(randState + id) / 5) {
        // Increase fire activity in current cell
        newActivity = cell.fuel * activity /
                      (params->cellArea / params->spreadSpeed * params->areaEffectMultiplier);
    } else if (activity <= params->fireDeathThreshold) {
        // Reduce fire activity in current cell
        newActivity /= 1 + (params->deathRate / (params->cellArea * params->areaEffectMultiplier));
    }

    boardCopy[id] = {
            newActivity,
            newFuel,
            cell.height, cell.landCoverSpreadRate
    };
}

class Simulation {
private:
    unsigned int width;
    unsigned int height;
    unsigned int size;
    curandState *d_randState = nullptr;
    bool cudaFailed = false;
    Cell *board;
    Cell *d_board = nullptr;
    Cell *d_boardCopy = nullptr;
    Params *d_params = nullptr;
    Params params{};
    cudaError_t cudaStatus = cudaError_t();
    int nThreads;

public:
    Simulation(unsigned int w, unsigned int h, int threads = 1024) {
        nThreads = threads;
        width = w;
        height = h;
        size = w * h;
        board = new Cell[size];
        params = {
                .1,
                2,
                1,
                1,
                .2,
                1.5,
                .2,
                1,
                .1,
                //           nw     w     sw     s      n     ne     e     se
                {1, 2, 3, 5, 0, 1, 2, 5},
        };
        initBoard();
        if (!initCuda()) {
            return;
        }
    }

    ~Simulation() {
        freeCuda();
    }

    bool tick(bool print = true) {
        if (cudaFailed)return false;
        // Execute on GPU
        gpuTick<<<size / nThreads + 1, nThreads>>>(
                d_randState, d_board, d_boardCopy, d_params,
                width, height
        );

        if (print) {
            // Copy data back to CPU
            cudaStatus = cudaMemcpy(board, d_boardCopy, size * sizeof(Cell), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess)
                return handleError("memCpy to CPU");
            printBoard();
        }

        std::swap(d_board, d_boardCopy);
        return true;
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

    bool initCuda() {
        size_t free, total;
        cudaStatus = cudaMemGetInfo(&free, &total);
        if (cudaStatus != cudaSuccess)
            return handleError("get mem info");

        printf("Checking GPU MemInfo: free: %zu, total: %zu\n", free, total);

        // Init random generator on GPU
        cudaStatus = cudaMalloc(&d_randState, sizeof(curandState));
        if (cudaStatus != cudaSuccess)
            return handleError("malloc random");
        setup_kernel<<<size / nThreads + 1, nThreads>>>(d_randState);

        //
        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess)
            return handleError("set device");

        // allocate gpu buffers for board and copy
        cudaStatus = cudaMalloc((void **) &d_board, size * sizeof(Cell));
        if (cudaStatus != cudaSuccess)
            return handleError("malloc d_board");

        cudaStatus = cudaMalloc((void **) &d_boardCopy, size * sizeof(Cell));
        if (cudaStatus != cudaSuccess)
            return handleError("malloc d_boardCopy");

        // copy board from CPU to GPU
        cudaStatus = cudaMemcpy(d_board, board, size * sizeof(Cell), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
            return handleError("memCpy to GPU");

        // allocate gpu buffers for board and copy
        cudaStatus = cudaMalloc((void **) &d_params, sizeof(Params));
        if (cudaStatus != cudaSuccess)
            return handleError("malloc d_params");

        // copy board from CPU to GPU
        cudaStatus = cudaMemcpy(d_params, &params, sizeof(Params), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
            return handleError("memCpy d_params to GPU");

        return true;
    }

    void printBoard() {
        if (cudaFailed)return;
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

    void freeCuda() {
        cudaFree(d_board);
        cudaFree(d_boardCopy);
        cudaFree(d_randState);
        printf("FREEING CUDA\n");
    }

    bool handleError(const std::string &reason) {
        cudaFailed = true;
        freeCuda();
        printf("Cuda error! %s\n", reason.c_str());
        return false;
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
    auto sim = Simulation(W, H, 1);
    sim.tick(false);

    for (int n = 32; n <= 1024; n += 32) {
        auto t1 = high_resolution_clock::now();

        sim = Simulation(W, H, n);
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

int batchSimulate(short *landCoverGrid,
                  short *elevation,
                  bool **fire,
                  double ***weather,
                  double *psoConfigs,
                  double *landCoverRates,
                  int width, int height, int timeSteps, int checkpoints,
                  int weatherElements, int psoParams, int landCoverTypes, int batchSize,
                  double output[]) {
    for (int b = 0; b < batchSize; b++) {

    }
    return 12;
}

namespace p = boost::python;
namespace np = boost::python::numpy;


np::ndarray wrapBatchSimulate(np::ndarray const &npLandCoverGrid,
                              np::ndarray const &npElevation,
                              np::ndarray const &npFire,
                              np::ndarray const &npWeather,
                              np::ndarray const &npPsoConfigs,
                              np::ndarray const &npLandCoverRates) {
    // Make sure we get right types
    // 2D WxH array, each cell value is index for landCoverRates array
    if (npLandCoverGrid.get_dtype() != np::dtype::get_builtin<short>()) {
        PyErr_SetString(PyExc_TypeError, "Incorrect landCoverGrid data type");
        p::throw_error_already_set();
    }
    // 2D WxH array
    if (npElevation.get_dtype() != np::dtype::get_builtin<short>()) {
        PyErr_SetString(PyExc_TypeError, "Incorrect elevation data type");
        p::throw_error_already_set();
    }
    // 3D WxHxC array C is fire checkpoint steps count
    if (npFire.get_dtype() != np::dtype::get_builtin<bool>()) {
        PyErr_SetString(PyExc_TypeError, "Incorrect fire data type");
        p::throw_error_already_set();
    }
    // 4D WxHxTxE array T is time steps. E is elements size (wind X, wind Y)
    if (npWeather.get_dtype() != np::dtype::get_builtin<double>()) {
        PyErr_SetString(PyExc_TypeError, "Incorrect weather data type");
        p::throw_error_already_set();
    }

    // 2D PxN array P is params count, N is batch size
    if (npPsoConfigs.get_dtype() != np::dtype::get_builtin<double>()) {
        PyErr_SetString(PyExc_TypeError, "Incorrect psoConfigs data type");
        p::throw_error_already_set();
    }
    // 2D LxN array L is land cover type count, N is batch size
    if (npLandCoverRates.get_dtype() != np::dtype::get_builtin<double>()) {
        PyErr_SetString(PyExc_TypeError, "Incorrect landCoverRates data type");
        p::throw_error_already_set();
    }

    int width = (int) npLandCoverGrid.shape(0);
    auto height = (int) npLandCoverGrid.shape(1);
    auto timeSteps = (int) npWeather.shape(2);
    auto checkpoints = (int) npFire.shape(2);
    auto weatherElements = (int) npWeather.shape(3);
    auto psoParams = (int) npPsoConfigs.shape(0);
    auto batchSize = (int) npPsoConfigs.shape(1);
    auto landCoverTypes = (int) npLandCoverRates.shape(0);

    auto landCoverGrid = reinterpret_cast<short *>(npLandCoverGrid.get_data());
    auto elevation = reinterpret_cast<short *>(npElevation.get_data());
    auto fire = reinterpret_cast<bool **>(npFire.get_data());
    auto weather = reinterpret_cast<double ***>(npWeather.get_data());
    auto psoConfigs = reinterpret_cast<double *>(npPsoConfigs.get_data());
    auto landCoverRates = reinterpret_cast<double *>(npLandCoverRates.get_data());

    static double output[2];

    auto temp = batchSimulate(landCoverGrid, elevation, fire, weather, psoConfigs, landCoverRates,
                              width, height, timeSteps, checkpoints,
                              weatherElements, psoParams, landCoverTypes, batchSize, output);

    output[0] = temp;
    output[1] = temp * 12;
    np::dtype dt = np::dtype::get_builtin<double>();
    p::tuple shape = p::make_tuple(3); // It has shape (2,)
    p::tuple stride = p::make_tuple(sizeof(double)); // 1D array, so its just size of double
    np::ndarray result = np::from_data(output, dt, shape, stride, p::object());
    return result;
}

BOOST_PYTHON_MODULE (cuda_python) {  // Thing in brackets should match output library name
    Py_Initialize();
    np::initialize();
    p::def("batch_simulate", wrapBatchSimulate);
    p::def("find_best_thread_count", findBestThreadCount);
}