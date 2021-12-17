#include <iostream>
#include <chrono>
#include "main.h"
#include <curand.h>
#include <curand_kernel.h>
#include <string>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

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

__global__ void gpuTick(curandState *randStates, const Cell *board, Cell *boardCopy, const Params *params,
                        const unsigned int width, const unsigned int height) {
    const unsigned int size = width * height;
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= size) return;
    curandState localRandState = randStates[id];

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
//             hD > 0 when neighbouring cell is higher than neighbour (fire would spread up)
//             hD < 0 when neighbouring cell is lower than neighbour (fire would spread down)
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
    float randomNum = curand_uniform(&localRandState);
//    float randomNum = .5;
    if (activity > params->activityThreshold + randomNum / 5) {
//        // Increase fire activity in current cell
        newActivity = cell.fuel * activity /
                      (params->cellArea / params->spreadSpeed * params->areaEffectMultiplier);
    } else if (activity <= params->fireDeathThreshold) {
//        // Reduce fire activity in current cell
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

    [[nodiscard]] unsigned int gridDim() const {
        return size / nThreads + 1;
    }

    void tick(bool print = true) {
        // Execute on GPU
        gpuTick<<<gridDim(), nThreads>>>(
                d_randState, d_board, d_boardCopy, d_params,
                width, height
        );
//        cudaCheck( cudaPeekAtLastError() );
//        cudaCheck( cudaDeviceSynchronize() );

        if (print) {
            // Copy data back to CPU
            cudaCheck(cudaMemcpy(board, d_boardCopy, size * sizeof(Cell), cudaMemcpyDeviceToHost));
            printBoard();
        }

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
        cudaCheck(cudaSetDevice(0));

//        size_t free, total;
//        cudaCheck(cudaMemGetInfo(&free, &total));
//        printf("Checking GPU MemInfo: free: %zu, total: %zu\n", free, total);

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
    auto sim = Simulation(W, H, 32);
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

template<class T>
struct NDimArray {
    T *array;
    int nDims;
    long *shape;
};

int batchSimulate(NDimArray<short> landCoverGrid,
                  NDimArray<short> elevation,
                  NDimArray<bool> fire,
                  NDimArray<double> weather,
                  NDimArray<double> psoConfigs,
                  NDimArray<double> landCoverRates,
                  double output[]) {
    long width = landCoverGrid.shape[0];
    long height = landCoverGrid.shape[1];
    printf("Width %ld, height %ld\n", width, height);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            printf("%i ", landCoverGrid.array[y * width + x]);
        }
        printf("\n");
    }
    printf("12\n");
    return 12;
}

int main() {
    return 0;
}

namespace p = boost::python;
namespace np = boost::python::numpy;

unsigned long getOffset(int nDims, const long *strides, const int *indices, const size_t typeSize) {
    unsigned long offset = 0;
    for (int i = 0; i < nDims; i++) {
        offset += indices[i] * strides[i] / typeSize;
    }
    return offset;
}

template<typename T>
NDimArray<T> npToArray(const np::ndarray &npArray) {
    if (npArray.get_dtype() != np::dtype::get_builtin<T>()) {
        PyErr_SetString(PyExc_TypeError, "Incorrect data type for np array");
        p::throw_error_already_set();
    }

    auto nDims = npArray.get_nd();
    auto strides = npArray.get_strides();
    long *shape = const_cast<long *>(npArray.get_shape());
    long size = 1;
    for (int i = 0; i < nDims; i++) {
        printf("Shape[%i] = %li\n", i, shape[i]);
        size *= shape[i];
    }
    auto mallocSize = size * sizeof(T);
    printf("array size %lu\n", size);
    printf("sizeof T %lu\n", sizeof(T));
    printf("Malloc size %lu\n", mallocSize);
    printf("nDims %d\n", nDims);
    T *result = static_cast<double *>(malloc(mallocSize));
    T *strideArray = reinterpret_cast<T *>(npArray.get_data());
    result[0] = strideArray[0];
    if (nDims == 1) {
        for (int x = 0; x < shape[0]; x++) {
            int indices[1] = {x};
            auto offset = getOffset(nDims, strides, indices, sizeof(T));
            result[x] = strideArray[offset];
        }
    }
    if (nDims == 2) {
        for (int x = 0; x < shape[0]; x++) {
            for (int y = 0; y < shape[1]; y++) {
                int indices[2] = {x, y};
                auto offset = getOffset(nDims, strides, indices, sizeof(T));
                result[y * shape[0] + x] = strideArray[offset];
            }
        }
    }
    if (nDims == 3) {
        for (int x = 0; x < shape[0]; x++) {
            for (int y = 0; y < shape[1]; y++) {
                for (int z = 0; z < shape[2]; z++) {
                    int indices[3] = {x, y, z};
                    auto offset = getOffset(nDims, strides, indices, sizeof(T));
                    result[z * shape[0] * shape[1]
                           + y * shape[0]
                           + x] = strideArray[offset];
                }
            }
        }
    }
    if (nDims == 4) {
        for (int x = 0; x < shape[0]; x++) {
            for (int y = 0; y < shape[1]; y++) {
                for (int z = 0; z < shape[2]; z++) {
                    for (int q = 0; q < shape[3]; q++) {
                        int indices[4] = {x, y, z, q};
                        auto offset = getOffset(nDims, strides, indices, sizeof(T));
                        auto index = q * shape[0] * shape[1] * shape[2]
                                     + z * shape[0] * shape[1]
                                     + y * shape[0]
                                     + x;
                        result[index] = strideArray[offset];
                    }
                }
            }
        }
    }
    return NDimArray<T>{
            result,
            nDims,
            shape,
    };
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

//    auto landCoverGrid = npToArray<short>(npLandCoverGrid);
//    auto elevation = npToArray<short>(npElevation);
//    auto fire = npToArray<bool>(npFire);
    auto weather = npToArray<double>(npWeather);
//    auto psoConfigs = npToArray<double>(npPsoConfigs);
//    auto landCoverRates = npToArray<double>(npLandCoverRates);

    static double output[2];

//    auto temp = batchSimulate(landCoverGrid, elevation, fire, weather, psoConfigs, landCoverRates, output);
//    auto temp = psoConfigs.array[0];
    auto temp = weather.array[0];

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