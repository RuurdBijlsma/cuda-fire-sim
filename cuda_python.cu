#include "main.h"
#include "pythonHelpers.h"
#include <curand.h>
#include <curand_kernel.h>
#include <string>

//TODO:
// Remove malloc on repeated Simulations
// add weather

#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void setupKernel(curandState *states, int seed) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, id, 0, &states[id]);  // 	Initialize CURAND
}

__global__ void gpuTick(curandState *randStates,
                        Cell *board, Cell *boardCopy,
                        const double *landCoverRates,
                        const double *params,
                        const double *weather,
                        const NDimArrayShape lcrShape,
                        const NDimArrayShape paramsShape,
                        const NDimArrayShape weatherShape,
                        const unsigned int batchIndex
) {
    const auto width = weatherShape.s0;
    const auto height = weatherShape.s1;
    const auto size = width * height;
    auto id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= size) return;
//    auto localRandState = randStates[id];

    auto x = id % width;
    auto y = id / width;

    Cell cell = board[id];
    auto newFuel = cell.fuel - cell.fireActivity * params[Params::burnRate + batchIndex * paramsShape.s0];

    const int ns[3] = {-1, 0, 1};
    double activityGrid[8];
    auto dirIndex = -1;
    // weather[x, y, t, e] t = batch index, e = weather element
    // e: 0 -> wind U component (horizontal towards east, +x)
    // e: 1 -> wind V component (vertical towards north, -y)
    auto windUElement = 0;
    auto windVElement = 1;
    auto windX = weather[
            windUElement * weatherShape.s0 * weatherShape.s1 * weatherShape.s2 +
            batchIndex * weatherShape.s0 * weatherShape.s1 +
            y * weatherShape.s0 +
            x
    ];
    auto windY = -1 * weather[
            windVElement * weatherShape.s0 * weatherShape.s1 * weatherShape.s2 +
            batchIndex * weatherShape.s0 * weatherShape.s1 +
            y * weatherShape.s0 +
            x
    ];
    for (auto xOffset: ns) {
        for (auto yOffset: ns) {
            dirIndex++;
            if (xOffset == 0 && yOffset == 0)
                continue; // skip current cell because it's not a neighbour
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
            activityGrid[dirIndex] = board[nI].fireActivity;
            auto wx = windX * xOffset * -1;
            auto wy = windY * yOffset * -1;
            // try to keep windFromNeighbour between 0 and 1
            auto windFromNeighbour =
                    (wx + wy) / 200 * params[Params::windEffectMultiplier + batchIndex * paramCount];
            activityGrid[dirIndex] *= 1 + windFromNeighbour;
            // ------ HEIGHT ------
            // Same but for height, going down decreases activity spread, going up increases it
            double heightDifference = (double) (cell.height - board[nI].height) / 20;
//             hD > 0 when neighbouring cell is higher than neighbour (fire would spread up)
//             hD < 0 when neighbouring cell is lower than neighbour (fire would spread down)
            heightDifference *= heightDifference > 0 ?
                                params[Params::heightEffectMultiplierUp + batchIndex * paramCount] :
                                params[Params::heightEffectMultiplierDown + batchIndex * paramCount];
            activityGrid[dirIndex] *= 1 + heightDifference;
        }
    }
    double activitySum = 0;
    for (auto activity: activityGrid)
        activitySum += activity;
    auto activity = (activitySum / 8) *
                    landCoverRates[cell.landCoverSpreadIndex + batchIndex * lcrShape.s0];
    auto newActivity = cell.fireActivity;
    auto randomNum = curand_uniform(randStates + id);
    auto cellArea = params[Params::cellArea + batchIndex * paramsShape.s0];
    auto activityThreshold = params[Params::activityThreshold + batchIndex * paramsShape.s0];
    auto areaEffectMultiplier = params[Params::areaEffectMultiplier + batchIndex * paramsShape.s0];
    auto fireDeathThreshold = params[Params::fireDeathThreshold + batchIndex * paramsShape.s0];
    if (activity > activityThreshold + randomNum / 5) {
        auto spreadSpeed = params[Params::spreadSpeed + batchIndex * paramsShape.s0];
//        // Increase fire activity in current cell
        newActivity = cell.fuel * activity /
                      (cellArea / spreadSpeed * areaEffectMultiplier);
    } else if (activity <= fireDeathThreshold) {
        auto deathRate = params[Params::deathRate + batchIndex * paramsShape.s0];
//        // Reduce fire activity in current cell
        newActivity /= 1 + (deathRate / (cellArea * areaEffectMultiplier));
    }

    boardCopy[id].fireActivity = newActivity;
    boardCopy[id].fuel = newFuel;
    boardCopy[id].height = cell.height;
    boardCopy[id].landCoverSpreadIndex = cell.landCoverSpreadIndex;
}

class Simulation {
private:
    unsigned int width;
    unsigned int height;
    unsigned int size;
    unsigned int batchIndex;
    Cell *board;
    NDimArray<short> landCoverGrid{};
    NDimArray<short> elevation{};
    NDimArray<bool> fire{};
    NDimArray<double> weather{};
    NDimArray<double> psoConfigs{};
    NDimArray<double> landCoverRates{};

    NDimArrayShape lcrShape{};
    NDimArrayShape paramsShape{};
    NDimArrayShape weatherShape{};

    curandState *d_randState = nullptr;
    Cell *d_board = nullptr;
    Cell *d_boardCopy = nullptr;
    double *d_weather = nullptr;
    double *d_landCoverRates = nullptr;
    double *d_params = nullptr;
    int nThreads;

public:
    Simulation(unsigned int w, unsigned int h, unsigned int batchIndex, int threads,
               const NDimArray<short> &landCoverGrid,
               const NDimArray<short> &elevation,
               const NDimArray<bool> &fire,
               const NDimArray<double> &weather,
               const NDimArray<double> &params,
               const NDimArray<double> &landCoverRates) {
        this->landCoverGrid = landCoverGrid;
        this->elevation = elevation;
        this->fire = fire;
        this->weather = weather;
        this->psoConfigs = params;
        this->landCoverRates = landCoverRates;
        this->lcrShape.s0 = landCoverGrid.shape[0];
        this->lcrShape.s1 = landCoverGrid.shape[1];
        this->paramsShape.s0 = params.shape[0];
        this->paramsShape.s1 = params.shape[1];
        this->weatherShape.s0 = weather.shape[0];
        this->weatherShape.s1 = weather.shape[1];
        this->weatherShape.s2 = weather.shape[2];
        this->weatherShape.s3 = weather.shape[3];

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
                d_landCoverRates, d_params, d_weather,
                lcrShape, paramsShape, weatherShape,
                batchIndex
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
                        fire.array[index] ? 1. : 0.,
                        1,
                        elevation.array[index],
                        landCoverGrid.array[index],
                };
            }
        }
    }

    void initCuda() {
        cudaCheck(cudaSetDevice(0))

//        size_t free, total;
//        cudaCheck(cudaMemGetInfo(&free, &total));
//        printf("Checking GPU MemInfo: free: %zu, total: %zu\n", free, total);

        // Init [size] random generators on GPU for each thread
        cudaCheck(cudaMalloc(&d_randState, gridDim() * nThreads * sizeof(curandState)))
        setupKernel<<<gridDim(), nThreads>>>(d_randState, rand()); // NOLINT(cert-msc50-cpp)

        auto weatherSize = nDimArrayLength(weather) * sizeof(double);
        auto psoConfigsSize = nDimArrayLength(psoConfigs) * sizeof(double);
        auto landCoverRatesSize = nDimArrayLength(landCoverRates) * sizeof(double);

        printf("nDimArrayLength(weather) = %lu\n", nDimArrayLength(weather));

        // allocate gpu buffers for board and copy
        cudaCheck(cudaMalloc((void **) &d_board, size * sizeof(Cell)))
        cudaCheck(cudaMalloc((void **) &d_boardCopy, size * sizeof(Cell)))
        cudaCheck(cudaMalloc((void **) &d_weather, weatherSize))
        cudaCheck(cudaMalloc((void **) &d_params, psoConfigsSize))
        cudaCheck(cudaMalloc((void **) &d_landCoverRates, landCoverRatesSize))
        // copy board from CPU to GPU
        cudaCheck(cudaMemcpy(d_board, board, size * sizeof(Cell), cudaMemcpyHostToDevice))
        cudaCheck(cudaMemcpy(d_weather, weather.array, weatherSize, cudaMemcpyHostToDevice))
        cudaCheck(cudaMemcpy(d_params, psoConfigs.array, psoConfigsSize, cudaMemcpyHostToDevice))
        cudaCheck(cudaMemcpy(d_landCoverRates, landCoverRates.array, landCoverRatesSize, cudaMemcpyHostToDevice))
    }

    void printBoard() {
        for (int j = 0; j < width * height; j++) {
            auto cell = board[j];
            if (cell.fireActivity <= 0)
                printf(". ");
            else if (cell.fireActivity <= .3)
                printf("c ");
            else if (cell.fireActivity <= .6)
                printf("o ");
            else
                printf("O ");
            if (j % width == width - 1)
                printf("\n");
        }
    }

    static void freeCuda() {
        printf("Free CUDA\n");
        cudaCheck(cudaDeviceSynchronize())
        cudaCheck(cudaDeviceReset())
    }
};

int batchSimulate(NDimArray<short> landCoverGrid,
                  NDimArray<short> elevation,
                  NDimArray<bool> fire,
                  NDimArray<double> weather,
                  NDimArray<double> psoConfigs,
                  NDimArray<double> landCoverRates,
                  double output[]) {
    auto width = landCoverGrid.shape[0];
    auto height = landCoverGrid.shape[1];
    auto batchSize = psoConfigs.shape[1];
    auto timeSteps = weather.shape[2];
    for (int i = 0; i < batchSize; i++) {
        printf("Iteration %i\n", i);
        auto sim = Simulation(width, height, i, 96,
                              landCoverGrid, elevation, fire, weather, psoConfigs, landCoverRates);
        sim.printBoard();
        for (int t = 0; t < timeSteps; t++) {
            printf("Tick %i\n", t);
            sim.tick(true);
        }
        Simulation::freeCuda();
    }

    return 0;
}

int main() {
    int width = 10;
    int height = 8;
    int timeSteps = 20;
    int checkpoints = 3;
    int weatherElements = 2;
    int psoParams = 10;
    int batchSize = 1;
    int landCoverTypes = 8;

    auto landCoverGrid = createNDimArray<short>(2, new long[2]{width, height}, 1);
    auto landCoverRates = createNDimArray<double>(2, new long[2]{width, height}, 1);
    auto elevation = createNDimArray<short>(2, new long[2]{landCoverTypes, batchSize}, 3);
    auto fire = createNDimArray<bool>(3, new long[3]{width, height, checkpoints}, false);
    auto weather = createNDimArray<double>(4, new long[4]{width, height, timeSteps, weatherElements}, 20);
    auto params = createNDimArray<double>(2, new long[2]{psoParams, batchSize}, 1);

    fire.array[height / 2 * width + width / 2] = true;
    fire.array[(height / 2 + 1) * width + width / 2] = true;
    fire.array[height / 2 * width + width / 2 + 1] = true;
    fire.array[(height / 2 + 1) * width + width / 2 + 1] = true;
//    fire.array[1] = true;
//    fire.array[1 * width + 1] = true;
//    fire.array[1 * width + 0] = true;

    for (int x = 0; x < weather.shape[0]; x++) {
        for (int y = 0; y < weather.shape[1]; y++) {
            for (int z = 0; z < weather.shape[2]; z++) {
                weather.array[1 * weather.shape[0] * weather.shape[1] * weather.shape[2] +
                              z * weather.shape[1] * weather.shape[0] +
                              y * weather.shape[0] +
                              x] = -0;
            }
        }
    }

    params.array[Params::activityThreshold] = 0.2;
    params.array[Params::burnRate] = 0.1;
    params.array[Params::fireDeathThreshold] = 0.1;
    params.array[Params::deathRate] = 0.2;
    params.array[Params::areaEffectMultiplier] = 1;
    params.array[Params::heightEffectMultiplierDown] = 1;
    params.array[Params::heightEffectMultiplierUp] = 1;
    params.array[Params::spreadSpeed] = 1.5;
    params.array[Params::windEffectMultiplier] = 3;

//    printNDimArray(weather, "Weather");

    static double output[2];
    auto temp = batchSimulate(landCoverGrid, elevation, fire, weather, params, landCoverRates, output);
    printf("bs output %i\n", temp);
    printf("output again %f %f\n", output[0], output[1]);

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

    printf("Sizeof landCoverGrid = %lu\n", sizeOfNDimArray(landCoverGrid));
    printf("Sizeof elevation = %lu\n", sizeOfNDimArray(elevation));
    printf("Sizeof fire = %lu\n", sizeOfNDimArray(fire));
    printf("Sizeof weather = %lu\n", sizeOfNDimArray(weather));
    printf("Sizeof psoConfigs = %lu\n", sizeOfNDimArray(psoConfigs));
    printf("Sizeof landCoverRates = %lu\n", sizeOfNDimArray(landCoverRates));

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
//    p::def("find_best_thread_count", findBestThreadCount);
}