#include "main.h"
#include "helpers.h"
#include <curand.h>
#include <curand_kernel.h>
#include <string>
#include <random>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;

//TODO:
// Remove malloc on repeated Simulations

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
    if (newFuel < 0)
        newFuel = 0;

    const int ns[3] = {-1, 0, 1};
    double activityGrid[8];
    auto dirIndex = -1;
    // weather[x, y, t, e], t = batch index, e = weather element
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
                    (wx + wy) / 60 * params[Params::windEffectMultiplier + batchIndex * paramCount];
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

void cpuTick(Cell *board, Cell *boardCopy,
             const NDimArray<double> landCoverRates,
             const NDimArray<double> params,
             const NDimArray<double> weather,
             const unsigned int batchIndex, const unsigned int x, const unsigned int y
) {
    const auto width = weather.shape[0];
    const auto height = weather.shape[1];
    const unsigned int id = y * width + x;
//    auto localRandState = randStates[id];

    Cell cell = board[id];
    auto newFuel = cell.fuel - cell.fireActivity * params.array[Params::burnRate + batchIndex * params.shape[0]];

    const int ns[3] = {-1, 0, 1};
    double activityGrid[8];
    auto dirIndex = -1;
    // weather[x, y, t, e] t = batch index, e = weather element
    // e: 0 -> wind U component (horizontal towards east, +x)
    // e: 1 -> wind V component (vertical towards north, -y)
    auto windUElement = 0;
    auto windVElement = 1;
    auto windX = weather.array[
            windUElement * weather.shape[0] * weather.shape[1] * weather.shape[2] +
            batchIndex * weather.shape[0] * weather.shape[1] +
            y * weather.shape[0] +
            x
    ];
    auto windY = -1 * weather.array[
            windVElement * weather.shape[0] * weather.shape[1] * weather.shape[2] +
            batchIndex * weather.shape[0] * weather.shape[1] +
            y * weather.shape[0] +
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
                    (wx + wy) / 60 * params.array[Params::windEffectMultiplier + batchIndex * paramCount];
            activityGrid[dirIndex] *= 1 + windFromNeighbour;
            // ------ HEIGHT ------
            // Same but for height, going down decreases activity spread, going up increases it
            double heightDifference = (double) (cell.height - board[nI].height) / 20;
//             hD > 0 when neighbouring cell is higher than neighbour (fire would spread up)
//             hD < 0 when neighbouring cell is lower than neighbour (fire would spread down)
            heightDifference *= heightDifference > 0 ?
                                params.array[Params::heightEffectMultiplierUp + batchIndex * paramCount] :
                                params.array[Params::heightEffectMultiplierDown + batchIndex * paramCount];
            activityGrid[dirIndex] *= 1 + heightDifference;
        }
    }
    double activitySum = 0;
    for (auto activity: activityGrid)
        activitySum += activity;
    auto activity = (activitySum / 8) *
                    landCoverRates.array[cell.landCoverSpreadIndex + batchIndex * landCoverRates.shape[0]];
    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(0, 1);

    auto newActivity = cell.fireActivity;
    auto randomNum = dist(e2);
    auto cellArea = params.array[Params::cellArea + batchIndex * params.shape[0]];
    auto activityThreshold = params.array[Params::activityThreshold + batchIndex * params.shape[0]];
    auto areaEffectMultiplier = params.array[Params::areaEffectMultiplier + batchIndex * params.shape[0]];
    auto fireDeathThreshold = params.array[Params::fireDeathThreshold + batchIndex * params.shape[0]];
    if (activity > activityThreshold + randomNum / 5) {
        auto spreadSpeed = params.array[Params::spreadSpeed + batchIndex * params.shape[0]];
//        // Increase fire activity in current cell
        newActivity = cell.fuel * activity /
                      (cellArea / spreadSpeed * areaEffectMultiplier);
    } else if (activity <= fireDeathThreshold) {
        auto deathRate = params.array[Params::deathRate + batchIndex * params.shape[0]];
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
    Cell *boardCopy;
    NDimArray<short> landCoverGrid{};
    NDimArray<short> elevation{};
    NDimArray<bool> fire{};
    NDimArray<double> weather{};
    NDimArray<double> params{};
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
    Cell *board;

    Simulation(unsigned int w, unsigned int h, unsigned int batchIndex, int threads,
               const NDimArray<short> &landCoverGrid,
               const NDimArray<double> &landCoverRates,
               const NDimArray<short> &elevation,
               const NDimArray<bool> &fire,
               const NDimArray<double> &weather,
               const NDimArray<double> &params) {
        this->landCoverGrid = landCoverGrid;
        this->landCoverRates = landCoverRates;
        this->elevation = elevation;
        this->fire = fire;
        this->weather = weather;
        this->params = params;
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
        boardCopy = new Cell[size];

        initBoard();
        initCuda();
    }

    ~Simulation() {
        if (board != nullptr) {
            delete[] board;
            board = nullptr;
        }
        if (boardCopy != nullptr) {
            delete[] boardCopy;
            boardCopy = nullptr;
        }
    }

    [[nodiscard]] unsigned int gridDim() const {
        return size / nThreads + 1;
    }

    void tick(int tickIndex, bool print = false, bool cpu = false) {
        if (cpu) {
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                    cpuTick(board, boardCopy, landCoverRates, params, weather, batchIndex, x, y);
        } else {
//            setupKernel<<<gridDim(), nThreads>>>(d_randState, rand()); // NOLINT(cert-msc50-cpp)
            gpuTick<<<gridDim(), nThreads>>>(
                    d_randState, d_board, d_boardCopy,
                    d_landCoverRates, d_params, d_weather,
                    lcrShape, paramsShape, weatherShape,
                    batchIndex
            );
        }

        if (print) {
            // Copy data back to CPU
            if (cpu) {
                board = boardCopy;
            } else {
                cudaCheck(cudaMemcpy(board, d_boardCopy, size * sizeof(Cell), cudaMemcpyDeviceToHost))
            }
            printBoard();
        } else {
            cudaCheck(cudaMemcpy(board, d_boardCopy, size * sizeof(Cell), cudaMemcpyDeviceToHost))
        }
//        boardToImage(tickIndex);

        if (cpu) {
            std::swap(board, boardCopy);
        } else {
            std::swap(d_board, d_boardCopy);
        }
    }

    void initBoard() const {
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
        auto t = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
        setupKernel<<<gridDim(), nThreads>>>(d_randState, (int) t); // NOLINT(cert-msc50-cpp)

        auto weatherSize = nDimArrayLength(weather) * sizeof(double);
        auto psoConfigsSize = nDimArrayLength(params) * sizeof(double);
        auto landCoverRatesSize = nDimArrayLength(landCoverRates) * sizeof(double);

        // allocate gpu buffers for board and copy
        cudaCheck(cudaMalloc((void **) &d_board, size * sizeof(Cell)))
        cudaCheck(cudaMalloc((void **) &d_boardCopy, size * sizeof(Cell)))
        cudaCheck(cudaMalloc((void **) &d_weather, weatherSize))
        cudaCheck(cudaMalloc((void **) &d_params, psoConfigsSize))
        cudaCheck(cudaMalloc((void **) &d_landCoverRates, landCoverRatesSize))
        // copy board from CPU to GPU
        cudaCheck(cudaMemcpy(d_board, board, size * sizeof(Cell), cudaMemcpyHostToDevice))
        cudaCheck(cudaMemcpy(d_weather, weather.array, weatherSize, cudaMemcpyHostToDevice))
        cudaCheck(cudaMemcpy(d_params, params.array, psoConfigsSize, cudaMemcpyHostToDevice))
        cudaCheck(cudaMemcpy(d_landCoverRates, landCoverRates.array, landCoverRatesSize, cudaMemcpyHostToDevice))
    }

    void boardToImage(int tickIndex) const {
        auto *pix = static_cast<unsigned char *>(malloc(width * height * 3));

        for (int i = 0; i < width * height; i++) {
            int pixI = i * 3;
            pix[pixI] = static_cast<unsigned char>(board[i].fireActivity * 255);
            pix[pixI + 1] = static_cast<unsigned char>(board[i].fuel * 255);
            pix[pixI + 2] = 0;
        }

//        printf("Exporting board to image\n");
        charArrToImage(pix, width, height, "board" + std::to_string(tickIndex));
        free(pix);
    }

    void printBoard() const {
        for (int j = 0; j < width * height; j++) {
            auto cell = board[j];
            if (cell.fireActivity <= 0)
                printf("_ ");
            else if (cell.fireActivity <= .1)
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
        cudaCheck(cudaDeviceSynchronize())
        cudaCheck(cudaDeviceReset())
    }
};

void batchSimulate(NDimArray<short> landCoverGrid,
                   NDimArray<double> landCoverRates,
                   NDimArray<short> elevation,
                   NDimArray<bool> fire,
                   NDimArray<double> weather,
                   NDimArray<double> params,
                   double output[]) {
    auto width = landCoverGrid.shape[0];
    auto height = landCoverGrid.shape[1];
    auto batchSize = params.shape[1];
    auto timeSteps = weather.shape[2];

    for (int i = 0; i < batchSize; i++) {
//        printf("Iteration %i\n", i);
        auto sim = Simulation(width, height, i, 96,
                              landCoverGrid, landCoverRates, elevation, fire, weather, params);
//        sim.printBoard();
        for (int t = 0; t < timeSteps; t++) {
//            printf("Tick %i\n", t);
            sim.tick(t, false, false);
        }

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                output[i * height * width +
                       y * width +
                       x] = sim.board[y * width + x].fuel;
            }
        }

        Simulation::freeCuda();
    }
}

int main() {
    int size = 100;
    printf("SIZE = %d\n", size);
    int width = size;
    int height = size;
    int timeSteps = 72;
    int checkpoints = 10;
    int weatherElements = 2;
    int psoParams = 10;
    int batchSize = 1;
    int landCoverTypes = 47;

    auto landCoverGrid = createNDimArray<short>(2, new long[2]{width, height}, 1);
    auto landCoverRates = createNDimArray<double>(2, new long[2]{width, height}, 1);
    auto elevation = createNDimArray<short>(2, new long[2]{landCoverTypes, batchSize}, 3);
    auto fire = createNDimArray<bool>(3, new long[3]{width, height, checkpoints}, false);
    auto weather = createNDimArray<double>(4, new long[4]{width, height, timeSteps, weatherElements}, 0);
    auto params = createNDimArray<double>(2, new long[2]{psoParams, batchSize}, 1);

    fire.array[height / 2 * width + width / 2] = true;
    fire.array[(height / 2 - 1) * width + width / 2] = true;
    fire.array[height / 2 * width + width / 2 - 1] = true;
    fire.array[(height / 2 - 1) * width + width / 2 - 1] = true;
    //    fire.array[1] = true;
    //    fire.array[1 * width + 1] = true;
    //    fire.array[1 * width + 0] = true;

    for (int x = 0; x < weather.shape[0]; x++) {
        for (int y = 0; y < weather.shape[1]; y++) {
            for (int z = 0; z < weather.shape[2]; z++) {
                weather.array[1 * weather.shape[0] * weather.shape[1] * weather.shape[2] +
                              z * weather.shape[1] * weather.shape[0] +
                              y * weather.shape[0] +
                              x] = 0;
            }
        }
    }

    for (int b = 0; b < params.shape[1]; b++) {
        auto s = params.shape[0];
        params.array[b * s + Params::activityThreshold] = 0.2;
        params.array[b * s + Params::burnRate] = 0.1;
        params.array[b * s + Params::fireDeathThreshold] = 0.1;
        params.array[b * s + Params::deathRate] = 0.2;
        params.array[b * s + Params::areaEffectMultiplier] = 1;
        params.array[b * s + Params::heightEffectMultiplierDown] = 1;
        params.array[b * s + Params::heightEffectMultiplierUp] = 1;
        params.array[b * s + Params::spreadSpeed] = 1.5;
        params.array[b * s + Params::windEffectMultiplier] = 3;
    }

//    printNDimArray(weather, "Weather");
    auto *output = static_cast<double *>(malloc(batchSize * width * height * sizeof(double)));

    int simTicks = 50;
    int avgRepeats = 10;
    for (int threadCount = 12; threadCount < 512; threadCount += 12) {
        long sum = 0;
        for (int i = 0; i < avgRepeats; i++) {
            auto sim = Simulation(width, height, 0, threadCount, landCoverGrid, landCoverRates, elevation, fire,
                                  weather, params);
            auto start = std::chrono::high_resolution_clock::now();
            for (int j = 0; j < simTicks; j++) {
                sim.tick(0);
            }
            auto stop = std::chrono::high_resolution_clock::now();
            sum += std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
        }
        printf("%d %ld\n", threadCount, sum / avgRepeats);
    }

    free(output);
    if (landCoverGrid.array != nullptr) {
        free(landCoverGrid.array);
        landCoverGrid.array = nullptr;
    }
    if (landCoverRates.array != nullptr) {
        free(landCoverRates.array);
        landCoverRates.array = nullptr;
    }
    if (elevation.array != nullptr) {
        free(elevation.array);
        elevation.array = nullptr;
    }
    if (fire.array != nullptr) {
        free(fire.array);
        fire.array = nullptr;
    }
    if (weather.array != nullptr) {
        free(weather.array);
        weather.array = nullptr;
    }
    if (params.array != nullptr) {
        free(params.array);
        params.array = nullptr;
    }
    return 0;
}

np::ndarray wrapBatchSimulate(np::ndarray const &npLandCoverGrid,
                              np::ndarray const &npLandCoverRates,
                              np::ndarray const &npElevation,
                              np::ndarray const &npFire,
                              np::ndarray const &npWeather,
                              np::ndarray const &npParams) {

    // Make sure we get right types
    // lcg:         2D WxH short array,     each cell value is index for landCoverRates array
    // lcr:         2D LxN double array     L is land cover type count, N is batch size
    // elevation:   2D WxH short array
    // fire:        2D WxH bool array
    // weather:     4D WxHxTxE double       array T is time steps. E is elements size (wind X, wind Y)
    // params:      2D PxN double array     P is params count, N is batch size

    auto landCoverGrid = npToArray<short>(npLandCoverGrid);
    auto landCoverRates = npToArray<double>(npLandCoverRates);
    auto elevation = npToArray<short>(npElevation);
    auto fire = npToArray<bool>(npFire);
    auto weather = npToArray<double>(npWeather);
    auto params = npToArray<double>(npParams);

    auto batchSize = params.shape[1];
    auto width = weather.shape[0];
    auto height = weather.shape[1];
    auto *output = static_cast<double *>(malloc(batchSize * width * height * sizeof(double)));
    batchSimulate(landCoverGrid, landCoverRates, elevation, fire, weather, params, output);

    np::dtype dt = np::dtype::get_builtin<double>();
    p::tuple shape = p::make_tuple(width, height, batchSize);
    auto sd = sizeof(double);
    p::tuple stride = p::make_tuple(sd, sd * width, sd * height * width);
    np::ndarray result = np::from_data(output, dt, shape, stride, p::object());

    if (landCoverGrid.array != nullptr) {
        free(landCoverGrid.array);
        landCoverGrid.array = nullptr;
    }
    if (landCoverRates.array != nullptr) {
        free(landCoverRates.array);
        landCoverRates.array = nullptr;
    }
    if (elevation.array != nullptr) {
        free(elevation.array);
        elevation.array = nullptr;
    }
    if (fire.array != nullptr) {
        free(fire.array);
        fire.array = nullptr;
    }
    if (weather.array != nullptr) {
        free(weather.array);
        weather.array = nullptr;
    }
    if (params.array != nullptr) {
        free(params.array);
        params.array = nullptr;
    }

    return result;
}

BOOST_PYTHON_MODULE (cuda_python) {  // Thing in brackets should match output library name
    Py_Initialize();
    np::initialize();
    p::def("batch_simulate", wrapBatchSimulate);
    p::def("main", main);
}