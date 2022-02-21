//
// Created by ruurd on 17-12-21.
//
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <filesystem>

using std::filesystem::current_path;

namespace p = boost::python;
namespace np = boost::python::numpy;

#ifndef CUDA_PYTHON_NDARRAYHELPERS_H
#define CUDA_PYTHON_NDARRAYHELPERS_H

#endif //CUDA_PYTHON_NDARRAYHELPERS_H

void charArrToImage(unsigned char *charArr, unsigned int width, unsigned int height, const std::string &outputFile) {
    unsigned int length = width * height * 3;

    FILE *imageFile;

    imageFile = fopen((outputFile + ".ppm").c_str(), "wb");
    if (imageFile == nullptr) {
        perror("ERROR: Cannot open output file");
        exit(EXIT_FAILURE);
    }

    fprintf(imageFile, "P6\n"); // P6 filetype
    fprintf(imageFile, "%d %d\n", width, height); // dimensions
    fprintf(imageFile, "255\n"); // Max pixel

    fwrite(charArr, 1, length, imageFile);
    fclose(imageFile);
    auto command = std::string("convert ") + outputFile + ".ppm " + outputFile + ".png";
    system(command.c_str());
    system(("rm " + outputFile + ".ppm").c_str());
}

template<class T>
struct NDimArray {
    T *array;
    int nDims;
    long *shape;
};

struct NDimArrayShape {
    long s0 = 0;
    long s1 = 0;
    long s2 = 0;
    long s3 = 0;
};

template<typename T>
unsigned long nDimArrayLength(NDimArray<T> array) {
    unsigned long arraySize = 1;
    for (int i = 0; i < array.nDims; i++) {
        arraySize *= array.shape[i];
    }
    return arraySize;
}

template<typename T>
size_t sizeOfNDimArray(NDimArray<T> array) {
    auto size = sizeof(int) + sizeof(long) * array.nDims;
    return size + sizeof(T) * nDimArrayLength(array);
}

template<typename T>
void nDimArrayToImage(const std::string &imgName, NDimArray<T> array, float min = 0, float max = 1) {
    if (array.nDims < 2) {
        printf("Wrong ndims");
        return;
    }
    auto width = array.shape[0];
    auto height = array.shape[1];
    auto *pix = static_cast<unsigned char *>(malloc(width * height * 3));

    if (array.nDims == 2) {
        for (int x = 0; x < array.shape[0]; x++) {
            for (int y = 0; y < array.shape[1]; y++) {
                int i = array.shape[0] * y + x;
                int pixI = i * 3;
                auto value = static_cast<double>(array.array[i]);
                value = (value - min) / (max - min);
                auto charValue = static_cast<unsigned char>(value * 255);
                pix[pixI] = charValue;
                pix[pixI + 1] = charValue;
                pix[pixI + 2] = charValue;
            }
        }
    } else if (array.nDims == 3) {
        printf("Grabbing first 3 values in 3rd dim for ndarray to image");
        for (int x = 0; x < array.shape[0]; x++) {
            for (int y = 0; y < array.shape[1]; y++) {
                int i = array.shape[0] * y + x;
                int pixI = i * 3;
                unsigned char charValueR = 0, charValueG = 0, charValueB = 0;
                if (array.shape[2] >= 1) {
                    auto valueR = static_cast<double>(array.array[i + 0]);
                    valueR = (valueR - min) / (max - min);
                    charValueR = static_cast<unsigned char>(valueR * 255);
                }
                if (array.shape[2] >= 2) {
                    auto valueG = static_cast<double>(array.array[i + 1]);
                    valueG = (valueG - min) / (max - min);
                    charValueG = static_cast<unsigned char>(valueG * 255);
                }
                if (array.shape[2] >= 3) {
                    auto valueB = static_cast<double>(array.array[i + 2]);
                    valueB = (valueB - min) / (max - min);
                    charValueB = static_cast<unsigned char>(valueB * 255);
                }
                pix[pixI] = charValueR;
                pix[pixI + 1] = charValueG;
                pix[pixI + 2] = charValueB;
            }
        }
    } else {
        printf("ndim array has too many dims for image!");
    }

    printf(("Exporting ndarray to image " + imgName).c_str());
    charArrToImage(pix, width, height, imgName);
    free(pix);
}

unsigned long getIndex(long x) {
    return x;
}

unsigned long getIndex(long x, long y, long width) {
    return y * width
           + x;
}

unsigned long getIndex(long x, long y, long z, long width, long height) {
    return z * width * height
           + y * width
           + x;
}

unsigned long getIndex(long x, long y, long z, long q, long width, long height, long depth) {
    return q * width * height * depth
           + z * width * height
           + y * width
           + x;
}

unsigned long getOffset(int nDims, const long *strides, const long *indices, const size_t typeSize) {
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
    long *arrShape = const_cast<long *>(npArray.get_shape());
    auto shape = reinterpret_cast<long *>(malloc(4 * sizeof(long)));
    for (int i = 0; i < 4; i++)
        shape[i] = i >= nDims ? 1 : arrShape[i];

    long size = 1;
    for (int i = 0; i < nDims; i++)
        size *= arrShape[i];

    auto mallocSize = size * sizeof(T);
    T *result = static_cast<T *>(malloc(mallocSize));
    T *strideArray = reinterpret_cast<T *>(npArray.get_data());

    for (long x = 0; x < shape[0]; x++)
        for (long y = 0; y < shape[1]; y++)
            for (long z = 0; z < shape[2]; z++)
                for (long q = 0; q < shape[3]; q++) {
                    long indices[4] = {x, y, z, q};
                    auto offset = getOffset(nDims, strides, indices, sizeof(T));
                    auto index = getIndex(x, y, z, q, shape[0], shape[1], shape[2]);
                    result[index] = strideArray[offset];
                }

    return NDimArray<T>{
            result,
            nDims,
            arrShape,
    };
}


template<typename T>
void printNDimArray(NDimArray<T> array, const std::string &arrayName = "ARRAY") {
    printf("=============== %s ===============\n", arrayName.c_str());
    for (int i = 0; i < array.nDims; i++) {
        printf("%s.shape[%i] = %li\n", arrayName.c_str(), i, array.shape[i]);
    }
    if (array.nDims == 4) {
        for (int q = 0; q < array.shape[3]; q++) {
            printf("-------- DIM3: %i --------\n", q);
            for (int z = 0; z < array.shape[2]; z++) {
                printf("________ DIM2: %i ________\n", z);
                for (int y = 0; y < array.shape[1]; y++) {
                    for (int x = 0; x < array.shape[0]; x++) {
                        auto index = getIndex(x, y, z, q, array.shape[0], array.shape[1], array.shape[2]);
                        printf("%s  ", std::to_string(array.array[index]).c_str());
                    }
                    printf("\n");
                }
                printf("\n");
            }
            printf("\n");
        }
    }
    if (array.nDims == 3) {
        for (int z = 0; z < array.shape[2]; z++) {
            printf("======== DIM2: %i =========\n", z);
            for (int y = 0; y < array.shape[1]; y++) {
                for (int x = 0; x < array.shape[0]; x++) {
                    auto index = getIndex(x, y, z, array.shape[0], array.shape[1]);
                    printf("%s  ", std::to_string(array.array[index]).c_str());
                }
                printf("\n");
            }
            printf("\n");
        }
    }
    if (array.nDims == 2) {
        for (int y = 0; y < array.shape[1]; y++) {
            for (int x = 0; x < array.shape[0]; x++) {
                auto index = getIndex(x, y, array.shape[0]);
//                printf("%i  ", index);
                printf("%s  ", std::to_string(array.array[index]).c_str());
            }
            printf("\n");
        }
    }
    if (array.nDims == 1) {
        for (int x = 0; x < array.shape[0]; x++) {
            auto index = getIndex(x);
            printf("%s  ", std::to_string(array.array[index]).c_str());
        }
        printf("\n");
    }
}

template<typename T>
NDimArray<T> createNDimArray(long nDims, long *shape, T fillValue) {
    auto result = NDimArray<T>();
    result.nDims = nDims;
    result.shape = shape;
    auto length = nDimArrayLength(result);
    result.array = static_cast<T *>(malloc(length * sizeof(T)));
    for (long i = 0; i < length; i++)
        result.array[i] = fillValue;
    return result;
}