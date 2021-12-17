//
// Created by ruurd on 17-12-21.
//
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

namespace p = boost::python;
namespace np = boost::python::numpy;

#ifndef CUDA_PYTHON_NDARRAYHELPERS_H
#define CUDA_PYTHON_NDARRAYHELPERS_H

#endif //CUDA_PYTHON_NDARRAYHELPERS_H


template<class T>
struct NDimArray {
    T *array;
    int nDims;
    long *shape;
};

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