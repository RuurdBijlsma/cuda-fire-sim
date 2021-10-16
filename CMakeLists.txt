cmake_minimum_required(VERSION 3.20)
project(cuda_sim CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(cuda_sim main.cu main.h)

set_target_properties(cuda_sim PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
