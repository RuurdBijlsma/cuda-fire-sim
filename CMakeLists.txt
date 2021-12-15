cmake_minimum_required(VERSION 3.5)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
#set(BOOST_ROOT=/home/ruurd/Documents/SDK/boost178)
SET(Boost_DEBUG ON)
#SET(BOOST_INCLUDEDIR "/usr/include/boost")
#SET(BOOST_LIBRARYDIR "/usr/lib")

set(Boost_USE_STATIC_RUNTIME=ON)
# This logic needs to be considered before project()

project(cuda_python LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)

# Find python and Boost - both are required dependencies
set(PYTHON_INCLUDE_DIR /home/ruurd/anaconda3/include/python3.9)
set(PYTHON_LIBRARY /home/ruurd/anaconda3/lib/libpython3.9.so)
find_package(PythonLibs 3.9 REQUIRED)

find_package(Boost COMPONENTS python39 numpy39 REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})
set(_Boost_NUMPY_DEPENDENCIES python${component_python_version})

#add_definitions(-DBOOST_PYTHON_STATIC_LIB)

# Without this, any build libraries automatically have names "lib{x}.so"
set(CMAKE_SHARED_LIBRARY_PREFIX "")
#set(CMAKE_BUILD_TYPE Debug)

# Add a shared module - modules are intended to be imported at runtime.
# - This is where you add the source files
#add_library(cuda_python SHARED cuda_python.cu)
#add_executable(cuda_python cuda_python.cu)
add_executable(cuda_python example.cu)

set_target_properties(cuda_python PROPERTIES CUDA_ARCHITECTURES "72")
set_target_properties(cuda_python PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Set up the libraries and header search paths for this target
target_link_libraries(cuda_python ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})
target_include_directories(cuda_python PRIVATE ${PYTHON_INCLUDE_DIRS})

set_target_properties(cuda_python PROPERTIES DEFINE_SYMBOL "BOOST_ALL_NO_LIB")
#set_target_properties(cuda_python PROPERTIES SUFFIX ".pyd")