cmake_minimum_required(VERSION 3.5)

set(Boost_USE_STATIC_RUNTIME ON)
# This logic needs to be considered before project()
set(_change_MSVC_flags FALSE)
if (WIN32)
    if (CMAKE_VERSION VERSION_LESS 3.15.0)
        set(_change_MSVC_flags TRUE)
    else ()
        # Set MSVC runtime to MultiThreaded (/MT)
        cmake_policy(SET CMP0091 NEW)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif ()
endif ()

project(cuda_python LANGUAGES CXX CUDA)

if (_change_MSVC_flags)
    # Modify compile flags to change MSVC runtime from /MD to /MT
    set(_re_match "([\\/\\-]M)D")
    set(_re_replace "\\1T")
    string(REGEX REPLACE ${_re_match} ${_re_replace}
            CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REGEX REPLACE ${_re_match} ${_re_replace}
            CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    string(REGEX REPLACE ${_re_match} ${_re_replace}
            CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REGEX REPLACE ${_re_match} ${_re_replace}
            CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL}")
    string(REGEX REPLACE ${_re_match} ${_re_replace}
            CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
endif ()

set(CMAKE_CUDA_STANDARD 20)

# Find python and Boost - both are required dependencies
find_package(PythonLibs 3.9 REQUIRED)

find_package(Boost COMPONENTS python39 numpy39 REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})
set(_Boost_NUMPY_DEPENDENCIES python${component_python_version})

add_definitions(-DBOOST_PYTHON_STATIC_LIB)

# Without this, any build libraries automatically have names "lib{x}.so"
set(CMAKE_SHARED_MODULE_PREFIX "")
set(CMAKE_BUILD_TYPE Debug)

# Add a shared module - modules are intended to be imported at runtime.
# - This is where you add the source files
add_library(cuda_python SHARED test.cpp)
#add_executable(cuda_python cuda_python.cu)

set_target_properties(cuda_python PROPERTIES CUDA_ARCHITECTURES "72")
set_target_properties(cuda_python PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Set up the libraries and header search paths for this target
target_link_libraries(cuda_python ${PYTHON_LIBRARIES} ${Boost_LIBRARIES} Boost::numpy39)
target_include_directories(cuda_python PRIVATE ${PYTHON_INCLUDE_DIRS})

set_target_properties(cuda_python PROPERTIES DEFINE_SYMBOL "BOOST_ALL_NO_LIB")
#set_target_properties(cuda_python PROPERTIES SUFFIX ".pyd")