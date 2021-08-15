cmake_minimum_required(VERSION 3.10)
Include(FetchContent)
Include(ExternalProject)

# set the project name
project(NoarrPipelinesKmeansExample VERSION 0.1 LANGUAGES CXX CUDA)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# use threading library
find_package(Threads REQUIRED)

# use cuda library
find_package(CUDA 9.0 REQUIRED)

# use noarr structures
ExternalProject_Add(
    noarr-structures
    PREFIX ${CMAKE_BINARY_DIR}/noarr-structures
    GIT_REPOSITORY https://github.com/ParaCoToUl/noarr-structures
    GIT_TAG origin/main
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    LOG_DOWNLOAD ON
    )
ExternalProject_Get_Property(noarr-structures source_dir)
set(NoarrStructuresIncludeDir ${source_dir}/include)

# setup the test runner executable
add_executable(kmeans main.cpp main.cu)
target_include_directories(kmeans PUBLIC ./include)
target_include_directories(kmeans PUBLIC ../../include)
add_dependencies(kmeans noarr-structures)
target_include_directories(kmeans PUBLIC ${NoarrStructuresIncludeDir})
target_link_libraries(kmeans PRIVATE Threads::Threads)
target_link_libraries(kmeans PRIVATE ${CUDA_LIBRARIES})

# https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#using-separate-compilation-in-cuda
set_target_properties(kmeans PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# cuda architecture to compile against
set_target_properties(kmeans PROPERTIES CUDA_ARCHITECTURES "35;50;72")

# make nvcc use C++ 14
set_target_properties(kmeans PROPERTIES CUDA_STANDARD 14)

# allow constexpr functions in kernels for noarr structures to work
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
