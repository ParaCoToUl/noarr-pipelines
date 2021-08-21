cmake_minimum_required(VERSION 3.10)

# set the project name
project(NoarrPipelinesUppercaseExample VERSION 0.1 LANGUAGES CXX CUDA)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# use threading library
find_package(Threads REQUIRED)

# setup the executable
add_executable(uppercase main.cpp)
target_include_directories(uppercase PUBLIC ../../include)
target_link_libraries(uppercase PRIVATE Threads::Threads)

# setup the CUDA executable
add_executable(uppercase-cuda main.cu)
target_include_directories(uppercase-cuda PUBLIC ../../include)
target_link_libraries(uppercase-cuda PRIVATE Threads::Threads)
