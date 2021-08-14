cmake_minimum_required(VERSION 3.10)

# set the project name
project(NoarrPipelinesUpcaseExample VERSION 0.1)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# use threading library
find_package(Threads REQUIRED)

# setup the test runner executable
add_executable(upcase main.cpp)
target_include_directories(upcase PUBLIC ../../include)
target_link_libraries(upcase PRIVATE Threads::Threads)
