cmake_minimum_required(VERSION 3.10)
project(NoarrPipelinesCudaTests VERSION 0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

Include(FetchContent)

# CUDA-related code taken from:
# - https://developer.nvidia.com/blog/building-cuda-applications-cmake/
# - https://stackoverflow.com/questions/48527238/linking-of-cuda-library-in-cmake

# download testing library Catch2
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v2.13.1)
FetchContent_MakeAvailable(Catch2)

# use threading library
find_package(Threads REQUIRED)

# use cuda library
find_package(CUDA 9.0 REQUIRED)

# setup the test runner executable
file(GLOB_RECURSE tests_source_files CONFIGURE_DEPENDS
    "tests/*_test.hpp"
    "tests/*_test.cpp"
)
add_executable(test-runner main_test.cpp ${tests_source_files})
target_include_directories(test-runner PUBLIC ../../include)
target_link_libraries(test-runner PRIVATE Catch2::Catch2)
target_link_libraries(test-runner PRIVATE Threads::Threads)
target_link_libraries(test-runner PRIVATE ${CUDA_LIBRARIES})
set_target_properties(test-runner PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


if(MSVC)
  target_compile_options(test-runner PRIVATE /W4)
else()
  target_compile_options(test-runner PRIVATE -Wall -Wextra -pedantic)
endif()
