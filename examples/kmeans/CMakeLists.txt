cmake_minimum_required(VERSION 3.10)
Include(FetchContent)
Include(ExternalProject)

# set the project name
project(NoarrPipelinesKmeansExample VERSION 0.1)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# use threading library
find_package(Threads REQUIRED)

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
set(NoarrStructuresIncludeDir ${source_dir}/noarr/include)

# setup the test runner executable
add_executable(kmeans main.cpp)
target_include_directories(kmeans PUBLIC ./include)
target_include_directories(kmeans PUBLIC ../../include)
add_dependencies(kmeans noarr-structures)
target_include_directories(kmeans PUBLIC ${NoarrStructuresIncludeDir})
target_link_libraries(kmeans PRIVATE Threads::Threads)

if(MSVC)
  target_compile_options(kmeans PRIVATE /W4)
else()
  target_compile_options(kmeans PRIVATE -Wall -Wextra -pedantic)
endif()
