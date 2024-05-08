cmake_minimum_required (VERSION 3.8)
cmake_policy(SET CMP0048 NEW)
project(mapf VERSION 0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Boost
find_package(Boost REQUIRED COMPONENTS program_options system filesystem)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RELEASE") # other possible values: "DEBUG", "RELWITHDEBINFO", "MINSIZEREL"
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG)

include_directories("${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/include/PIBT")

file(GLOB SOURCES "include/*.h" "src/*.cpp" "include/PIBT/*.h" "src/PIBT/*.cpp")
add_library(mapf_lib SHARED ${SOURCES})

add_subdirectory("${CMAKE_SOURCE_DIR}/include/pybind11")

pybind11_add_module(mapf MODULE python_api/MAPFApi.cpp)
target_link_libraries(mapf PRIVATE mapf_lib)
