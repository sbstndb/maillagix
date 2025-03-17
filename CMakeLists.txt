cmake_minimum_required(VERSION 3.10)
project(maillagix LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=native -march=native -O3 -funroll-loops")

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(MPI)
find_package(HDF5 REQUIRED)
find_package(HighFive)
include_directories(${HDF5_INCLUDE_DIRS} ${HIGHFIVE_INCLUDE_DIR})

add_executable(maillagix src/main.cpp)

target_link_libraries(maillagix PRIVATE ${HDF5_CXX_LIBRARIES} OpenMP::OpenMP_CXX HighFive)

target_include_directories(maillagix PRIVATE include)

