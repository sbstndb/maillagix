cmake_minimum_required(VERSION 3.10)
project(maillagix LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=native -march=native -O3 -funroll-loops")

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(CLI11 CONFIG REQUIRED)


#find_package(HDF5 REQUIRED)
#find_package(HighFive)
#include_directories(${HDF5_INCLUDE_DIRS} ${HIGHFIVE_INCLUDE_DIR})


option(USE_MPI_RDMA "Enable MPI RDMA support" OFF)
if(USE_MPI_RDMA)
    add_compile_definitions(USE_MPI_RDMA)
endif()

add_executable(maillagix src/main.cpp)
target_link_libraries(maillagix PRIVATE ${HDF5_CXX_LIBRARIES} OpenMP::OpenMP_CXX MPI::MPI_CXX CLI11::CLI11)# HighFive)
include_directories(${CLI11_DIR}/include)
#target_include_directories(maillagix PRIVATE include)

