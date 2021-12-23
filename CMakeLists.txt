cmake_minimum_required(VERSION 3.19)

project(kernels LANGUAGES C CXX Fortran)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CTest)

find_package(Catch2 3 REQUIRED)

add_subdirectory(m4)
add_subdirectory(src)
