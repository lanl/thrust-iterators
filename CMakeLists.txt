cmake_minimum_required(VERSION 3.19)

project(kernels LANGUAGES C CXX Fortran)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(THRUST_CPP_DIALECT ${CMAKE_CXX_STANDARD})

include(CTest)

find_package(Catch2 3 REQUIRED)

find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust FROM_OPTIONS)

thrust_is_cuda_system_found(USE_CUDA)
if (USE_CUDA)
  enable_language(CUDA)
endif()

# helper function for defining tests
function(add_unit_test t label)
  if (BUILD_TESTING)
    add_executable(t-${t} ${t}.t.cpp)
    target_link_libraries(t-${t} Catch2::Catch2WithMain ${ARGN})
    add_test(NAME t-${t} COMMAND t-${t})
    set_tests_properties(t-${t} PROPERTIES LABELS "${label}")
  endif()
endfunction()

add_subdirectory(m4)
add_subdirectory(src)
