file(GLOB f77_files *.f)
add_library(f77_kernels ${f77_files})

add_library(random random.cpp)

if (USE_CUDA)
  file(GLOB thrust_kernels cuda/*.cpp)
  set_source_files_properties(
    ${thrust_kernels}
    PROPERTIES
    LANGUAGE CUDA)
endif()

add_library(cuda_kernels
  cuda/cd_stencil_coeffs_1d_cuda.cpp
  cuda/cdf_1d_cuda.cpp
  cuda/cd_apply_1d_cuda.cpp
  cuda/cdf_2d_cuda.cpp
  cuda/cd_apply_2d_cuda.cpp
  cuda/cdf_3d_cuda.cpp
  cuda/cd_apply_3d_cuda.cpp)
target_link_libraries(cuda_kernels Thrust Boost::boost)

add_library(cuda_it
  cuda/window_iterator_test.cpp
  cuda/sliding_iterator_test.cpp
  cuda/submatrix_iterator_test.cpp
  cuda/transpose_iterator_test.cpp
  cuda/forward_stencil_iterator_test.cpp
  cuda/lazy_vec.cpp)

target_link_libraries(cuda_it Thrust Boost::boost)

add_unit_test(window_iterator "iterator" cuda_it)
add_unit_test(sliding_iterator "iterator" cuda_it)
add_unit_test(submatrix_iterator "iterator" cuda_it)
add_unit_test(transpose_iterator "iterator" cuda_it random)
add_unit_test(forward_stencil_iterator "iterator" cuda_it random)
add_unit_test(lazy_vec "iterator" cuda_it random)

add_unit_test(cd_stencil_coeffs_1d "coeffs" f77_kernels cuda_kernels random)
add_unit_test(cd_flux_1d "flux" f77_kernels cuda_kernels random)
add_unit_test(cd_apply_1d "apply" f77_kernels cuda_kernels random)
add_unit_test(cd_flux_2d "flux" f77_kernels cuda_kernels random)
add_unit_test(cd_apply_2d "apply" f77_kernels cuda_kernels random)
add_unit_test(cd_flux_3d "flux" f77_kernels cuda_kernels random)
add_unit_test(cd_apply_3d "apply" f77_kernels cuda_kernels random)
