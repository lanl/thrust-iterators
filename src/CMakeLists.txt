file(GLOB f77_files *.f)
add_library(f77_kernels ${f77_files})

add_library(random random.cpp)

add_library(cuda_kernels cd_stencil_coeffs_1d_cuda.cu)
target_link_libraries(cuda_kernels Thrust)

add_library(cuda_it window_iterator_test.cu sliding_iterator_test.cu)
target_link_libraries(cuda_it Thrust)

add_unit_test(window_iterator "iterator" cuda_it)
add_unit_test(sliding_iterator "iterator" cuda_it)

add_unit_test(cd_stencil_coeffs_1d "coeffs" f77_kernels cuda_kernels random)
