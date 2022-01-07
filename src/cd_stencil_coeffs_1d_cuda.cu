#include "cd_stencil_coeffs_1d_cuda.hpp"

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>

// For now these are wrappers around the device kernels that simply handle data transfer.
// The current assumption is that all data coming in is host data.

template <typename T>
using It = typename thrust::device_vector<T>::iterator;

template <typename T>
struct offdiag1d_f {
    T d0;
    It<T> stencil;

    offdiag1d_f(T d0, It<T> stencil) : d0{d0}, stencil{stencil} {}

    template <typename Tuple>
    __host__ __device__ void operator()(Tuple t)
    {
        int i = thrust::get<0>(t);

        stencil[3 * i + 1] = d0 * thrust::get<1>(t);
        stencil[3 * i + 2] = d0 * thrust::get<2>(t);
    }
};

template <typename T>
void cd_stencil_coeffs_1d_cuda<T>::offdiag1d(const int& ifirst0,
                                             const int& ilast0,
                                             const int& bilo0,
                                             const int& bihi0,
                                             const T* dx,
                                             const T& beta,
                                             const T* b0,
                                             const int& sgcw,
                                             T* stencil)
{
    thrust::device_vector<T> d_stencil(
        stencil, stencil + 3 * (1 + ilast0 + sgcw - (ifirst0 - sgcw)));
    thrust::device_vector<T> d_b0(b0, b0 + 1 + bihi0 + 1 - bilo0);
    const T d0 = -beta / (*dx * *dx);

    // form zip iterator to access b0 [ifirst0:ilast0] and b0[ifirst0+1:ilast0+1]
    auto b0_first = d_b0.begin() + ifirst0 - bilo0;
    int n = ilast0 - ifirst0 + 1;

    auto first = thrust::make_zip_iterator(
        thrust::make_tuple(thrust::counting_iterator(0), b0_first, b0_first + 1));

    thrust::for_each_n(first, n, offdiag1d_f<T>(d0, d_stencil.begin() + 3 * sgcw));

    // copy data back out
    thrust::copy(d_stencil.begin(), d_stencil.end(), stencil);
}

template struct cd_stencil_coeffs_1d_cuda<double>;
template struct cd_stencil_coeffs_1d_cuda<float>;
