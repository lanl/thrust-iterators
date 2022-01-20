#include "cd_stencil_coeffs_1d_cuda.hpp"

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include "md_device_vector.hpp"

// For now these are wrappers around the device kernels that simply handle data transfer.
// The current assumption is that all data coming in is host data.

template <typename T>
using It = typename thrust::device_vector<T>::iterator;

template <typename T>
struct offdiag_f {
    T d0;
    It<T> stencil;

    offdiag_f(T d0, It<T> stencil) : d0{d0}, stencil{stencil} {}

    template <typename Tuple>
    __host__ __device__ void operator()(Tuple t)
    {
        int i = thrust::get<0>(t);

        stencil[3 * i + 1] = d0 * thrust::get<1>(t);
        stencil[3 * i + 2] = d0 * thrust::get<2>(t);
    }
};

template <typename T>
void cd_stencil_coeffs_1d_cuda<T>::offdiag(const int& ifirst0,
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

    // what about something like md_device_vec<T>(stencil, bounds0, bounds1)
    // md_device_vec(stencil, bounds{ifirst0-sgcw,ilast0+sgcw}, ibounds(0,2))
    // md_device_vec(b0, bounds(bilo0,bihi0+1))
    auto v = make_md_vec(stencil, bounds(ifirst0 - sgcw, ilast0 + sgcw), bounds(0, 2));
    auto b = make_md_vec(b0, bounds(bilo0, bihi0 + 1));

    // form zip iterator to access b0 [ifirst0:ilast0] and b0[ifirst0+1:ilast0+1]
    auto b0_first = d_b0.begin() + ifirst0 - bilo0;
    int n = ilast0 - ifirst0 + 1;

    auto first = thrust::make_zip_iterator(
        thrust::make_tuple(thrust::counting_iterator(0), b0_first, b0_first + 1));

    thrust::for_each_n(first, n, offdiag_f<T>(d0, d_stencil.begin() + 3 * sgcw));

    // copy data back out
    thrust::copy(d_stencil.begin(), d_stencil.end(), stencil);
}

template <typename T>
struct poisson_offdiag_f {
    T d0;
    It<T> stencil;

    poisson_offdiag_f(T d0, It<T> stencil) : d0{d0}, stencil{stencil} {}

    // template <typename Tuple>
    __host__ __device__ void operator()(int i)
    {
        stencil[3 * i + 1] = d0;
        stencil[3 * i + 2] = d0;
    }
};

template <typename T>
void cd_stencil_coeffs_1d_cuda<T>::poisson_offdiag(const int& ifirst0,
                                                   const int& ilast0,
                                                   const T* dx,
                                                   const T& beta,
                                                   const int& sgcw,
                                                   T* stencil)
{
    thrust::device_vector<T> d_stencil(
        stencil, stencil + 3 * (1 + ilast0 + sgcw - (ifirst0 - sgcw)));
    const T d0 = -beta / (*dx * *dx);

    thrust::for_each_n(thrust::counting_iterator(0),
                       ilast0 - ifirst0 + 1,
                       poisson_offdiag_f<T>(d0, d_stencil.begin() + 3 * sgcw));

    // copy data back out
    thrust::copy(d_stencil.begin(), d_stencil.end(), stencil);
}

//
// v1diag
//

template <typename T>
struct v1diag_f {
    T alpha;
    It<T> stencil;

    v1diag_f(T alpha, It<T> stencil) : alpha{alpha}, stencil{stencil} {}

    template <typename Tuple>
    __host__ __device__ void operator()(Tuple t)
    {
        int i = thrust::get<0>(t);
        stencil[3 * i] =
            -(stencil[3 * i + 1] + stencil[3 * i + 2]) + alpha * thrust::get<1>(t);
    }
};

template <typename T>
void cd_stencil_coeffs_1d_cuda<T>::v1diag(const int& ifirst0,
                                          const int& ilast0,
                                          const int& ailo0,
                                          const int& aihi0,
                                          const T& alpha,
                                          const T* a,
                                          const int& sgcw,
                                          T* stencil)
{
    thrust::device_vector<T> d_stencil(
        stencil, stencil + 3 * (1 + ilast0 + sgcw - (ifirst0 - sgcw)));
    thrust::device_vector<T> d_a(a, a + 1 + aihi0 - ailo0);

    // form zip iterator to access b0 [ifirst0:ilast0] and b0[ifirst0+1:ilast0+1]
    auto a_first = d_a.begin() + ifirst0 - ailo0;
    int n = ilast0 - ifirst0 + 1;

    auto first = thrust::make_zip_iterator(
        thrust::make_tuple(thrust::counting_iterator(0), a_first));

    thrust::for_each_n(first, n, v1diag_f<T>(alpha, d_stencil.begin() + 3 * sgcw));

    // copy data back out
    thrust::copy(d_stencil.begin(), d_stencil.end(), stencil);
}

//
// v2diag
//
template <typename T>
struct v2diag_f {
    T alpha;
    It<T> stencil;

    v2diag_f(T alpha, It<T> stencil) : alpha{alpha}, stencil{stencil} {}

    __host__ __device__ void operator()(int i)
    {
        stencil[3 * i] = -(stencil[3 * i + 1] + stencil[3 * i + 2]) + alpha;
    }
};

template <typename T>
void cd_stencil_coeffs_1d_cuda<T>::v2diag(
    const int& ifirst0, const int& ilast0, const T& alpha, const int& sgcw, T* stencil)
{
    thrust::device_vector<T> d_stencil(
        stencil, stencil + 3 * (1 + ilast0 + sgcw - (ifirst0 - sgcw)));

    thrust::for_each_n(thrust::counting_iterator(0),
                       ilast0 - ifirst0 + 1,
                       v2diag_f<T>(alpha, d_stencil.begin() + 3 * sgcw));

    // copy data back out
    thrust::copy(d_stencil.begin(), d_stencil.end(), stencil);
}

//
// poisson_diag
//
template <typename T>
struct poisson_diag_f {
    It<T> stencil;

    poisson_diag_f(It<T> stencil) : stencil{stencil} {}

    __host__ __device__ void operator()(int i)
    {
        stencil[3 * i] = -(stencil[3 * i + 1] + stencil[3 * i + 2]);
    }
};

template <typename T>
void cd_stencil_coeffs_1d_cuda<T>::poisson_diag(const int& ifirst0,
                                                const int& ilast0,
                                                const int& sgcw,
                                                T* stencil)
{
    thrust::device_vector<T> d_stencil(
        stencil, stencil + 3 * (1 + ilast0 + sgcw - (ifirst0 - sgcw)));

    thrust::for_each_n(thrust::counting_iterator(0),
                       ilast0 - ifirst0 + 1,
                       poisson_diag_f<T>(d_stencil.begin() + 3 * sgcw));

    // copy data back out
    thrust::copy(d_stencil.begin(), d_stencil.end(), stencil);
}

template struct cd_stencil_coeffs_1d_cuda<double>;
template struct cd_stencil_coeffs_1d_cuda<float>;
