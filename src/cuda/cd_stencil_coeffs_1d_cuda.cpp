#include "../cd_stencil_coeffs_1d_cuda.hpp"

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/zip_function.h>

#include "md_device_vector.hpp"
#include <cassert>

// For now these are wrappers around the device kernels that simply handle data transfer.
// The current assumption is that all data coming in is host data.
template <typename T>
using It = typename thrust::device_vector<T>::iterator;

template <typename T>
struct offdiag_f {
    T d0;

    template <typename St, typename B>
    __host__ __device__ void operator()(St st, B b)
    {
        st[1] = d0 * b[0];
        st[2] = d0 * b[1];
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
    auto st = make_md_vec(stencil, bounds(ifirst0 - sgcw, ilast0 + sgcw), bounds(0, 2));
    auto b = make_md_vec(b0, bounds(bilo0, bihi0 + 1));
    const T d0 = -beta / (*dx * *dx);
    //
    auto&& [col_first, col_last] = st.column(std::array{ifirst0}, std::array{ilast0});
    auto&& [b_first, b_last] = b.sliding(2, std::array{ifirst0}, std::array{ilast0});

    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(col_first, b_first)),
                     thrust::make_zip_iterator(thrust::make_tuple(col_last, b_last)),
                     thrust::make_zip_function(offdiag_f<T>{d0}));

    // copy data back out
    thrust::copy(st.begin(), st.end(), stencil);
}

template <typename T>
struct poisson_offdiag_f {
    T d0;

    template <typename St>
    __host__ __device__ void operator()(St st)
    {
        st[1] = d0;
        st[2] = d0;
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
    auto st = make_md_vec(stencil, bounds(ifirst0 - sgcw, ilast0 + sgcw), bounds(0, 2));
    const T d0 = -beta / (*dx * *dx);

    auto&& [first, last] = st.column(std::array{ifirst0}, std::array{ilast0});
    thrust::for_each(first, last, poisson_offdiag_f<T>{d0});

    // copy data back out
    thrust::copy(st.begin(), st.end(), stencil);
}

//
// v1diag
//

template <typename T>
struct v1diag_f {
    T alpha;

    template <typename St>
    __host__ __device__ void operator()(St st, const T& a)
    {
        st[0] = -(st[1] + st[2]) + alpha * a;
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
    auto st = make_md_vec(stencil, bounds(ifirst0 - sgcw, ilast0 + sgcw), bounds(0, 2));
    auto c = make_md_vec(a, bounds(ailo0, aihi0));

    auto [col_first, col_last] = st.column(std::array{ifirst0}, std::array{ilast0});
    auto [a_first, a_last] = c.offset(std::array{ifirst0}, std::array{ilast0});

    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(col_first, a_first)),
                     thrust::make_zip_iterator(thrust::make_tuple(col_last, a_last)),
                     thrust::make_zip_function(v1diag_f<T>{alpha}));

    // copy data back out
    thrust::copy(st.begin(), st.end(), stencil);
}

//
// v2diag
//
template <typename T>
struct v2diag_f {
    T alpha;

    template <typename St>
    __host__ __device__ void operator()(St st)
    {
        st[0] = -(st[1] + st[2]) + alpha;
    }
};

template <typename T>
void cd_stencil_coeffs_1d_cuda<T>::v2diag(
    const int& ifirst0, const int& ilast0, const T& alpha, const int& sgcw, T* stencil)
{
    auto st = make_md_vec(stencil, bounds(ifirst0 - sgcw, ilast0 + sgcw), bounds(0, 2));
    auto [first, last] = st.column(std::array{ifirst0}, std::array{ilast0});

    thrust::for_each(first, last, v2diag_f<T>{alpha});

    // copy data back out
    thrust::copy(st.begin(), st.end(), stencil);
}

//
// poisson_diag
//
template <typename T>
struct poisson_diag_f {
    template <typename St>
    __host__ __device__ void operator()(St st)
    {
        st[0] = -(st[1] + st[2]);
    }
};

template <typename T>
void cd_stencil_coeffs_1d_cuda<T>::poisson_diag(const int& ifirst0,
                                                const int& ilast0,
                                                const int& sgcw,
                                                T* stencil)
{
    auto st = make_md_vec(stencil, bounds(ifirst0 - sgcw, ilast0 + sgcw), bounds(0, 2));
    auto [first, last] = st.column(std::array{ifirst0}, std::array{ilast0});
    thrust::for_each(first, last, poisson_diag_f<T>{});

    // copy data back out
    thrust::copy(st.begin(), st.end(), stencil);
}

template struct cd_stencil_coeffs_1d_cuda<double>;
template struct cd_stencil_coeffs_1d_cuda<float>;
