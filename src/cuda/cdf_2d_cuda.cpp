#include "../cd_flux_2d_cuda.hpp"

#include "md_device_vector.hpp"
#include "thrust/transform.h"

using B = bounds;

namespace
{
template <typename T>
struct flux_f {
    T dx;

    template <typename It>
    __host__ __device__ T operator()(const T& b, const stencil_t<It>& st)
    {
        auto&& [x0, x1] = st;
        return b * (x1 - x0) / dx;
    }
};

} // namespace

template <typename T>
void cdf_2d_cuda<T>::flux(const int& i0,
                          const int& j0,
                          const int& i1,
                          const int& j1,
                          const T* dx,
                          const T* b0_,
                          const T* b1_,
                          const int& gcw,
                          const T* u_,
                          T* f0_,
                          T* f1_)
{

    const auto I = B{i0, i1}, J = B{j0, j1};
    auto b0 = make_md_vec(b0_, J, I + 1);
    auto f0 = make_md_vec(f0_, J, I + 1);
    auto b1 = make_md_vec(b1_, I, J + 1);
    auto f1 = make_md_vec(f1_, I, J + 1);
    auto u = make_md_vec(u_, gcw, J, I);

    thrust::transform(
        b0.begin(), b0.end(), u(J, I.expand(1)).stencil(1), f0.begin(), flux_f<T>{dx[0]});

    thrust::transform(b1.begin(),
                      b1.end(),
                      u(J.expand(1), I).ij().stencil(1),
                      f1.begin(),
                      flux_f<T>{dx[1]});

    thrust::copy(f0.begin(), f0.end(), f0_);
    thrust::copy(f1.begin(), f1.end(), f1_);
}

namespace
{
template <typename T>
struct poisson_flux_f {
    T dx;

    template <typename It>
    __host__ __device__ T operator()(const stencil_t<It>& st)
    {
        auto&& [x0, x1] = st;
        return (x1 - x0) / dx;
    }
};

} // namespace

template <typename T>
void cdf_2d_cuda<T>::poisson_flux(const int& i0,
                                  const int& j0,
                                  const int& i1,
                                  const int& j1,
                                  const T* dx,
                                  const int& gcw,
                                  const T* u_,
                                  T* f0_,
                                  T* f1_)
{
    const auto I = B{i0, i1}, J = B{j0, j1};
    auto u = make_md_vec(u_, gcw, J, I);
    auto f0 = make_md_vec(f0_, J, I + 1);
    auto f1 = make_md_vec(f1_, I, J + 1);

    auto ux = u(J, I.expand(1)).stencil(1);
    thrust::transform(ux, ux + ux.size(), f0.begin(), poisson_flux_f<T>{dx[0]});

    auto uy = u(J.expand(1), I).ij().stencil(1);
    thrust::transform(uy, uy + uy.size(), f1.begin(), poisson_flux_f<T>{dx[1]});

    thrust::copy(f0.begin(), f0.end(), f0_);
    thrust::copy(f1.begin(), f1.end(), f1_);
}

template struct cdf_2d_cuda<double>;
template struct cdf_2d_cuda<float>;
