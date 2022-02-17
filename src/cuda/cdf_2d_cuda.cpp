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
                          T* f0,
                          T* f1)
{

    auto b0 = make_md_vec(b0_, B{j0, j1}, B{i0, i1 + 1});
    auto b1 = make_md_vec(b1_, B{i0, i1}, B{j0, j1 + 1});
    auto u = make_md_vec(u_, B{j0 - gcw, j1 + gcw}, B{i0 - gcw, i1 + gcw});

    thrust::transform(b0.begin(),
                      b0.end(),
                      u(B{j0, j1}, B{i0 - 1, i1 + 1}).stencil(1),
                      f0,
                      flux_f<T>{dx[0]});

    thrust::transform(b1.begin(),
                      b1.end(),
                      u(B{j0 - 1, j1 + 1}, B{i0, i1}).ij().stencil(1),
                      f1,
                      flux_f<T>{dx[1]});
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
                                  T* f0,
                                  T* f1)
{
    auto u = make_md_vec(u_, B{j0 - gcw, j1 + gcw}, B{i0 - gcw, i1 + gcw});

    auto ux = u(B{j0, j1}, B{i0 - 1, i1 + 1}).stencil(1);
    thrust::transform(ux, ux + ux.size(), f0, poisson_flux_f<T>{dx[0]});

    auto uy = u(B{j0 - 1, j1 + 1}, B{i0, i1}).ij().stencil(1);
    thrust::transform(uy, uy + uy.size(), f1, poisson_flux_f<T>{dx[1]});
}

template struct cdf_2d_cuda<double>;
template struct cdf_2d_cuda<float>;
