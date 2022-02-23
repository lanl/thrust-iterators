#include "../cd_flux_3d_cuda.hpp"

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
void cdf_3d_cuda<T>::flux(const int& i0,
                          const int& j0,
                          const int& k0,
                          const int& i1,
                          const int& j1,
                          const int& k1,
                          const T* dx,
                          const T* b0_,
                          const T* b1_,
                          const T* b2_,
                          const int& gcw,
                          const T* u_,
                          T* f0_,
                          T* f1_,
                          T* f2_)
{

    const B I = B{i0, i1}, J = B{j0, j1}, K = {k0, k1};
    auto u = make_md_vec(u_, gcw, K, J, I);

    auto b0 = make_md_vec(b0_, K, J, I + 1);
    auto b1 = make_md_vec(b1_, I, K, J + 1);
    auto b2 = make_md_vec(b2_, J, I, K + 1);
    auto f0 = make_md_vec(f0_, K, J, I + 1);
    auto f1 = make_md_vec(f1_, I, K, J + 1);
    auto f2 = make_md_vec(f2_, J, I, K + 1);

    thrust::transform(b0.begin(),
                      b0.end(),
                      u(K, J, I.expand(1)).istencil(),
                      f0.begin(),
                      flux_f<T>{dx[0]});

    // b1/f1 is ikj
    thrust::transform(b1.begin(),
                      b1.end(),
                      u(K, J.expand(1), I).ikj().istencil(),
                      f1.begin(),
                      flux_f<T>{dx[1]});

    // b2/f2 is jik
    thrust::transform(b2.begin(),
                      b2.end(),
                      u(K.expand(1), J, I).jik().istencil(),
                      f2.begin(),
                      flux_f<T>{dx[2]});
    thrust::copy(f0.begin(), f0.end(), f0_);
    thrust::copy(f1.begin(), f1.end(), f1_);
    thrust::copy(f2.begin(), f2.end(), f2_);
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
void cdf_3d_cuda<T>::poisson_flux(const int& i0,
                                  const int& j0,
                                  const int& k0,
                                  const int& i1,
                                  const int& j1,
                                  const int& k1,
                                  const T* dx,
                                  const int& gcw,
                                  const T* u_,
                                  T* f0_,
                                  T* f1_,
                                  T* f2_)
{
    const B I = B{i0, i1}, J = B{j0, j1}, K = B{k0, k1};

    auto u = make_md_vec(u_, gcw, K, J, I);
    auto f0 = make_md_vec(f0_, K, J, I + 1);
    auto f1 = make_md_vec(f1_, I, K, J + 1);
    auto f2 = make_md_vec(f2_, J, I, K + 1);

    auto ux = u(K, J, I.expand(1)).istencil();
    thrust::transform(ux, ux + ux.size(), f0.begin(), poisson_flux_f<T>{dx[0]});

    // u is kji; f1 is ikj
    // stencil before transpose is in j, after transpose in i
    auto uy = u(K, J.expand(1), I).ikj().istencil();
    thrust::transform(uy, uy + uy.size(), f1.begin(), poisson_flux_f<T>{dx[1]});

    // u is kji; f2 is jik
    auto uz = u(K.expand(1), J, I).jik().istencil();
    thrust::transform(uz, uz + uz.size(), f2.begin(), poisson_flux_f<T>{dx[2]});

    thrust::copy(f0.begin(), f0.end(), f0_);
    thrust::copy(f1.begin(), f1.end(), f1_);
    thrust::copy(f2.begin(), f2.end(), f2_);
}

template struct cdf_3d_cuda<double>;
template struct cdf_3d_cuda<float>;
