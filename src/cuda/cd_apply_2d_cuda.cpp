#include "../cd_apply_2d_cuda.hpp"

#include "md_device_vector.hpp"
#include "thrust/copy.h"
#include "thrust/transform.h"
#include <thrust/iterator/zip_iterator.h>
#include <thrust/zip_function.h>

using B = bounds;

template <typename T>
struct diffusion_v1_res_f {
    T alpha, beta, dx, dy;

    template <typename Tp>
    __host__ __device__ T operator()(const T& f, Tp&& tp)
    {
        auto&& [x0, x1] = thrust::get<0>(tp);
        auto&& [y0, y1] = thrust::get<1>(tp);
        T u = thrust::get<2>(tp);
        T a = thrust::get<3>(tp);

        return f + beta * ((x1 - x0) / dx + (y1 - y0) / dy) - alpha * a * u;
    }
};

template <typename T>
void cd_apply_2d_cuda<T>::diffusion_v1_res(const int& i0,
                                           const int& i1,
                                           const int& j0,
                                           const int& j1,
                                           const T& alpha,
                                           const T& beta,
                                           const T* dx,
                                           const int& agcw,
                                           const T* a_,
                                           const int& ugcw,
                                           const T* u_,
                                           const int& fgcw,
                                           const T* f_,
                                           const T* f0_,
                                           const T* f1_,
                                           const int& rgcw,
                                           T* res_)
{
    const auto I = B{i0, i1}, J = B{j0, j1};

    auto a = make_md_vec(a_, agcw, J, I);
    auto u = make_md_vec(u_, ugcw, J, I);
    auto f = make_md_vec(f_, fgcw, J, I);
    auto res = make_md_vec(res_, rgcw, J, I);
    auto f0 = make_md_vec(f0_, J, I + 1);
    auto f1 = make_md_vec(f1_, I, J + 1);

    auto f_mat = f(J, I);

    thrust::transform(f_mat,
                      f_mat + f_mat.size(),
                      thrust::make_zip_iterator(thrust::make_tuple(
                          f0.istencil(), f1.ij().jstencil(), u(J, I), a(J, I))),
                      res(J, I),
                      diffusion_v1_res_f<T>{alpha, beta, dx[0], dx[1]});

    thrust::copy(res.begin(), res.end(), res_);
}

template <typename T>
struct diffusion_v2_res_f {
    T alpha, beta, dx, dy;

    template <typename Tp>
    __host__ __device__ T operator()(const T& f, Tp&& tp)
    {
        auto&& [x0, x1] = thrust::get<0>(tp);
        auto&& [y0, y1] = thrust::get<1>(tp);
        T u = thrust::get<2>(tp);

        return f + beta * ((x1 - x0) / dx + (y1 - y0) / dy) - alpha * u;
    }
};

template <typename T>
void cd_apply_2d_cuda<T>::diffusion_v2_res(const int& i0,
                                           const int& i1,
                                           const int& j0,
                                           const int& j1,
                                           const T& alpha,
                                           const T& beta,
                                           const T* dx,
                                           const int& ugcw,
                                           const T* u_,
                                           const int& fgcw,
                                           const T* f_,
                                           const T* f0_,
                                           const T* f1_,
                                           const int& rgcw,
                                           T* res_)
{
    const auto I = B{i0, i1}, J = B{j0, j1};

    auto u = make_md_vec(u_, ugcw, J, I);
    auto f = make_md_vec(f_, fgcw, J, I);
    auto res = make_md_vec(res_, rgcw, J, I);
    auto f0 = make_md_vec(f0_, J, I + 1);
    auto f1 = make_md_vec(f1_, I, J + 1);

    auto f_mat = f(J, I);

    thrust::transform(f_mat,
                      f_mat + f_mat.size(),
                      thrust::make_zip_iterator(
                          thrust::make_tuple(f0.istencil(), f1.ij().jstencil(), u(J, I))),
                      res(J, I),
                      diffusion_v2_res_f<T>{alpha, beta, dx[0], dx[1]});

    thrust::copy(res.begin(), res.end(), res_);
}

template <typename T>
struct poisson_v1_res_f {
    T beta, dx, dy;

    template <typename Tp>
    __host__ __device__ T operator()(const T& f, Tp&& tp)
    {
        auto&& [x0, x1] = thrust::get<0>(tp);
        auto&& [y0, y1] = thrust::get<1>(tp);

        return f + beta * ((x1 - x0) / dx + (y1 - y0) / dy);
    }
};

template <typename T>
void cd_apply_2d_cuda<T>::poisson_v1_res(const int& i0,
                                         const int& i1,
                                         const int& j0,
                                         const int& j1,
                                         const T& beta,
                                         const T* dx,
                                         const int& fgcw,
                                         const T* f_,
                                         const T* f0_,
                                         const T* f1_,
                                         const int& rgcw,
                                         T* res_)
{
    const auto I = B{i0, i1}, J = B{j0, j1};

    auto f = make_md_vec(f_, fgcw, J, I);
    auto res = make_md_vec(res_, rgcw, J, I);
    auto f0 = make_md_vec(f0_, J, I + 1);
    auto f1 = make_md_vec(f1_, I, J + 1);

    auto f_mat = f(J, I);

    thrust::transform(
        f_mat,
        f_mat + f_mat.size(),
        thrust::make_zip_iterator(thrust::make_tuple(f0.istencil(), f1.ij().jstencil())),
        res(J, I),
        poisson_v1_res_f<T>{beta, dx[0], dx[1]});

    thrust::copy(res.begin(), res.end(), res_);
}

template <typename T>
struct diffusion_v1_apply_f {
    T alpha, beta, dx, dy;

    template <typename Tp>
    __host__ __device__ T operator()(const T& u, Tp&& tp)
    {
        auto&& [x0, x1] = thrust::get<0>(tp);
        auto&& [y0, y1] = thrust::get<1>(tp);
        T a = thrust::get<2>(tp);

        return -beta * ((x1 - x0) / dx + (y1 - y0) / dy) + alpha * a * u;
    }
};

template <typename T>
void cd_apply_2d_cuda<T>::diffusion_v1_apply(const int& i0,
                                             const int& i1,
                                             const int& j0,
                                             const int& j1,
                                             const T& alpha,
                                             const T& beta,
                                             const T* dx,
                                             const int& agcw,
                                             const T* a_,
                                             const int& ugcw,
                                             const T* u_,
                                             const T* f0_,
                                             const T* f1_,
                                             const int& rgcw,
                                             T* res_)
{
    const auto I = B{i0, i1}, J = B{j0, j1};

    auto a = make_md_vec(a_, agcw, J, I);
    auto u = make_md_vec(u_, ugcw, J, I);
    auto res = make_md_vec(res_, rgcw, J, I);
    auto f0 = make_md_vec(f0_, J, I + 1);
    auto f1 = make_md_vec(f1_, I, J + 1);

    auto u_mat = u(J, I);

    thrust::transform(u_mat,
                      u_mat + u_mat.size(),
                      thrust::make_zip_iterator(
                          thrust::make_tuple(f0.istencil(), f1.ij().jstencil(), a(J, I))),
                      res(J, I),
                      diffusion_v1_apply_f<T>{alpha, beta, dx[0], dx[1]});

    thrust::copy(res.begin(), res.end(), res_);
}

template <typename T>
struct diffusion_v2_apply_f {
    T alpha, beta, dx, dy;

    template <typename Tp>
    __host__ __device__ T operator()(const T& u, Tp&& tp)
    {
        auto&& [x0, x1] = thrust::get<0>(tp);
        auto&& [y0, y1] = thrust::get<1>(tp);

        return -beta * ((x1 - x0) / dx + (y1 - y0) / dy) + alpha * u;
    }
};

template <typename T>
void cd_apply_2d_cuda<T>::diffusion_v2_apply(const int& i0,
                                             const int& i1,
                                             const int& j0,
                                             const int& j1,
                                             const T& alpha,
                                             const T& beta,
                                             const T* dx,
                                             const int& ugcw,
                                             const T* u_,
                                             const T* f0_,
                                             const T* f1_,
                                             const int& rgcw,
                                             T* res_)
{
    const auto I = B{i0, i1}, J = B{j0, j1};

    auto u = make_md_vec(u_, ugcw, J, I);
    auto res = make_md_vec(res_, rgcw, J, I);
    auto f0 = make_md_vec(f0_, J, I + 1);
    auto f1 = make_md_vec(f1_, I, J + 1);

    auto u_mat = u(J, I);

    thrust::transform(
        u_mat,
        u_mat + u_mat.size(),
        thrust::make_zip_iterator(thrust::make_tuple(f0.istencil(), f1.ij().jstencil())),
        res(J, I),
        diffusion_v2_apply_f<T>{alpha, beta, dx[0], dx[1]});

    thrust::copy(res.begin(), res.end(), res_);
}

template <typename T>
struct poisson_v2_apply_f {
    T beta, dx, dy;

    template <typename It0, typename It1>
    __host__ __device__ T operator()(const stencil_t<It0>& f0, const stencil_t<It1>& f1)
    {
        auto&& [x0, x1] = f0;
        auto&& [y0, y1] = f1;
        return -beta * ((x1 - x0) / dx + (y1 - y0) / dy);
    }
};

template <typename T>
void cd_apply_2d_cuda<T>::poisson_v2_apply(const int& i0,
                                           const int& i1,
                                           const int& j0,
                                           const int& j1,
                                           const T& beta,
                                           const T* dx,
                                           const T* f0_,
                                           const T* f1_,
                                           const int& rgcw,
                                           T* res_)
{
    const auto I = B{i0, i1}, J = B{j0, j1};

    auto res = make_md_vec(res_, rgcw, J, I);
    auto f0 = make_md_vec(f0_, J, I + 1);
    auto f1 = make_md_vec(f1_, I, J + 1);

    auto st = f0.istencil();

    thrust::transform(st,
                      st + st.size(),
                      f1.ij().jstencil(),
                      res(J, I),
                      poisson_v2_apply_f<T>{beta, dx[0], dx[1]});

    thrust::copy(res.begin(), res.end(), res_);
}

template struct cd_apply_2d_cuda<double>;
template struct cd_apply_2d_cuda<float>;
