// Copyright (c) 2022. Triad National Security, LLC. All rights reserved.
// This program was produced under U.S. Government contract
// 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is
// operated by Triad National Security, LLC for the U.S. Department of
// Energy/National Nuclear Security Administration. All rights in the
// program are reserved by Triad National Security, LLC, and the
// U.S. Department of Energy/National Nuclear Security
// Administration. The Government is granted for itself and others acting
// on its behalf a nonexclusive, paid-up, irrevocable worldwide license
// in this material to reproduce, prepare derivative works, distribute
// copies to the public, perform publicly and display publicly, and to
// permit others to do so.


#include "../cd_apply_1d_cuda.hpp"

#include "md_device_span.hpp"

template <typename T>
void cd_apply_1d_cuda<T>::diffusion_v1_res(const int& i0,
                                           const int& i1,
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
                                           const int& rgcw,
                                           T* res_)
{
    const auto i = Ib{i0, i1};

    auto a = make_md_span(a_, agcw, i);
    auto u = make_md_span(u_, ugcw, i);
    auto f = make_md_span(f_, fgcw, i);
    auto res = make_md_span(res_, rgcw, i);
    auto f0 = make_md_span(f0_, i + 1);

    with_domain(i)(res = f + beta * (f0.grad_x(dx[0])) - alpha * a * u);
    res.copy_to(res_);
}

template <typename T>
void cd_apply_1d_cuda<T>::diffusion_v2_res(const int& i0,
                                           const int& i1,
                                           const T& alpha,
                                           const T& beta,
                                           const T* dx,
                                           const int& ugcw,
                                           const T* u_,
                                           const int& fgcw,
                                           const T* f_,
                                           const T* f0_,
                                           const int& rgcw,
                                           T* res_)
{
    const auto i = Ib{i0, i1};

    auto u = make_md_span(u_, ugcw, i);
    auto f = make_md_span(f_, fgcw, i);
    auto res = make_md_span(res_, rgcw, i);
    auto f0 = make_md_span(f0_, i + 1);

    with_domain(i)(res = f + beta * (f0.grad_x(dx[0])) - alpha * u);
    res.copy_to(res_);
}

template <typename T>
void cd_apply_1d_cuda<T>::poisson_v1_res(const int& i0,
                                         const int& i1,
                                         const T& beta,
                                         const T* dx,
                                         const int& fgcw,
                                         const T* f_,
                                         const T* f0_,
                                         const int& rgcw,
                                         T* res_)
{
    const auto i = Ib{i0, i1};

    auto f = make_md_span(f_, fgcw, i);
    auto res = make_md_span(res_, rgcw, i);
    auto f0 = make_md_span(f0_, i + 1);

    with_domain(i)(res = f + beta * (f0.grad_x(dx[0])));
    res.copy_to(res_);
}

template <typename T>
void cd_apply_1d_cuda<T>::diffusion_v1_apply(const int& i0,
                                             const int& i1,
                                             const T& alpha,
                                             const T& beta,
                                             const T* dx,
                                             const int& agcw,
                                             const T* a_,
                                             const int& ugcw,
                                             const T* u_,
                                             const T* f0_,
                                             const int& rgcw,
                                             T* res_)
{
    const auto i = Ib{i0, i1};

    auto a = make_md_span(a_, agcw, i);
    auto u = make_md_span(u_, ugcw, i);
    auto res = make_md_span(res_, rgcw, i);
    auto f0 = make_md_span(f0_, i + 1);

    with_domain(i)(res = -beta * (f0.grad_x(dx[0])) + alpha * a * u);
    res.copy_to(res_);
}

template <typename T>
void cd_apply_1d_cuda<T>::diffusion_v2_apply(const int& i0,
                                             const int& i1,
                                             const T& alpha,
                                             const T& beta,
                                             const T* dx,
                                             const int& ugcw,
                                             const T* u_,
                                             const T* f0_,
                                             const int& rgcw,
                                             T* res_)
{
    const auto i = Ib{i0, i1};

    auto u = make_md_span(u_, ugcw, i);
    auto res = make_md_span(res_, rgcw, i);
    auto f0 = make_md_span(f0_, i + 1);

    with_domain(i)(res = -beta * (f0.grad_x(dx[0])) + alpha * u);
    res.copy_to(res_);
}

template <typename T>
void cd_apply_1d_cuda<T>::poisson_v2_apply(const int& i0,
                                           const int& i1,
                                           const T& beta,
                                           const T* dx,
                                           const T* f0_,
                                           const int& rgcw,
                                           T* res_)
{
    const auto i = Ib{i0, i1};

    auto res = make_md_span(res_, rgcw,  i);
    auto f0 = make_md_span(f0_, i + 1);

    with_domain(i)(res = -beta * (f0.grad_x(dx[0])));
    res.copy_to(res_);
}

template struct cd_apply_1d_cuda<double>;
template struct cd_apply_1d_cuda<float>;
