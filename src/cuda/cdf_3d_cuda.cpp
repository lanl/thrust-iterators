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


#include "../cd_flux_3d_cuda.hpp"

#include "md_device_span.hpp"

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
    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};
    const auto k = Kb{k0, k1};

    auto u = make_md_span(u_, gcw, k, j, i);
    auto f0 = make_md_span(f0_, k, j, i + 1);
    auto b0 = make_md_span(b0_, k, j, i + 1);
    auto f1 = make_md_span(f1_, i, k, j + 1);
    auto b1 = make_md_span(b1_, i, k, j + 1);
    auto f2 = make_md_span(f2_, j, i, k + 1);
    auto b2 = make_md_span(b2_, j, i, k + 1);

    with_lhs_domain(f0 = b0 * u.grad_x(dx[0], down));
    with_lhs_domain(f1 = b1 * u.grad_y(dx[1], down));
    with_lhs_domain(f2 = b2 * u.grad_z(dx[2], down));

    f0.copy_to(f0_);
    f1.copy_to(f1_);
    f2.copy_to(f2_);
}

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
    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};
    const auto k = Kb{k0, k1};

    auto u = make_md_span(u_, gcw, k, j, i);
    auto f0 = make_md_span(f0_, k, j, i + 1);
    auto f1 = make_md_span(f1_, i, k, j + 1);
    auto f2 = make_md_span(f2_, j, i, k + 1);

    with_lhs_domain(f0 = u.grad_x(dx[0], down));
    with_lhs_domain(f1 = u.grad_y(dx[1], down));
    with_lhs_domain(f2 = u.grad_z(dx[2], down));

    f0.copy_to(f0_);
    f1.copy_to(f1_);
    f2.copy_to(f2_);
}

template struct cdf_3d_cuda<double>;
template struct cdf_3d_cuda<float>;
