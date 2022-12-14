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


#include "../cd_correct_bc_1d_cuda.hpp"

#include "md_device_span.hpp"

template <typename T>
void cd_correct_bc_1d_cuda<T>::set_bc(const int& i0,
                                      const int& i1,
                                      const T* dx,
                                      const int& dgcw,
                                      const T* d0_,
                                      const int& ugcw,
                                      T* u_,
                                      const int* bLo,
                                      const int* bHi,
                                      const int& exOrder,
                                      const int& face,
                                      const int& btype,
                                      const T& alpha,
                                      const T& beta)
{
    const auto i = Ib{i0, i1};

    auto d0 = make_md_span(d0_, dgcw, i + 1);
    auto u = make_md_span(u_, ugcw, i);

    // boundary bounds
    auto ib = Ib{std::max(bLo[0], i0), std::min(bHi[0], i1)};
    // lo/hi bounds
    const auto il = Ib{bLo[0], bLo[0]}, ih = Ib{bHi[0], bHi[0]};

    auto t1 = [&](auto&& u1, auto&& u2, auto&& d, T h, auto&&... p) mutable {
        switch (btype) {
        case 0:
            if (exOrder == 1) {
                with_domain(FWD(p)...)(u = -1 * u1);
                u.copy_to(u_);
            } else if (exOrder == 2) {
                with_domain(FWD(p)...)(u = -2 * u1 + u2 / 3);
                u.copy_to(u_);
            }
            return;
        case 1:
            with_domain(FWD(p)...)(u = u1);
            u.copy_to(u_);

            return;
        case 4:
            if (exOrder == 1) {
                with_domain(FWD(p)...)(
                    u = ((2 * alpha * d - h * beta) / (2 * alpha * d + h * beta)) * u1);
                u.copy_to(u_);
            } else if (exOrder == 2) {
                with_domain(FWD(p)...)(u = ((8 * alpha * d - 6 * beta * h) /
                                            ((8 * alpha * d + 3 * beta * h))) *
                                               u1 +
                                           h / ((8 * alpha * d + 3 * beta * h)) * u2);
                u.copy_to(u_);
            }
            return;
        }
    };

    switch (face) {
    case 0:
        t1(u.shift_x(1), u.shift_x(2), d0.shift_x(1), dx[0], il);
        break;
    case 1:
        t1(u.shift_x(-1), u.shift_x(-2), d0, dx[0], ih);
        break;
    }
    return;
}

template <typename T>
void cd_correct_bc_1d_cuda<T>::set_poisson_bc(const int& i0,
                                              const int& i1,
                                              const T* dx,
                                              const int& ugcw,
                                              T* u_,
                                              const int* bLo,
                                              const int* bHi,
                                              const int& exOrder,
                                              const int& face,
                                              const int& btype,
                                              const T& alpha,
                                              const T& beta)
{
    const auto i = Ib{i0, i1};

    auto u = make_md_span(u_, ugcw, i);

    // boundary bounds
    auto ib = Ib{std::max(bLo[0], i0), std::min(bHi[0], i1)};
    // lo/hi bounds
    const auto il = Ib{bLo[0], bLo[0]}, ih = Ib{bHi[0], bHi[0]};

    auto t1 = [&](auto&& u1, auto&& u2, T h, auto&&... p) mutable {
        switch (btype) {
        case 0:
            if (exOrder == 1) {
                with_domain(FWD(p)...)(u = -1 * u1);
                u.copy_to(u_);
            } else if (exOrder == 2) {
                with_domain(FWD(p)...)(u = -2 * u1 + u2 / 3);
                u.copy_to(u_);
            }
            return;
        case 1:
            with_domain(FWD(p)...)(u = u1);
            u.copy_to(u_);

            return;
        case 4:
            if (exOrder == 1) {
                with_domain(FWD(p)...)(u = ((4 - h) / (4 + h)) * u1);
                u.copy_to(u_);
            } else if (exOrder == 2) {
                with_domain(FWD(p)...)(u = ((16 - 6 * h) / ((16 + 3 * h))) * u1 +
                                           h / ((16 + 3 * h)) * u2);
                u.copy_to(u_);
            }
            return;
        }
    };

    switch (face) {
    case 0:
        t1(u.shift_x(1), u.shift_x(2), dx[0], il);
        break;
    case 1:
        t1(u.shift_x(-1), u.shift_x(-2), dx[0], ih);
        break;
    }
    return;
}

template struct cd_correct_bc_1d_cuda<double>;
template struct cd_correct_bc_1d_cuda<float>;
