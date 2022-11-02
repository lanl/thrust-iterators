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


#include "../cd_correct_bc_2d_cuda.hpp"

#include "md_device_span.hpp"

template <typename T>
void cd_correct_bc_2d_cuda<T>::set_bc(const int& i0,
                                      const int& i1,
                                      const int& j0,
                                      const int& j1,
                                      const T* dx,
                                      const int& dgcw,
                                      const T* d0_,
                                      const T* d1_,
                                      const int& ugcw,
                                      T* u_,
                                      const int* bLo,
                                      const int* bHi,
                                      const int& exOrder,
                                      const int& face,
                                      const int& type,
                                      const int& btype,
                                      const T& alpha,
                                      const T& beta)
{
    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};

    auto d0 = make_md_span(d0_, dgcw, j, i + 1);
    auto d1 = make_md_span(d1_, dgcw, i, j + 1);
    auto u = make_md_span(u_, ugcw, j, i);

    // boundary bounds
    auto ib = Ib{std::max(bLo[0], i0), std::min(bHi[0], i1)};
    auto jb = Jb{std::max(bLo[1], j0), std::min(bHi[1], j1)};
    // lo/hi bounds
    const auto il = Ib{bLo[0], bLo[0]}, ih = Ib{bHi[0], bHi[0]};
    const auto jl = Jb{bLo[1], bLo[1]}, jh = Jb{bHi[1], bHi[1]};

    auto t1 = [&](auto&& u1, auto&& u2, auto&& d, T h, auto&&... p) mutable {
        switch (btype) {
        case 0:
            if (exOrder == 0) {
                with_domain(FWD(p)...)(u = 0);
                u.copy_to(u_);
            } else if (exOrder == 1) {
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

    auto t2 = [&](auto&& u1, auto&& u2, auto&& u3, auto&&... p) mutable {
        with_domain(FWD(p)...)(u = u2 + u3 - u1);
        u.copy_to(u_);
    };

    switch (type) {
    case 1:
        switch (face) {
        case 0:
            t1(u.shift_x(1), u.shift_x(2), d0.shift_x(1), dx[0], jb, il);
            break;
        case 1:
            t1(u.shift_x(-1), u.shift_x(-2), d0, dx[0], jb, ih);
            break;
        case 2:
            t1(u.shift_y(1), u.shift_y(2), d1.shift_y(1), dx[1], jl, ib);
            break;
        case 3:
            t1(u.shift_y(-1), u.shift_y(-2), d1, dx[1], jh, ib);
            break;
        }
        return;
    case 2:
        switch (face) {
        case 0:
            t2(u.shift_xy(1, 1), u.shift_x(1), u.shift_y(1), jl, il);
            break;
        case 1:
            t2(u.shift_xy(-1, 1), u.shift_x(-1), u.shift_y(1), jl, ih);
            break;
        case 2:
            t2(u.shift_xy(1, -1), u.shift_x(1), u.shift_y(-1), jh, il);
            break;
        case 3:
            t2(u.shift_xy(-1, -1), u.shift_x(-1), u.shift_y(-1), jh, ih);
            break;
        }
        return;
    }
}

template <typename T>
void cd_correct_bc_2d_cuda<T>::set_poisson_bc(const int& i0,
                                              const int& i1,
                                              const int& j0,
                                              const int& j1,
                                              const T* dx,
                                              const int& ugcw,
                                              T* u_,
                                              const int* bLo,
                                              const int* bHi,
                                              const int& exOrder,
                                              const int& face,
                                              const int& type,
                                              const int& btype,
                                              const T& alpha,
                                              const T& beta)
{
    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};

    auto u = make_md_span(u_, ugcw, j, i);

    // boundary bounds
    auto ib = Ib{std::max(bLo[0], i0), std::min(bHi[0], i1)};
    auto jb = Jb{std::max(bLo[1], j0), std::min(bHi[1], j1)};
    // lo/hi bounds
    const auto il = Ib{bLo[0], bLo[0]}, ih = Ib{bHi[0], bHi[0]};
    const auto jl = Jb{bLo[1], bLo[1]}, jh = Jb{bHi[1], bHi[1]};

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
                with_domain(FWD(p)...)(
                    u = ((2 * alpha - h * beta) / (2 * alpha + h * beta)) * u1);
                u.copy_to(u_);
            } else if (exOrder == 2) {
                with_domain(FWD(p)...)(
                    u = ((8 * alpha - 6 * beta * h) / ((8 * alpha + 3 * beta * h))) * u1 +
                        h / ((8 * alpha + 3 * beta * h)) * u2);
                u.copy_to(u_);
            }
            return;
        }
    };

    auto t2 = [&](auto&& u1, auto&& u2, auto&& u3, auto&&... p) mutable {
        with_domain(FWD(p)...)(u = u2 + u3 - u1);
        u.copy_to(u_);
    };

    switch (type) {
    case 1:
        switch (face) {
        case 0:
            t1(u.shift_x(1), u.shift_x(2), dx[0], jb, il);
            break;
        case 1:
            t1(u.shift_x(-1), u.shift_x(-2), dx[0], jb, ih);
            break;
        case 2:
            if (btype != 4) t1(u.shift_y(1), u.shift_y(2), dx[1], jl, ib);
            break;
        case 3:
            if (btype != 4) t1(u.shift_y(-1), u.shift_y(-2), dx[1], jh, ib);
            break;
        }
        return;
    case 2:
        switch (face) {
        case 0:
            t2(u.shift_xy(1, 1), u.shift_x(1), u.shift_y(1), jl, il);
            break;
        case 1:
            t2(u.shift_xy(-1, 1), u.shift_x(-1), u.shift_y(1), jl, ih);
            break;
        case 2:
            t2(u.shift_xy(1, -1), u.shift_x(1), u.shift_y(-1), jh, il);
            break;
        case 3:
            t2(u.shift_xy(-1, -1), u.shift_x(-1), u.shift_y(-1), jh, ih);
            break;
        }
        return;
    }
}

template <typename T>
void cd_correct_bc_2d_cuda<T>::set_corner_bc(const int& i0,
                                             const int& i1,
                                             const int& j0,
                                             const int& j1,
                                             const int& gcw,
                                             const T* dx,
                                             const T* d0_,
                                             const T* d1_,
                                             T* u_,
                                             const int* bLo,
                                             const int* bHi,
                                             const int& exOrder,
                                             const int& face,
                                             const int& type,
                                             const int& btype)
{
    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};

    auto u = make_md_span(u_, gcw, j, i);

    // lo/hi bounds
    const auto il = Ib{bLo[0], bLo[0]}, ih = Ib{bHi[0], bHi[0]};
    const auto jl = Jb{bLo[1], bLo[1]}, jh = Jb{bHi[1], bHi[1]};

    auto f = [&](auto&& u1, auto&&... p) mutable {
        with_domain(FWD(p)...)(u = -5 * u1);
        u.copy_to(u_);
    };

    switch (face) {
    case 0:
        f(u.shift_xy(1, 1), jl, il);
        break;
    case 1:
        f(u.shift_xy(-1, 1), jl, ih);
        break;
    case 2:
        f(u.shift_xy(1, -1), jh, il);
        break;
    case 3:
        f(u.shift_xy(-1, -1), jh, ih);
        break;
    }
}

template <typename T>
void cd_correct_bc_2d_cuda<T>::set_homogenous_bc(const int& i0,
                                                 const int& j0,
                                                 const int& i1,
                                                 const int& j1,
                                                 const int& face,
                                                 const int* bLo,
                                                 const int* bHi,
                                                 const int& exOrder,
                                                 T* u_)
{
    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};

    auto u = make_md_span(u_, 1, j, i);

    // boundary bounds
    const auto ib = Ib{std::max(i0, bLo[0]), std::min(i1, bHi[0])};
    const auto jb = Jb{std::max(j0, bLo[1]), std::min(j1, bHi[1])};
    // lo/hi bounds
    const auto il = Ib{i0 - 1, i0 - 1}, ih = Ib{i1 + 1, i1 + 1};
    const auto jl = Jb{j0 - 1, j0 - 1}, jh = Jb{j1 + 1, j1 + 1};

    auto f = [&](auto&& u1, auto&& u2, auto&& u3, auto&& u4, auto&&... p) mutable {
        T t13 = 1.0 / 3.0;
        T t15 = 1.0 / 5.0;
        T t45 = 4.0 / 5.0;
        T t17 = 1.0 / 7.0;
        switch (exOrder) {
        case 1:
            with_domain(FWD(p)...)(u = -1 * u1);
            break;
        case 2:
            with_domain(FWD(p)...)(u = -2 * u1 + t13 * u2);
            break;
        case 3:
            with_domain(FWD(p)...)(u = -3 * u1 + u2 - t15 * u3);
            break;
        case 4:
            with_domain(FWD(p)...)(u = -4 * u1 + 2 * u2 - t45 * u3 + t17 * u4);
            break;
        }
    };

    auto g = [](auto&& u, auto&& v, auto&&... p) { with_domain(FWD(p)...)(u = v); };

    switch (face) {
    case 0:
        f(u.shift_x(1), u.shift_x(2), u.shift_x(3), u.shift_x(4), jb, il);
        g(u, u.shift_xy(1, 1), jl, il);
        g(u, u.shift_xy(1, -1), jh, il);
        u.copy_to(u_);
        break;
    case 1:
        f(u.shift_x(-1), u.shift_x(-2), u.shift_x(-3), u.shift_x(-4), jb, ih);
        g(u, u.shift_xy(-1, 1), jl, ih);
        g(u, u.shift_xy(-1, -1), jh, ih);
        u.copy_to(u_);
        break;
    case 2:
        f(u.shift_y(1), u.shift_y(2), u.shift_y(3), u.shift_y(4), jl, ib);
        g(u, u.shift_xy(1, 1), jl, il);
        g(u, u.shift_xy(-1, 1), jl, ih);
        u.copy_to(u_);
        break;
    case 3:
        f(u.shift_y(-1), u.shift_y(-2), u.shift_y(-3), u.shift_y(-4), jh, ib);
        g(u, u.shift_xy(1, -1), jh, il);
        g(u, u.shift_xy(-1, -1), jh, ih);
        u.copy_to(u_);
        break;
    }
}

template struct cd_correct_bc_2d_cuda<double>;
template struct cd_correct_bc_2d_cuda<float>;
