\\ Copyright (c) 2022. Triad National Security, LLC. All rights reserved.
\\ This program was produced under U.S. Government contract
\\ 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is
\\ operated by Triad National Security, LLC for the U.S. Department of
\\ Energy/National Nuclear Security Administration. All rights in the
\\ program are reserved by Triad National Security, LLC, and the
\\ U.S. Department of Energy/National Nuclear Security
\\ Administration. The Government is granted for itself and others acting
\\ on its behalf a nonexclusive, paid-up, irrevocable worldwide license
\\ in this material to reproduce, prepare derivative works, distribute
\\ copies to the public, perform publicly and display publicly, and to
\\ permit others to do so.


#include "../cd_correct_bc_3d_cuda.hpp"

#include "md_device_span.hpp"

template <typename T>
void cd_correct_bc_3d_cuda<T>::set_bc(const int& i0,
                                      const int& i1,
                                      const int& j0,
                                      const int& j1,
                                      const int& k0,
                                      const int& k1,
                                      const T* dx,
                                      const int& dgcw,
                                      const T* d0_,
                                      const T* d1_,
                                      const T* d2_,
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
    const auto k = Kb{k0, k1};

    auto d0 = make_md_span(d0_, dgcw, k, j, i + 1);
    auto d1 = make_md_span(d1_, dgcw, i, k, j + 1);
    auto d2 = make_md_span(d2_, dgcw, j, i, k + 1);
    auto u = make_md_span(u_, ugcw, k, j, i);

    // boundary bounds
    auto ib = Ib{std::max(bLo[0], i0), std::min(bHi[0], i1)};
    auto jb = Jb{std::max(bLo[1], j0), std::min(bHi[1], j1)};
    auto kb = Kb{std::max(bLo[2], k0), std::min(bHi[2], k1)};
    // lo/hi bounds
    const auto il = Ib{bLo[0], bLo[0]}, ih = Ib{bHi[0], bHi[0]};
    const auto jl = Jb{bLo[1], bLo[1]}, jh = Jb{bHi[1], bHi[1]};
    const auto kl = Kb{bLo[2], bLo[2]}, kh = Kb{bHi[2], bHi[2]};

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

    auto t2 = [&](auto&& u1, auto&& u2, auto&& u3, auto&&... p) mutable {
        with_domain(FWD(p)...)(u = u2 + u3 - u1);
        u.copy_to(u_);
    };

    switch (type) {
    case 1:
        switch (face) {
        case 0:
            t1(u.shift_x(1), u.shift_x(2), d0.shift_x(1), dx[0], kb, jb, il);
            break;
        case 1:
            t1(u.shift_x(-1), u.shift_x(-2), d0, dx[0], kb, jb, ih);
            break;
        case 2:
            t1(u.shift_y(1), u.shift_y(2), d1.shift_y(1), dx[1], kb, jl, ib);
            break;
        case 3:
            t1(u.shift_y(-1), u.shift_y(-2), d1, dx[1], kb, jh, ib);
            break;
        case 4:
            t1(u.shift_z(1), u.shift_z(2), d2.shift_z(1), dx[2], kl, jb, ib);
            break;
        case 5:
            t1(u.shift_z(-1), u.shift_z(-2), d2, dx[2], kh, jb, ib);
            break;
        }
        return;
    case 2:
        switch (face) {
        case 0:
            t2(u.shift_xy(1, 1), u.shift_x(1), u.shift_y(1), kb, jl, il);
            break;
        case 1:
            t2(u.shift_xy(-1, 1), u.shift_x(-1), u.shift_y(1), kb, jl, il);
            break;
        case 2:
            t2(u.shift_xy(1, -1), u.shift_x(1), u.shift_y(-1), kb, jl, il);
            break;
        case 3:
            t2(u.shift_xy(-1, -1), u.shift_x(-1), u.shift_y(-1), kb, jl, il);
            break;
        case 4:
            t2(u.shift_xz(1, 1), u.shift_x(1), u.shift_z(1), kl, jb, il);
            break;
        case 5:
            t2(u.shift_xz(-1, 1), u.shift_x(-1), u.shift_z(1), kl, jb, il);
            break;
        case 6:
            t2(u.shift_xz(1, -1), u.shift_x(1), u.shift_z(-1), kl, jb, il);
            break;
        case 7:
            t2(u.shift_xz(-1, -1), u.shift_x(-1), u.shift_z(-1), kl, jb, il);
            break;
        case 8:
            t2(u.shift_yz(1, 1), u.shift_y(1), u.shift_z(1), kl, jl, ib);
            break;
        case 9:
            t2(u.shift_yz(-1, 1), u.shift_y(-1), u.shift_z(1), kl, jl, ib);
            break;
        case 10:
            t2(u.shift_yz(1, -1), u.shift_y(1), u.shift_z(-1), kl, jl, ib);
            break;
        case 11:
            t2(u.shift_yz(-1, -1), u.shift_y(-1), u.shift_z(-1), kl, jl, ib);
            break;
        }
        return;
    case 3:
        switch (face) {
        case 0:
        case 4:
            t2(u.shift_xy(1, 1), u.shift_x(1), u.shift_y(1), kl, jl, il);
            break;
        case 1:
        case 5:
            t2(u.shift_xy(-1, 1), u.shift_x(-1), u.shift_y(1), kl, jl, il);
            break;
        case 2:
        case 6:
            t2(u.shift_xy(1, -1), u.shift_x(1), u.shift_y(-1), kl, jl, il);
            break;
        case 3:
        case 7:
            t2(u.shift_xy(-1, -1), u.shift_x(-1), u.shift_y(-1), kl, jl, il);
            break;
        }
        return;
    }
}

template <typename T>
void cd_correct_bc_3d_cuda<T>::set_poisson_bc(const int& i0,
                                              const int& i1,
                                              const int& j0,
                                              const int& j1,
                                              const int& k0,
                                              const int& k1,
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
    const auto k = Kb{k0, k1};

    auto u = make_md_span(u_, ugcw, k, j, i);

    // boundary bounds
    auto ib = Ib{std::max(bLo[0], i0), std::min(bHi[0], i1)};
    auto jb = Jb{std::max(bLo[1], j0), std::min(bHi[1], j1)};
    auto kb = Kb{std::max(bLo[2], k0), std::min(bHi[2], k1)};
    // lo/hi bounds
    const auto il = Ib{bLo[0], bLo[0]}, ih = Ib{bHi[0], bHi[0]};
    const auto jl = Jb{bLo[1], bLo[1]}, jh = Jb{bHi[1], bHi[1]};
    const auto kl = Kb{bLo[2], bLo[2]}, kh = Kb{bHi[2], bHi[2]};

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
            t1(u.shift_x(1), u.shift_x(2), dx[0], kb, jb, il);
            break;
        case 1:
            t1(u.shift_x(-1), u.shift_x(-2), dx[0], kb, jb, ih);
            break;
        case 2:
            t1(u.shift_y(1), u.shift_y(2), dx[1], kb, jl, ib);
            break;
        case 3:
            t1(u.shift_y(-1), u.shift_y(-2), dx[1], kb, jh, ib);
            break;
        case 4:
            t1(u.shift_z(1), u.shift_z(2), dx[2], kl, jb, ib);
            break;
        case 5:
            t1(u.shift_z(-1), u.shift_z(-2), dx[2], kh, jb, ib);
            break;
        }
        return;
    case 2:
        switch (face) {
        case 0:
            t2(u.shift_xy(1, 1), u.shift_x(1), u.shift_y(1), kb, jl, il);
            break;
        case 1:
            t2(u.shift_xy(-1, 1), u.shift_x(-1), u.shift_y(1), kb, jl, il);
            break;
        case 2:
            t2(u.shift_xy(1, -1), u.shift_x(1), u.shift_y(-1), kb, jl, il);
            break;
        case 3:
            t2(u.shift_xy(-1, -1), u.shift_x(-1), u.shift_y(-1), kb, jl, il);
            break;
        case 4:
            t2(u.shift_xz(1, 1), u.shift_x(1), u.shift_z(1), kl, jb, il);
            break;
        case 5:
            t2(u.shift_xz(-1, 1), u.shift_x(-1), u.shift_z(1), kl, jb, il);
            break;
        case 6:
            t2(u.shift_xz(1, -1), u.shift_x(1), u.shift_z(-1), kl, jb, il);
            break;
        case 7:
            t2(u.shift_xz(-1, -1), u.shift_x(-1), u.shift_z(-1), kl, jb, il);
            break;
        case 8:
            t2(u.shift_yz(1, 1), u.shift_y(1), u.shift_z(1), kl, jl, ib);
            break;
        case 9:
            t2(u.shift_yz(-1, 1), u.shift_y(-1), u.shift_z(1), kl, jl, ib);
            break;
        case 10:
            t2(u.shift_yz(1, -1), u.shift_y(1), u.shift_z(-1), kl, jl, ib);
            break;
        case 11:
            t2(u.shift_yz(-1, -1), u.shift_y(-1), u.shift_z(-1), kl, jl, ib);
            break;
        }
        return;
    case 3:
        switch (face) {
        case 0:
        case 4:
            t2(u.shift_xy(1, 1), u.shift_x(1), u.shift_y(1), kl, jl, il);
            break;
        case 1:
        case 5:
            t2(u.shift_xy(-1, 1), u.shift_x(-1), u.shift_y(1), kl, jl, il);
            break;
        case 2:
        case 6:
            t2(u.shift_xy(1, -1), u.shift_x(1), u.shift_y(-1), kl, jl, il);
            break;
        case 3:
        case 7:
            t2(u.shift_xy(-1, -1), u.shift_x(-1), u.shift_y(-1), kl, jl, il);
            break;
        }
        return;
    }
}

template struct cd_correct_bc_3d_cuda<double>;
template struct cd_correct_bc_3d_cuda<float>;
