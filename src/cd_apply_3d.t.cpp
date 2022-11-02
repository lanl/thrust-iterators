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


#include <array>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <iostream>

#include "prototypes3d.h"
#include "random.hpp"
#include <algorithm>

#include "cd_apply_3d_cuda.hpp"
#include "md_device_vector.hpp"

using Catch::Matchers::Approx;

constexpr auto f = []() { return pick(0.0, 1.0); };

TEST_CASE("diffusion v1 res")
{
    using T = double;
    using vec = std::vector<T>;
    randomize();

    const int i0 = 7, j0 = 99, k0 = 12;
    const int i1 = 54, j1 = 250, k1 = 43;
    const std::array dx{0.98989, 0.15, 0.13};
    const int ugcw = pick(2, 10);
    const int fgcw = pick(1, 4);
    const int rgcw = pick(2, 5);
    const int agcw = pick(3, 9);

    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};
    const auto k = Kb{k0, k1};

    auto a = make_md_vec<T>(agcw, k, j, i);
    auto u = make_md_vec<T>(ugcw, k, j, i);
    auto ff = make_md_vec<T>(fgcw, k, j, i);
    auto res = make_md_vec<T>(rgcw, k, j, i);
    auto res_cuda = make_md_vec<T>(rgcw, k, j, i);
    auto f0 = make_md_vec<T>(k, j, i + 1);
    auto f1 = make_md_vec<T>(i, k, j + 1);
    auto f2 = make_md_vec<T>(j, i, k + 1);

    a.fill_random();
    u.fill_random();
    ff.fill_random();
    f0.fill_random();
    f1.fill_random();
    f2.fill_random();

    T beta = f();
    T alpha = f();

    celldiffusionv1res3d_(i0,
                          i1,
                          j0,
                          j1,
                          k0,
                          k1,
                          alpha,
                          beta,
                          dx.data(),
                          agcw,
                          a.host_data(),
                          ugcw,
                          u.host_data(),
                          fgcw,
                          ff.host_data(),
                          f0.host_data(),
                          f1.host_data(),
                          f2.host_data(),
                          rgcw,
                          res.host_data());

    cd_apply_3d_cuda<T>::diffusion_v1_res(i0,
                                          i1,
                                          j0,
                                          j1,
                                          k0,
                                          k1,
                                          alpha,
                                          beta,
                                          dx.data(),
                                          agcw,
                                          a.data(),
                                          ugcw,
                                          u.data(),
                                          fgcw,
                                          ff.data(),
                                          f0.data(),
                                          f1.data(),
                                          f2.data(),
                                          rgcw,
                                          res_cuda.data());
    vec res_v = res;
    vec res_cuda_v = res_cuda;

    REQUIRE_THAT(res_v, Approx(res_cuda_v));
}

TEST_CASE("diffusion v2 res")
{
    using T = double;
    using vec = std::vector<T>;
    randomize();

    const int i0 = 5, j0 = 12, k0 = 98;
    const int i1 = 73, j1 = 129, k1 = 130;
    const std::array dx{0.15, 0.111, 0.1212};
    const int ugcw = pick(1, 3);
    const int fgcw = pick(1, 4);
    const int rgcw = pick(2, 5);

    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};
    const auto k = Kb{k0, k1};

    auto u = make_md_vec<T>(ugcw, k, j, i);
    auto ff = make_md_vec<T>(fgcw, k, j, i);
    auto res = make_md_vec<T>(rgcw, k, j, i);
    auto res_cuda = make_md_vec<T>(rgcw, k, j, i);
    auto f0 = make_md_vec<T>(k, j, i + 1);
    auto f1 = make_md_vec<T>(i, k, j + 1);
    auto f2 = make_md_vec<T>(j, i, k + 1);

    u.fill_random();
    ff.fill_random();
    f0.fill_random();
    f1.fill_random();
    f2.fill_random();

    T beta = f();
    T alpha = f();

    celldiffusionv2res3d_(i0,
                          i1,
                          j0,
                          j1,
                          k0,
                          k1,
                          alpha,
                          beta,
                          dx.data(),
                          ugcw,
                          u.host_data(),
                          fgcw,
                          ff.host_data(),
                          f0.host_data(),
                          f1.host_data(),
                          f2.host_data(),
                          rgcw,
                          res.host_data());

    cd_apply_3d_cuda<T>::diffusion_v2_res(i0,
                                          i1,
                                          j0,
                                          j1,
                                          k0,
                                          k1,
                                          alpha,
                                          beta,
                                          dx.data(),
                                          ugcw,
                                          u.data(),
                                          fgcw,
                                          ff.data(),
                                          f0.data(),
                                          f1.data(),
                                          f2.data(),
                                          rgcw,
                                          res_cuda.data());

    vec res_v = res;
    vec res_cuda_v = res_cuda;

    REQUIRE_THAT(res_v, Approx(res_cuda_v));
}

TEST_CASE("poisson v1 res")
{
    using T = double;
    using vec = std::vector<T>;
    randomize();

    const int i0 = 5, j0 = 12, k0 = 32;
    const int i1 = 73, j1 = 129, k1 = 75;
    const std::array dx{0.15, 0.987654321, 0.13};
    const int fgcw = pick(1, 4);
    const int rgcw = pick(2, 5);

    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};
    const auto k = Kb{k0, k1};

    auto ff = make_md_vec<T>(fgcw, k, j, i);
    auto res = make_md_vec<T>(rgcw, k, j, i);
    auto res_cuda = make_md_vec<T>(rgcw, k, j, i);
    auto f0 = make_md_vec<T>(k, j, i + 1);
    auto f1 = make_md_vec<T>(i, k, j + 1);
    auto f2 = make_md_vec<T>(j, i, k + 1);

    ff.fill_random();
    f0.fill_random();
    f1.fill_random();
    f2.fill_random();

    T beta = f();
    T alpha = f();

    cellpoissonv1res3d_(i0,
                        i1,
                        j0,
                        j1,
                        k0,
                        k1,
                        beta,
                        dx.data(),
                        fgcw,
                        ff.host_data(),
                        f0.host_data(),
                        f1.host_data(),
                        f2.host_data(),
                        rgcw,
                        res.host_data());

    cd_apply_3d_cuda<T>::poisson_v1_res(i0,
                                        i1,
                                        j0,
                                        j1,
                                        k0,
                                        k1,
                                        beta,
                                        dx.data(),
                                        fgcw,
                                        ff.data(),
                                        f0.data(),
                                        f1.data(),
                                        f2.data(),
                                        rgcw,
                                        res_cuda.data());

    vec res_v = res;
    vec res_cuda_v = res_cuda;

    REQUIRE_THAT(res_v, Approx(res_cuda_v));
}

TEST_CASE("diffusion v1 apply")
{
    using T = double;
    using vec = std::vector<T>;
    randomize();

    const int i0 = 5, j0 = 12, k0 = 55;
    const int i1 = 73, j1 = 129, k1 = 82;
    const std::array dx{0.123456789, 0.15, 0.13};
    const int rgcw = pick(1, 4);
    const int ugcw = pick(2, 5);
    const int agcw = pick(1, 7);

    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};
    const auto k = Kb{k0, k1};

    auto a = make_md_vec<T>(agcw, k, j, i);
    auto u = make_md_vec<T>(ugcw, k, j, i);
    auto res = make_md_vec<T>(rgcw, k, j, i);
    auto res_cuda = make_md_vec<T>(rgcw, k, j, i);
    auto f0 = make_md_vec<T>(k, j, i + 1);
    auto f1 = make_md_vec<T>(i, k, j + 1);
    auto f2 = make_md_vec<T>(j, i, k + 1);

    a.fill_random();
    u.fill_random();
    f0.fill_random();
    f1.fill_random();
    f2.fill_random();

    T beta = f();
    T alpha = f();

    celldiffusionv1apply3d_(i0,
                            i1,
                            j0,
                            j1,
                            k0,
                            k1,
                            alpha,
                            beta,
                            dx.data(),
                            agcw,
                            a.host_data(),
                            ugcw,
                            u.host_data(),
                            f0.host_data(),
                            f1.host_data(),
                            f2.host_data(),
                            rgcw,
                            res.host_data());

    cd_apply_3d_cuda<T>::diffusion_v1_apply(i0,
                                            i1,
                                            j0,
                                            j1,
                                            k0,
                                            k1,
                                            alpha,
                                            beta,
                                            dx.data(),
                                            agcw,
                                            a.data(),
                                            ugcw,
                                            u.data(),
                                            f0.data(),
                                            f1.data(),
                                            f2.data(),
                                            rgcw,
                                            res_cuda.data());

    vec res_v = res;
    vec res_cuda_v = res_cuda;

    REQUIRE_THAT(res_v, Approx(res_cuda_v));
}

TEST_CASE("diffusion v2 apply")
{
    using T = double;
    using vec = std::vector<T>;
    randomize();

    const int i0 = 5, j0 = 12, k0 = 22;
    const int i1 = 73, j1 = 29, k1 = 32;
    const std::array dx{0.15, 0.25, 0.13};
    const int rgcw = pick(1, 4);
    const int ugcw = pick(1, 4);

    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};
    const auto k = Kb{k0, k1};

    auto u = make_md_vec<T>(ugcw, k, j, i);
    auto res = make_md_vec<T>(rgcw, k, j, i);
    auto res_cuda = make_md_vec<T>(rgcw, k, j, i);
    auto f0 = make_md_vec<T>(k, j, i + 1);
    auto f1 = make_md_vec<T>(i, k, j + 1);
    auto f2 = make_md_vec<T>(j, i, k + 1);

    u.fill_random();
    f0.fill_random();
    f1.fill_random();
    f2.fill_random();

    T beta = f();
    T alpha = f();

    celldiffusionv2apply3d_(i0,
                            i1,
                            j0,
                            j1,
                            k0,
                            k1,
                            alpha,
                            beta,
                            dx.data(),
                            ugcw,
                            u.host_data(),
                            f0.host_data(),
                            f1.host_data(),
                            f2.host_data(),
                            rgcw,
                            res.host_data());

    cd_apply_3d_cuda<T>::diffusion_v2_apply(i0,
                                            i1,
                                            j0,
                                            j1,
                                            k0,
                                            k1,
                                            alpha,
                                            beta,
                                            dx.data(),
                                            ugcw,
                                            u.data(),
                                            f0.data(),
                                            f1.data(),
                                            f2.data(),
                                            rgcw,
                                            res_cuda.data());

    vec res_v = res;
    vec res_cuda_v = res_cuda;

    REQUIRE_THAT(res_v, Approx(res_cuda_v));
}

TEST_CASE("poisson v2 apply")
{
    using T = double;
    using vec = std::vector<T>;
    randomize();

    const int i0 = 35, j0 = 12, k0 = 45;
    const int i1 = 47, j1 = 92, k1 = 111;
    const std::array dx{0.15, 0.7, 0.13};
    const int rgcw = 2;

    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};
    const auto k = Kb{k0, k1};

    auto res = make_md_vec<T>(rgcw, k, j, i);
    auto res_cuda = make_md_vec<T>(rgcw, k, j, i);
    auto f0 = make_md_vec<T>(k, j, i + 1);
    auto f1 = make_md_vec<T>(i, k, j + 1);
    auto f2 = make_md_vec<T>(j, i, k + 1);

    f0.fill_random();
    f1.fill_random();
    f2.fill_random();

    T beta = f();

    cellpoissonv2apply3d_(i0,
                          i1,
                          j0,
                          j1,
                          k0,
                          k1,
                          beta,
                          dx.data(),
                          f0.host_data(),
                          f1.host_data(),
                          f2.host_data(),
                          rgcw,
                          res.host_data());

    cd_apply_3d_cuda<T>::poisson_v2_apply(i0,
                                          i1,
                                          j0,
                                          j1,
                                          k0,
                                          k1,
                                          beta,
                                          dx.data(),
                                          f0.data(),
                                          f1.data(),
                                          f2.data(),
                                          rgcw,
                                          res_cuda.data());

    vec res_v = res;
    vec res_cuda_v = res_cuda;

    REQUIRE_THAT(res_v, Approx(res_cuda_v));
}
