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

#include "prototypes2d.h"
#include "random.hpp"
#include <algorithm>

#include "cd_flux_2d_cuda.hpp"
#include "md_device_vector.hpp"

using Catch::Matchers::Approx;

constexpr auto f = []() { return pick(0.0, 1.0); };

template <typename T>
void compare(const std::vector<T>& t, const std::vector<T>& u)
{
    REQUIRE_THAT(t, Approx(u));
}

TEST_CASE("flux")
{
    using T = double;
    randomize();

    const int i0 = 1, j0 = 97;
    const int i1 = 51, j1 = 123;
    const std::array dx{0.5, 0.3};
    const int gcw = 2;

    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};

    auto u = make_md_vec<T>(gcw, j, i);
    auto b0 = make_md_vec<T>(j, i + 1);
    auto f0 = make_md_vec<T>(j, i + 1);
    auto f0_cuda = make_md_vec<T>(j, i + 1);
    auto b1 = make_md_vec<T>(i, j + 1);
    auto f1 = make_md_vec<T>(i, j + 1);
    auto f1_cuda = make_md_vec<T>(i, j + 1);

    b0.fill_random();
    b1.fill_random();
    u.fill_random();

    celldiffusionflux2d_(i0,
                         j0,
                         i1,
                         j1,
                         dx.data(),
                         b0.host_data(),
                         b1.host_data(),
                         gcw,
                         u.host_data(),
                         f0.host_data(),
                         f1.host_data());

    cdf_2d_cuda<T>::flux(i0,
                         j0,
                         i1,
                         j1,
                         dx.data(),
                         b0.data(),
                         b1.data(),
                         gcw,
                         u.data(),
                         f0_cuda.data(),
                         f1_cuda.data());

    compare<T>(f0, f0_cuda);
    compare<T>(f1, f1_cuda);
}

TEST_CASE("poisson")
{
    using T = double;
    randomize();

    const int i0 = 35, j0 = 12;
    const int i1 = 94, j1 = 19;
    const std::array dx{0.15, 0.13};
    const int gcw = 2;

    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};

    auto u = make_md_vec<T>(gcw, j, i);
    auto f0 = make_md_vec<T>(j, i + 1);
    auto f0_cuda = make_md_vec<T>(j, i + 1);
    auto f1 = make_md_vec<T>(i, j + 1);
    auto f1_cuda = make_md_vec<T>(i, j + 1);

    u.fill_random();

    cellpoissonflux2d_(
        i0, j0, i1, j1, dx.data(), gcw, u.host_data(), f0.host_data(), f1.host_data());

    cdf_2d_cuda<T>::poisson_flux(
        i0, j0, i1, j1, dx.data(), gcw, u.data(), f0_cuda.data(), f1_cuda.data());

    compare<T>(f0, f0_cuda);
    compare<T>(f1, f1_cuda);
}
