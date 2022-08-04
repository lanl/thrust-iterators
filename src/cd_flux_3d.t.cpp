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


#include <array>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <iostream>

#include "prototypes3d.h"
#include "random.hpp"
#include <algorithm>

#include "cd_flux_3d_cuda.hpp"
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

    const int i0 = 10, j0 = 11, k0 = 12;
    const int i1 = 20, j1 = 30, k1 = 33;
    const std::array dx{0.1, 0.5, 0.3};
    const int gcw = 5;

    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};
    const auto k = Kb{k0, k1};

    auto u = make_md_vec<T>(gcw, k, j, i);
    auto b0 = make_md_vec<T>(k, j, i + 1);
    auto f0 = make_md_vec<T>(k, j, i + 1);
    auto f0_cuda = make_md_vec<T>(k, j, i + 1);
    auto b1 = make_md_vec<T>(i, k, j + 1);
    auto f1 = make_md_vec<T>(i, k, j + 1);
    auto f1_cuda = make_md_vec<T>(i, k, j + 1);
    auto b2 = make_md_vec<T>(j, i, k + 1);
    auto f2 = make_md_vec<T>(j, i, k + 1);
    auto f2_cuda = make_md_vec<T>(j, i, k + 1);

    b0.fill_random();
    b1.fill_random();
    b2.fill_random();
    u.fill_random();

    celldiffusionflux3d_(i0,
                         j0,
                         k0,
                         i1,
                         j1,
                         k1,
                         dx.data(),
                         b0.host_data(),
                         b1.host_data(),
                         b2.host_data(),
                         gcw,
                         u.host_data(),
                         f0.host_data(),
                         f1.host_data(),
                         f2.host_data());

    cdf_3d_cuda<T>::flux(i0,
                         j0,
                         k0,
                         i1,
                         j1,
                         k1,
                         dx.data(),
                         b0.data(),
                         b1.data(),
                         b2.data(),
                         gcw,
                         u.data(),
                         f0_cuda.data(),
                         f1_cuda.data(),
                         f2_cuda.data());
    compare<T>(f0, f0_cuda);
    compare<T>(f1, f1_cuda);
    compare<T>(f2, f2_cuda);
}

TEST_CASE("poisson")
{
    using T = double;
    randomize();

    const int i0 = 35, j0 = 12, k0 = 102;
    const int i1 = 94, j1 = 31, k1 = 157;
    const std::array dx{0.15, 0.13, 0.9};
    const int gcw = 2;

    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};
    const auto k = Kb{k0, k1};

    auto u = make_md_vec<T>(gcw, k, j, i);
    auto f0 = make_md_vec<T>(k, j, i + 1);
    auto f0_cuda = make_md_vec<T>(k, j, i + 1);
    auto f1 = make_md_vec<T>(i, k, j + 1);
    auto f1_cuda = make_md_vec<T>(i, k, j + 1);
    auto f2 = make_md_vec<T>(j, i, k + 1);
    auto f2_cuda = make_md_vec<T>(j, i, k + 1);

    u.fill_random();

    cellpoissonflux3d_(i0,
                       j0,
                       k0,
                       i1,
                       j1,
                       k1,
                       dx.data(),
                       gcw,
                       u.host_data(),
                       f0.host_data(),
                       f1.host_data(),
                       f2.host_data());

    cdf_3d_cuda<T>::poisson_flux(i0,
                                 j0,
                                 k0,
                                 i1,
                                 j1,
                                 k1,
                                 dx.data(),
                                 gcw,
                                 u.data(),
                                 f0_cuda.data(),
                                 f1_cuda.data(),
                                 f2_cuda.data());

    compare<T>(f0, f0_cuda);
    compare<T>(f1, f1_cuda);
    compare<T>(f2, f2_cuda);
}
