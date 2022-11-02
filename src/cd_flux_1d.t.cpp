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

#include "md_device_vector.hpp"
#include "prototypes1d.h"
#include "random.hpp"
#include <algorithm>

#include "cd_flux_1d_cuda.hpp"
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
    using vec = std::vector<T>;
    randomize();

    const int i0 = 1;
    const int i1 = 51;
    const std::array dx{0.3};
    const int gcw = 2;

    const auto i = Ib{i0, i1};

    auto u = make_md_vec<T>(gcw, i);
    auto b0 = make_md_vec<T>(i + 1);
    auto f0 = make_md_vec<T>(i + 1);
    auto f0_cuda = make_md_vec<T>(i + 1);

    b0.fill_random();
    u.fill_random();

    celldiffusionflux1d_(
        i0, i1, dx.data(), b0.host_data(), gcw, u.host_data(), f0.host_data());

    cdf_1d_cuda<T>::flux(i0, i1, dx.data(), b0.data(), gcw, u.data(), f0_cuda.data());

    compare<T>(f0, f0_cuda);
}

TEST_CASE("poisson")
{
    using T = double;
    using vec = std::vector<T>;
    randomize();

    const int i0 = 35;
    const int i1 = 94;
    const std::array dx{0.121};
    const int gcw = 2;

    const auto i = Ib{i0, i1};

    auto u = make_md_vec<T>(gcw, i);
    auto f0 = make_md_vec<T>(i + 1);
    auto f0_cuda = make_md_vec<T>(i + 1);

    u.fill_random();

    cellpoissonflux1d_(i0, i1, dx.data(), gcw, u.host_data(), f0.host_data());

    cdf_1d_cuda<T>::poisson_flux(i0, i1, dx.data(), gcw, u.data(), f0_cuda.data());

    compare<T>(f0, f0_cuda);

}
