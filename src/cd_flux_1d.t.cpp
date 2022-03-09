#include <array>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <iostream>

#include "prototypes1d.h"
#include "random.hpp"
#include <algorithm>

#include "cd_flux_1d_cuda.hpp"
#include "md_host_vector.hpp"

using Catch::Matchers::Approx;

constexpr auto f = []() { return pick(0.0, 1.0); };
using B = hbounds;
using Dims = std::array<B, 1>;

TEST_CASE("flux")
{
    using T = double;
    using vec = md_host_vector<T, 1>;

    const int i0 = 1;
    const int i1 = 51;
    const std::array dx{0.3};
    const int gcw = 2;

    vec u(B{i0 - gcw, i1 + gcw});

    vec b0(B{i0, i1 + 1});
    vec f0(B{i0, i1 + 1});

    randomize();
    std::generate(b0.begin(), b0.end(), f);
    std::generate(u.begin(), u.end(), f);

    celldiffusionflux1d_(i0, i1, dx.data(), b0.data(), gcw, u.data(), f0.data());

    std::vector<T> f0_cuda(f0.size());

    cdf_1d_cuda<T>::flux(i0, i1, dx.data(), b0.data(), gcw, u.data(), &f0_cuda[0]);

    REQUIRE_THAT(f0.vec(), Approx(f0_cuda));
}

TEST_CASE("poisson")
{
    using T = double;
    using vec = md_host_vector<T, 1>;

    const int i0 = 35;
    const int i1 = 94;
    const std::array dx{0.121};
    const int gcw = 2;

    vec u(B{i0 - gcw, i1 + gcw});
    vec f0(B{i0, i1 + 1});

    randomize();
    std::generate(u.begin(), u.end(), f);

    cellpoissonflux1d_(i0, i1, dx.data(), gcw, u.data(), f0.data());

    std::vector<T> f0_cuda(f0.size());

    cdf_1d_cuda<T>::poisson_flux(i0, i1, dx.data(), gcw, u.data(), &f0_cuda[0]);

    REQUIRE_THAT(f0.vec(), Approx(f0_cuda));
}
