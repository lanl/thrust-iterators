#include <array>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <iostream>

#include "prototypes2d.h"
#include "random.hpp"
#include <algorithm>

#include "cd_flux_2d_cuda.hpp"
#include "md_host_vector.hpp"

using Catch::Matchers::Approx;

constexpr auto f = []() { return pick(0.0, 1.0); };
using B = hbounds;
using Dims = std::array<B, 2>;

TEST_CASE("flux")
{
    using T = double;
    using vec = md_host_vector<T, 2>;

    const int i0 = 1, j0 = 97;
    const int i1 = 51, j1 = 123;
    const std::array dx{0.5, 0.3};
    const int gcw = 2;

    vec u(B{j0 - gcw, j1 + gcw}, B{i0 - gcw, i1 + gcw});

    vec b0(B{j0, j1}, B{i0, i1 + 1});
    vec f0(B{j0, j1}, B{i0, i1 + 1});

    vec b1(B{i0, i1}, B{j0, j1 + 1});
    vec f1(B{i0, i1}, B{j0, j1 + 1});

    randomize();
    std::generate(b0.begin(), b0.end(), f);
    std::generate(b1.begin(), b1.end(), f);
    std::generate(u.begin(), u.end(), f);

    celldiffusionflux2d_(i0,
                         j0,
                         i1,
                         j1,
                         dx.data(),
                         b0.data(),
                         b1.data(),
                         gcw,
                         u.data(),
                         f0.data(),
                         f1.data());

    std::vector<T> f0_cuda(f0.size());
    std::vector<T> f1_cuda(f1.size());
    cdf_2d_cuda<T>::flux(i0,
                         j0,
                         i1,
                         j1,
                         dx.data(),
                         b0.data(),
                         b1.data(),
                         gcw,
                         u.data(),
                         &f0_cuda[0],
                         &f1_cuda[0]);
    REQUIRE_THAT(f0.vec(), Approx(f0_cuda));
    REQUIRE_THAT(f1.vec(), Approx(f1_cuda));
}

TEST_CASE("poisson")
{
    using T = double;
    using vec = md_host_vector<T, 2>;

    const int i0 = 35, j0 = 12;
    const int i1 = 94, j1 = 19;
    const std::array dx{0.15, 0.13};
    const int gcw = 2;

    vec u(B{j0 - gcw, j1 + gcw}, B{i0 - gcw, i1 + gcw});
    vec f0(B{j0, j1}, B{i0, i1 + 1});
    vec f1(B{i0, i1}, B{j0, j1 + 1});

    randomize();
    std::generate(u.begin(), u.end(), f);

    cellpoissonflux2d_(i0, j0, i1, j1, dx.data(), gcw, u.data(), f0.data(), f1.data());

    std::vector<T> f0_cuda(f0.size());
    std::vector<T> f1_cuda(f1.size());
    cdf_2d_cuda<T>::poisson_flux(
        i0, j0, i1, j1, dx.data(), gcw, u.data(), &f0_cuda[0], &f1_cuda[0]);

    REQUIRE_THAT(f0.vec(), Approx(f0_cuda));
    REQUIRE_THAT(f1.vec(), Approx(f1_cuda));
}
