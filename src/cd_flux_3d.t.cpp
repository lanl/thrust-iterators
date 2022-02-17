#include <array>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <iostream>

#include "prototypes3d.h"
#include "random.hpp"
#include <algorithm>

#include "cd_flux_3d_cuda.hpp"
#include "md_host_vector.hpp"

using Catch::Matchers::Approx;

constexpr auto f = []() { return pick(0.0, 1.0); };
using B = hbounds;
static constexpr auto N = 3;
using Dims = std::array<B, N>;

TEST_CASE("flux")
{
    using T = double;
    using vec = md_host_vector<T, N>;
    randomize();

    const int i0 = 10, j0 = 11, k0 = 12;
    const int i1 = 51, j1 = 30, k1 = 33;
    const std::array dx{0.1, 0.5, 0.3};
    const int gcw = 2;

    vec u(B{k0 - gcw, k0 + gcw}, B{j0 - gcw, j1 + gcw}, B{i0 - gcw, i1 + gcw});

    vec b0(B{k0, k1}, B{j0, j1}, B{i0, i1 + 1});
    vec f0(B{k0, k1}, B{j0, j1}, B{i0, i1 + 1});

    vec b1(B{i0, i1}, B{k0, k1}, B{j0, j1 + 1});
    vec f1(B{i0, i1}, B{k0, k1}, B{j0, j1 + 1});

    vec b2(B{j0, j1}, B{i0, i1}, B{k0, k1 + 1});
    vec f2(B{j0, j1}, B{i0, i1}, B{k0, k1 + 1});

    std::generate(b0.begin(), b0.end(), f);
    std::generate(b1.begin(), b1.end(), f);
    std::generate(b2.begin(), b2.end(), f);
    std::generate(u.begin(), u.end(), f);

    celldiffusionflux3d_(i0,
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
                         f0.data(),
                         f1.data(),
                         f2.data());

    std::vector<T> f0_cuda(f0.size());
    std::vector<T> f1_cuda(f1.size());
    std::vector<T> f2_cuda(f2.size());
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
                         &f0_cuda[0],
                         &f1_cuda[0],
                         &f2_cuda[0]);

    REQUIRE_THAT(f0.vec(), Approx(f0_cuda));
    REQUIRE_THAT(f1.vec(), Approx(f1_cuda));
    REQUIRE_THAT(f2.vec(), Approx(f2_cuda));
}

TEST_CASE("poisson")
{
    using T = double;
    using vec = md_host_vector<T, N>;

    randomize();

    const int i0 = 35, j0 = 12, k0 = 102;
    const int i1 = 94, j1 = 31, k1 = 157;
    const std::array dx{0.15, 0.13, 0.9};
    const int gcw = 2;

    vec u(B{k0 - gcw, k1 + gcw}, B{j0 - gcw, j1 + gcw}, B{i0 - gcw, i1 + gcw});
    vec f0(B{k0, k1}, B{j0, j1}, B{i0, i1 + 1});
    vec f1(B{i0, i1}, B{k0, k1}, B{j0, j1 + 1});
    vec f2(B{j0, j1}, B{i0, i1}, B{k0, k1 + 1});

    std::generate(u.begin(), u.end(), f);

    cellpoissonflux3d_(i0,
                       j0,
                       k0,
                       i1,
                       j1,
                       k1,
                       dx.data(),
                       gcw,
                       u.data(),
                       f0.data(),
                       f1.data(),
                       f2.data());

    std::vector<T> f0_cuda(f0.size());
    std::vector<T> f1_cuda(f1.size());
    std::vector<T> f2_cuda(f2.size());
    cdf_3d_cuda<T>::poisson_flux(i0,
                                 j0,
                                 k0,
                                 i1,
                                 j1,
                                 k1,
                                 dx.data(),
                                 gcw,
                                 u.data(),
                                 &f0_cuda[0],
                                 &f1_cuda[0],
                                 &f2_cuda[0]);

    REQUIRE_THAT(f0.vec(), Approx(f0_cuda));
    REQUIRE_THAT(f1.vec(), Approx(f1_cuda));
    REQUIRE_THAT(f2.vec(), Approx(f2_cuda));
}
