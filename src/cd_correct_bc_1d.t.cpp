#include <array>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <iostream>

#include "prototypes1d.h"
#include "random.hpp"
#include <algorithm>

#include "cd_correct_bc_1d_cuda.hpp"
#include "md_host_vector.hpp"

using Catch::Matchers::Approx;

template <typename T>
using bc = cd_correct_bc_1d_cuda<T>;

constexpr auto f = []() { return pick(0.0, 1.0); };
using B = hbounds;
static constexpr auto N = 1;
using Dims = std::array<B, N>;

TEST_CASE("set")
{
    using T = double;
    using vec = md_host_vector<T, N>;
    randomize();

    const int i0 = 10;
    const int i1 = 20;
    const std::array dx{0.1, 0.5, 0.3};
    const int dgcw = pick(1, 3);
    const int ugcw = pick(1, 3);
    const T alpha = pick(0.1, 10.0);
    const T beta = pick(0.1, 10.0);

    vec u(B{i0 - ugcw, i1 + ugcw});

    vec d0(B{i0 - dgcw, i1 + 1 + dgcw});

    std::generate(d0.begin(), d0.end(), f);
    std::generate(u.begin(), u.end(), f);
    std::vector<T> u_cuda{u.vec()};

    const std::array bLo{i0 + pick(1, 3)};
    const std::array bHi{i1 - pick(1, 3)};

    auto exOrder = GENERATE(1, 2);
    auto face = GENERATE(0, 1);
    auto btype = GENERATE(0, 1, 4);

    cellsetcorrectionbc1d_(i0,
                           i1,
                           dx.data(),
                           dgcw,
                           d0.data(),
                           ugcw,
                           u.data(),
                           bLo.data(),
                           bHi.data(),
                           exOrder,
                           face,
                           btype,
                           alpha,
                           beta);

    bc<T>::set_bc(i0,
                  i1,
                  dx.data(),
                  dgcw,
                  d0.data(),
                  ugcw,
                  &u_cuda[0],
                  bLo.data(),
                  bHi.data(),
                  exOrder,
                  face,
                  btype,
                  alpha,
                  beta);

    REQUIRE_THAT(u.vec(), Approx(u_cuda));
}

TEST_CASE("poisson")
{
    using T = double;
    using vec = md_host_vector<T, N>;
    randomize();

    const int i0 = 10;
    const int i1 = 20;
    const std::array dx{0.1, 0.5, 0.3};
    const int ugcw = pick(1, 3);
    const T alpha = pick(0.1, 10.0);
    const T beta = pick(0.1, 10.0);

    vec u(B{i0 - ugcw, i1 + ugcw});

    std::generate(u.begin(), u.end(), f);
    std::vector<T> u_cuda{u.vec()};

    const std::array bLo{i0 + pick(1, 3)};
    const std::array bHi{i1 - pick(1, 3)};

    auto exOrder = GENERATE(1, 2);
    auto face = GENERATE(0, 1);
    auto btype = GENERATE(0, 1, 4);

    cellsetpoissoncorrectionbc1d_(i0,
                                  i1,
                                  dx.data(),
                                  ugcw,
                                  u.data(),
                                  bLo.data(),
                                  bHi.data(),
                                  exOrder,
                                  face,
                                  btype,
                                  alpha,
                                  beta);

    bc<T>::set_poisson_bc(i0,
                          i1,
                          dx.data(),
                          ugcw,
                          &u_cuda[0],
                          bLo.data(),
                          bHi.data(),
                          exOrder,
                          face,
                          btype,
                          alpha,
                          beta);

    REQUIRE_THAT(u.vec(), Approx(u_cuda));
}
