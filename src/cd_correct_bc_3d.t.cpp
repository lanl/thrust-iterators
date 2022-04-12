#include <array>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <iostream>

#include "prototypes3d.h"
#include "random.hpp"
#include <algorithm>

#include "cd_correct_bc_3d_cuda.hpp"
#include "md_host_vector.hpp"

using Catch::Matchers::Approx;

template <typename T>
using bc = cd_correct_bc_3d_cuda<T>;

constexpr auto f = []() { return pick(0.0, 1.0); };
using B = hbounds;
static constexpr auto N = 3;
using Dims = std::array<B, N>;

TEST_CASE("set")
{
    using T = double;
    using vec = md_host_vector<T, N>;
    randomize();

    const int i0 = 10, j0 = 11, k0 = 12;
    const int i1 = 20, j1 = 30, k1 = 33;
    const std::array dx{0.1, 0.5, 0.3};
    const int dgcw = pick(1, 3);
    const int ugcw = pick(1, 3);
    const T alpha = pick(0.1, 10.0);
    const T beta = pick(0.1, 10.0);

    vec u(B{k0 - ugcw, k1 + ugcw}, B{j0 - ugcw, j1 + ugcw}, B{i0 - ugcw, i1 + ugcw});

    vec d0(B{k0 - dgcw, k1 + dgcw}, B{j0 - dgcw, j1 + dgcw}, B{i0 - dgcw, i1 + 1 + dgcw});
    vec d1(B{i0 - dgcw, i1 + dgcw}, B{k0 - dgcw, k1 + dgcw}, B{j0 - dgcw, j1 + 1 + dgcw});
    vec d2(B{j0 - dgcw, j1 + dgcw}, B{i0 - dgcw, i1 + dgcw}, B{k0 - dgcw, k1 + 1 + dgcw});

    std::generate(d0.begin(), d0.end(), f);
    std::generate(d1.begin(), d1.end(), f);
    std::generate(d2.begin(), d2.end(), f);
    std::generate(u.begin(), u.end(), f);
    std::vector<T> u_cuda{u.vec()};

    const std::array bLo{i0 + pick(1, 3), j0 + pick(1, 3), k0 + pick(1, 3)};
    const std::array bHi{i1 - pick(1, 3), j1 - pick(1, 3), k1 - pick(1, 3)};

    auto exOrder = GENERATE(1, 2);
    auto face = GENERATE(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
    auto type = GENERATE(1, 2, 3);
    auto btype = GENERATE(0, 1, 4);

    cellsetcorrectionbc3d_(i0,
                           i1,
                           j0,
                           j1,
                           k0,
                           k1,
                           dx.data(),
                           dgcw,
                           d0.data(),
                           d1.data(),
                           d2.data(),
                           ugcw,
                           u.data(),
                           bLo.data(),
                           bHi.data(),
                           exOrder,
                           face,
                           type,
                           btype,
                           alpha,
                           beta);

    bc<T>::set_bc(i0,
                  i1,
                  j0,
                  j1,
                  k0,
                  k1,
                  dx.data(),
                  dgcw,
                  d0.data(),
                  d1.data(),
                  d2.data(),
                  ugcw,
                  &u_cuda[0],
                  bLo.data(),
                  bHi.data(),
                  exOrder,
                  face,
                  type,
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

    const int i0 = 10, j0 = 11, k0 = 12;
    const int i1 = 20, j1 = 30, k1 = 33;
    const std::array dx{0.1, 0.5, 0.3};
    const int ugcw = pick(1, 3);
    const T alpha = pick(0.1, 10.0);
    const T beta = pick(0.1, 10.0);

    vec u(B{k0 - ugcw, k1 + ugcw}, B{j0 - ugcw, j1 + ugcw}, B{i0 - ugcw, i1 + ugcw});

    std::generate(u.begin(), u.end(), f);
    std::vector<T> u_cuda{u.vec()};

    const std::array bLo{i0 + pick(1, 3), j0 + pick(1, 3), k0 + pick(1, 3)};
    const std::array bHi{i1 - pick(1, 3), j1 - pick(1, 3), k1 - pick(1, 3)};

    auto exOrder = GENERATE(1, 2);
    auto face = GENERATE(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
    auto type = GENERATE(1, 2, 3);
    auto btype = GENERATE(0, 1, 4);

    cellsetpoissoncorrectionbc3d_(i0,
                                  i1,
                                  j0,
                                  j1,
                                  k0,
                                  k1,
                                  dx.data(),
                                  ugcw,
                                  u.data(),
                                  bLo.data(),
                                  bHi.data(),
                                  exOrder,
                                  face,
                                  type,
                                  btype,
                                  alpha,
                                  beta);

    bc<T>::set_poisson_bc(i0,
                          i1,
                          j0,
                          j1,
                          k0,
                          k1,
                          dx.data(),
                          ugcw,
                          &u_cuda[0],
                          bLo.data(),
                          bHi.data(),
                          exOrder,
                          face,
                          type,
                          btype,
                          alpha,
                          beta);

    REQUIRE_THAT(u.vec(), Approx(u_cuda));
}
