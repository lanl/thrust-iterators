#include <array>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <iostream>

#include "prototypes2d.h"
#include "random.hpp"
#include <algorithm>

#include "cd_correct_bc_2d_cuda.hpp"
#include "md_host_vector.hpp"

using Catch::Matchers::Approx;

template <typename T>
using bc = cd_correct_bc_2d_cuda<T>;

constexpr auto f = []() { return pick(0.0, 1.0); };
using B = hbounds;
static constexpr auto N = 2;
using Dims = std::array<B, N>;

TEST_CASE("set")
{
    using T = double;
    using vec = md_host_vector<T, N>;
    randomize();

    const int i0 = 10, j0 = 11;
    const int i1 = 20, j1 = 30;
    const std::array dx{0.1, 0.5, 0.3};
    const int dgcw = pick(1, 3);
    const int ugcw = pick(1, 3);
    const T alpha = pick(0.1, 10.0);
    const T beta = pick(0.1, 10.0);

    vec u(B{j0 - ugcw, j1 + ugcw}, B{i0 - ugcw, i1 + ugcw});

    vec d0(B{j0 - dgcw, j1 + dgcw}, B{i0 - dgcw, i1 + 1 + dgcw});
    vec d1(B{i0 - dgcw, i1 + dgcw}, B{j0 - dgcw, j1 + 1 + dgcw});

    std::generate(d0.begin(), d0.end(), f);
    std::generate(d1.begin(), d1.end(), f);
    std::generate(u.begin(), u.end(), f);
    std::vector<T> u_cuda{u.vec()};

    const std::array bLo{i0 + pick(1, 3), j0 + pick(1, 3)};
    const std::array bHi{i1 - pick(1, 3), j1 - pick(1, 3)};

    auto exOrder = GENERATE(0, 1, 2);
    auto face = GENERATE(0, 1, 2, 3);
    auto type = GENERATE(1, 2);
    auto btype = GENERATE(0, 1, 4);

    cellsetcorrectionbc2d_(i0,
                           i1,
                           j0,
                           j1,
                           dx.data(),
                           dgcw,
                           d0.data(),
                           d1.data(),
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
                  dx.data(),
                  dgcw,
                  d0.data(),
                  d1.data(),
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

    const int i0 = 10, j0 = 11;
    const int i1 = 20, j1 = 30;
    const std::array dx{0.1, 0.5, 0.3};
    const int ugcw = pick(1, 3);
    const T alpha = pick(0.1, 10.0);
    const T beta = pick(0.1, 10.0);

    vec u(B{j0 - ugcw, j1 + ugcw}, B{i0 - ugcw, i1 + ugcw});

    std::generate(u.begin(), u.end(), f);
    std::vector<T> u_cuda{u.vec()};

    const std::array bLo{i0 + pick(1, 3), j0 + pick(1, 3)};
    const std::array bHi{i1 - pick(1, 3), j1 - pick(1, 3)};

    auto exOrder = GENERATE(1, 2);
    auto face = GENERATE(0, 1, 2, 3);
    auto type = GENERATE(1, 2);
    auto btype = GENERATE(0, 1, 4);

    cellsetpoissoncorrectionbc2d_(i0,
                                  i1,
                                  j0,
                                  j1,
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

TEST_CASE("corner")
{
    using T = double;
    using vec = md_host_vector<T, N>;
    randomize();

    const int i0 = 10, j0 = 11;
    const int i1 = 20, j1 = 30;
    const std::array dx{0.1, 0.5, 0.3};
    const int ugcw = pick(1, 3);

    vec u(B{j0 - ugcw, j1 + ugcw}, B{i0 - ugcw, i1 + ugcw});

    vec d0(B{j0, j1}, B{i0, i1 + 1});
    vec d1(B{i0, i1}, B{j0, j1 + 1});

    std::generate(d0.begin(), d0.end(), f);
    std::generate(d1.begin(), d1.end(), f);
    std::generate(u.begin(), u.end(), f);
    std::vector<T> u_cuda{u.vec()};

    const std::array bLo{i0 + pick(1, 3), j0 + pick(1, 3)};
    const std::array bHi{i1 - pick(1, 3), j1 - pick(1, 3)};

    int exOrder = 0;
    auto face = GENERATE(0, 1, 2, 3);
    int type = 0;
    int btype = 0;

    cellsetinteriorcornerbc2d_(i0,
                               i1,
                               j0,
                               j1,
                               ugcw,
                               dx.data(),
                               d0.data(),
                               d1.data(),
                               u.data(),
                               bLo.data(),
                               bHi.data(),
                               exOrder,
                               face,
                               type,
                               btype);

    bc<T>::set_corner_bc(i0,
                         i1,
                         j0,
                         j1,
                         ugcw,
                         dx.data(),
                         d0.data(),
                         d1.data(),
                         &u_cuda[0],
                         bLo.data(),
                         bHi.data(),
                         exOrder,
                         face,
                         type,
                         btype);

    REQUIRE_THAT(u.vec(), Approx(u_cuda));
}

TEST_CASE("homogenous")
{
    using T = double;
    using vec = md_host_vector<T, N>;
    randomize();

    const int i0 = 10, j0 = 11;
    const int i1 = 20, j1 = 32;
    const int ugcw = 1;

    vec u(B{j0 - ugcw, j1 + ugcw}, B{i0 - ugcw, i1 + ugcw});

    std::generate(u.begin(), u.end(), f);
    std::vector<T> u_cuda{u.vec()};

    const std::array bLo{i0 + pick(1, 2), j0 + pick(1, 2)};
    const std::array bHi{i1 - pick(1, 2), j1 - pick(1, 2)};

    auto exOrder = GENERATE(1, 2, 3, 4);
    auto face = GENERATE(0, 1, 2, 3);

    cellsethomogenousbc2d_(
        i0, j0, i1, j1, face, bLo.data(), bHi.data(), exOrder, u.data());

    bc<T>::set_homogenous_bc(
        i0, j0, i1, j1, face, bLo.data(), bHi.data(), exOrder, &u_cuda[0]);

    REQUIRE_THAT(u.vec(), Approx(u_cuda));
}
