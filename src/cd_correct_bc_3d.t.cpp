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
#include <catch2/generators/catch_generators.hpp>

#include <iostream>

#include "prototypes3d.h"
#include "random.hpp"
#include <algorithm>

#include "cd_correct_bc_3d_cuda.hpp"
#include "md_device_vector.hpp"

using Catch::Matchers::Approx;

template <typename T>
using bc = cd_correct_bc_3d_cuda<T>;

constexpr auto f = []() { return pick(0.0, 1.0); };
template <typename T>
void compare(const std::vector<T>& t, const std::vector<T>& u)
{
    REQUIRE_THAT(t, Approx(u));
}

template <typename T>
void compare(const std::vector<T>& t, const thrust::device_vector<T>& u)
{
    compare<T>(t, to_std(u));
}

TEST_CASE("set")
{
    using T = double;
    randomize();

    const int i0 = 10, j0 = 11, k0 = 12;
    const int i1 = 20, j1 = 30, k1 = 33;
    const std::array dx{0.1, 0.5, 0.3};
    const int dgcw = pick(1, 3);
    const int ugcw = pick(1, 3);
    const T alpha = pick(0.1, 10.0);
    const T beta = pick(0.1, 10.0);

    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};
    const auto k = Kb{k0, k1};

    auto u = make_md_vec<T>(ugcw, k, j, i);
    auto d0 = make_md_vec<T>(dgcw, k, j, i + 1);
    auto d1 = make_md_vec<T>(dgcw, i, k, j + 1);
    auto d2 = make_md_vec<T>(dgcw, j, i, k + 1);

    u.fill_random();
    d0.fill_random();
    d1.fill_random();
    d2.fill_random();

    thrust::device_vector<T> u_cuda = u.host();

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
                           d0.host_data(),
                           d1.host_data(),
                           d2.host_data(),
                           ugcw,
                           u.host_data(),
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
                  thrust::raw_pointer_cast(u_cuda.data()),
                  bLo.data(),
                  bHi.data(),
                  exOrder,
                  face,
                  type,
                  btype,
                  alpha,
                  beta);

    compare<T>(u, u_cuda);
}

TEST_CASE("poisson")
{
    using T = double;
    randomize();

    const int i0 = 10, j0 = 11, k0 = 12;
    const int i1 = 20, j1 = 30, k1 = 33;
    const std::array dx{0.1, 0.5, 0.3};
    const int ugcw = pick(1, 3);
    const T alpha = pick(0.1, 10.0);
    const T beta = pick(0.1, 10.0);

    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};
    const auto k = Kb{k0, k1};

    auto u = make_md_vec<T>(ugcw, k, j, i);

    u.fill_random();

    thrust::device_vector<T> u_cuda = u.host();

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
                                  u.host_data(),
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
                          thrust::raw_pointer_cast(u_cuda.data()),
                          bLo.data(),
                          bHi.data(),
                          exOrder,
                          face,
                          type,
                          btype,
                          alpha,
                          beta);

    compare<T>(u, u_cuda);
}
