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
#include "md_device_vector.hpp"

using Catch::Matchers::Approx;

template <typename T>
using bc = cd_correct_bc_1d_cuda<T>;

constexpr auto f = []() { return pick(0.0, 1.0); };

template <typename T>
void compare(const std::vector<T>& t, const std::vector<T>& u)
{
    REQUIRE_THAT(t, Approx(u));
}

TEST_CASE("set")
{
    using T = double;
    randomize();

    const int i0 = 10;
    const int i1 = 20;
    const std::array dx{0.1, 0.5, 0.3};
    const int dgcw = pick(1, 3);
    const int ugcw = pick(1, 3);
    const T alpha = pick(0.1, 10.0);
    const T beta = pick(0.1, 10.0);

    const auto i = Ib{i0, i1};

    auto u = make_md_vec<T>(ugcw, i);
    auto d0 = make_md_vec<T>(dgcw, i + 1);

    u.fill_random();
    d0.fill_random();

    thrust::device_vector<T> u_cuda = u.host();

    const std::array bLo{i0 + pick(1, 3)};
    const std::array bHi{i1 - pick(1, 3)};

    auto exOrder = GENERATE(1, 2);
    auto face = GENERATE(0, 1);
    auto btype = GENERATE(0, 1, 4);

    cellsetcorrectionbc1d_(i0,
                           i1,
                           dx.data(),
                           dgcw,
                           d0.host_data(),
                           ugcw,
                           u.host_data(),
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
                  thrust::raw_pointer_cast(u_cuda.data()),
                  bLo.data(),
                  bHi.data(),
                  exOrder,
                  face,
                  btype,
                  alpha,
                  beta);

    compare<T>(u, to_std(u_cuda));
}

TEST_CASE("poisson")
{
    using T = double;
    randomize();

    const int i0 = 10;
    const int i1 = 20;
    const std::array dx{0.1, 0.5, 0.3};
    const int ugcw = pick(1, 3);
    const T alpha = pick(0.1, 10.0);
    const T beta = pick(0.1, 10.0);

    const auto i = Ib{i0, i1};

    auto u = make_md_vec<T>(ugcw, i);

    u.fill_random();

    thrust::device_vector<T> u_cuda = u.host();

    const std::array bLo{i0 + pick(1, 3)};
    const std::array bHi{i1 - pick(1, 3)};

    auto exOrder = GENERATE(1, 2);
    auto face = GENERATE(0, 1);
    auto btype = GENERATE(0, 1, 4);

    cellsetpoissoncorrectionbc1d_(i0,
                                  i1,
                                  dx.data(),
                                  ugcw,
                                  u.host_data(),
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
                          thrust::raw_pointer_cast(u_cuda.data()),
                          bLo.data(),
                          bHi.data(),
                          exOrder,
                          face,
                          btype,
                          alpha,
                          beta);

    compare<T>(u, to_std(u_cuda));
}
