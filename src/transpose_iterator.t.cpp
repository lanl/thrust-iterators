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


#include <catch2/catch_test_macros.hpp>

#include "transpose_iterator_test.hpp"
#include <array>
#include <vector>

#include "random.hpp"
#include <algorithm>

constexpr auto f = []() { return pick(0.0, 1.0); };

TEST_CASE("transpose 2D basic")
{
    using T = double;
    constexpr int N = 2;

    int sz[N] = {2, 3};

    std::vector<T> v{1, 2, 3, 4, 5, 6};
    std::vector<T> u(v.size(), 0);

    transpose_test_cuda<T, 0, 1>::init(&v[0], sz, &u[0]);
    REQUIRE(u == v);
    transpose_test_cuda<T, 1, 0>::init(&v[0], sz, &u[0]);
    REQUIRE(u == std::vector<T>{1, 4, 2, 5, 3, 6});
    {
        std::vector<T> w(u.size(), 0);
        int transpose_sz [N] = {3, 2};
        transpose_test_cuda<T, 1, 0>::init(&u[0], transpose_sz, &w[0]);
        REQUIRE(w == v);
    }

}

TEST_CASE("transpose 3D")
{
    using T = double;
    constexpr int N = 3;

    int sz[N] = {3, 8, 4};
    std::vector<T> v(sz[0] * sz[1] * sz[2]);
    std::vector<T> u(v.size());

    randomize();
    std::generate(v.begin(), v.end(), f);

    transpose_test_cuda<T, 0, 1, 2>::init(&v[0], sz, &u[0]);
    REQUIRE(u == v);

    // transpose and reverse
    transpose_test_cuda<T, 2, 0, 1>::init(&v[0], sz, &u[0]);
    std::vector<T> w(u.size());
    int transpose_sz[] = {sz[2], sz[0], sz[1]};
    transpose_test_cuda<T, 1, 2, 0>::init(&u[0], transpose_sz, &w[0]);
    REQUIRE(w == v);
}
