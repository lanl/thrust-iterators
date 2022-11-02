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

#include "submatrix_iterator_test.hpp"
#include <array>
#include <vector>

TEST_CASE("submatrix_iterator basic")
{
    using T = double;
    constexpr int N = 2;

    int sz[N] = {7, 8};
    int lb[N] = {1, 2};
    int ub[N] = {4, 6};

    std::vector<T> v(sz[0] * sz[1], 1);
    REQUIRE(v.size() == sz[0] * sz[1]);

    submatrix_test_cuda<N, T>::init(&v[0], sz, lb, ub);
    REQUIRE(v[lb[0] * sz[1] + lb[1]] == -1);
    REQUIRE(v[lb[0] * sz[1] + lb[1] + 1] == -2);
    REQUIRE(v[lb[0] * sz[1] + ub[1]] == -3);
    REQUIRE(v[(lb[0] + 1) * sz[1] + lb[1]] == -4);
    REQUIRE(v[(ub[0]) * sz[1] + ub[1]] == -5);
}

TEST_CASE("submatrix 3D")
{
    using T = double;
    constexpr int N = 3;

    int sz[N] = {7, 8, 9};
    int lb[N] = {1, 2, 3};
    int ub[N] = {4, 6, 5};

    std::vector<T> v(sz[0] * sz[1] * sz[2], 1);
    REQUIRE(v.size() == sz[0] * sz[1] * sz[2]);

    auto v_at = [&](auto&& coord) {
        int j = 0;
        for (int i = 0; i < N - 1; i++) j = (j + coord[i]) * sz[i + 1];
        return v[j + coord[N - 1]];
    };

    submatrix_test_cuda<N, T>::init(&v[0], sz, lb, ub);
    REQUIRE(v_at(lb) == -1);
    REQUIRE(v_at(std::array{lb[0], lb[1], lb[2] + 1}) == -2);
    REQUIRE(v_at(std::array{lb[0], lb[1], ub[2]}) == -3);
    REQUIRE(v_at(std::array{lb[0], lb[1] + 1, lb[2]}) == -4);
    REQUIRE(v_at(ub) == -5);
    REQUIRE(v_at(std::array{lb[0], ub[1], ub[2]}) == -6);
    REQUIRE(v_at(std::array{lb[0] + 1, lb[1], lb[2]}) == -7);
}

TEST_CASE("tabulate 2D")
{
    using T = double;
    constexpr int N = 2;

    int sz[N] = {5, 4};
    int lb[N] = {2, 1};
    int ub[N] = {4, 2};

    std::vector<T> v(sz[0] * sz[1], 1);
    REQUIRE(v.size() == sz[0] * sz[1]);

    submatrix_test_cuda<N, T>::tabulate(&v[0], sz, lb, ub);
    std::vector<T> ans = {
        1, 1,  1,  1, /* row 0 */
        1, 1,  1,  1, /* row 1 */
        1, 0,  -1, 1, /* row 2 */
        1, -2, -3, 1, /* row 3 */
        1, -4, -5, 1, /* row 4 */
    };
    REQUIRE(v == ans);
}

TEST_CASE("tabulate 3D")
{
    using T = double;
    constexpr int N = 3;

    int sz[N] = {9, 7, 8};
    int lb[N] = {2, 1, 3};
    int ub[N] = {6, 5, 5};

    std::vector<T> v(sz[0] * sz[1] * sz[2], 2);
    REQUIRE(v.size() == sz[0] * sz[1] * sz[2]);

    submatrix_test_cuda<N, T>::tabulate(&v[0], sz, lb, ub);

    std::vector<T> ans(v.size(), 2);
    auto at = [&](auto&& x, auto&& coord) -> decltype(auto) {
        int j = 0;
        for (int i = 0; i < N - 1; i++) j = (j + coord[i]) * sz[i + 1];
        return x[j + coord[N - 1]];
    };

    int m = 0;
    for (int i = lb[0]; i <= ub[0]; i++)
        for (int j = lb[1]; j <= ub[1]; j++)
            for (int k = lb[2]; k <= ub[2]; k++) at(ans, std::array{i, j, k}) = -m++;

    REQUIRE(v == ans);
}
