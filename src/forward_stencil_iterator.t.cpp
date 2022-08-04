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


#include <catch2/catch_test_macros.hpp>

#include "forward_stencil_test.hpp"
#include <vector>

#include "random.hpp"
#include <algorithm>

constexpr auto f = []() { return pick(0.0, 1.0); };

TEST_CASE("forward basic")
{
    using T = double;
    int dims[] = {10};
    std::vector<T> v(dims[0]);
    forward_stencil_test_cuda<T, 1, 0>::init(&v[0], dims);
}

TEST_CASE("forward basic 2D")
{
    using T = double;
    int dims[] = {3, 4};

    std::vector<T> v(dims[0] * dims[1]);

    forward_stencil_test_cuda<T, 2, 1>::init(&v[0], dims);
    forward_stencil_test_cuda<T, 2, 0>::init(&v[0], dims);
}

TEST_CASE("forward transformation")
{
    using T = double;
    static constexpr auto N = 1;
    int dims[] = {10};

    std::vector<T> v(dims[0]);
    randomize();
    std::generate(v.begin(), v.end(), f);

    std::vector<T> u(v.size() - 1);
    forward_stencil_test_cuda<T, N, 0>::transform(&v[0], dims, &u[0]);

    std::vector<T> result(u.size());
    for (int i = 0; i < result.size(); i++) result[i] = v[i + 1] - v[i];

    REQUIRE(u == result);
}

TEST_CASE("forward 2D transformation")
{
    using T = double;
    static constexpr auto N = 2;

    int dims[] = {33, 14};
    std::vector<T> v(dims[0] * dims[1]);
    randomize();
    std::generate(v.begin(), v.end(), f);

    {
        static constexpr int I = 1;
        std::vector<T> u(dims[0] * (dims[1] - 1));
        forward_stencil_test_cuda<T, N, I>::transform(&v[0], dims, &u[0]);
        std::vector<T> result(u.size());
        auto r = result.begin();
        for (int i = 0; i < dims[0]; i++)
            for (int j = 0; j < dims[1] - 1; j++)
                *r++ = v[i * dims[1] + j + 1] - v[i * dims[1] + j];

        REQUIRE(u == result);
    }

    {
        static constexpr int I = 0;
        int udims[] = {dims[0], dims[1]};
        udims[I] -= 1;
        std::vector<T> u(udims[0] * udims[1]);
        forward_stencil_test_cuda<T, N, I>::transform(&v[0], dims, &u[0]);
        std::vector<T> result(u.size());
        auto r = result.begin();
        for (int i = 0; i < udims[0]; i++)
            for (int j = 0; j < udims[1]; j++)
                *r++ = v[(i + 1) * dims[1] + j] - v[i * dims[1] + j];

        REQUIRE(u == result);
    }
}

TEST_CASE("forward 3D transformation")
{
    using T = double;
    static constexpr auto N = 3;

    int dims[] = {10, 12, 15};
    std::vector<T> v(dims[0] * dims[1] * dims[2]);
    randomize();
    std::generate(v.begin(), v.end(), f);

    {
        static constexpr int I = 2;
        int udims[] = {dims[0], dims[1], dims[2]};
        udims[I] -= 1;
        std::vector<T> u(udims[0] * udims[1] * udims[2]);
        forward_stencil_test_cuda<T, N, I>::transform(&v[0], dims, &u[0]);
        std::vector<T> result(u.size());
        auto r = result.begin();
        for (int i = 0; i < udims[0]; i++)
            for (int j = 0; j < udims[1]; j++)
                for (int k = 0; k < udims[2]; k++)
                    *r++ = v[dims[2] * (i * dims[1] + j) + k + 1] -
                           v[dims[2] * (i * dims[1] + j) + k];

        REQUIRE(u == result);
    }

    {
        static constexpr int I = 1;
        int udims[] = {dims[0], dims[1], dims[2]};
        udims[I] -= 1;
        std::vector<T> u(udims[0] * udims[1] * udims[2]);
        forward_stencil_test_cuda<T, N, I>::transform(&v[0], dims, &u[0]);
        std::vector<T> result(u.size());
        auto r = result.begin();
        for (int i = 0; i < udims[0]; i++)
            for (int j = 0; j < udims[1]; j++)
                for (int k = 0; k < udims[2]; k++)
                    *r++ = v[dims[2] * (i * dims[1] + j + 1) + k] -
                           v[dims[2] * (i * dims[1] + j) + k];

        REQUIRE(u == result);
    }

    {
        static constexpr int I = 0;
        int udims[] = {dims[0], dims[1], dims[2]};
        udims[I] -= 1;
        std::vector<T> u(udims[0] * udims[1] * udims[2]);
        forward_stencil_test_cuda<T, N, I>::transform(&v[0], dims, &u[0]);
        std::vector<T> result(u.size());
        auto r = result.begin();
        for (int i = 0; i < udims[0]; i++)
            for (int j = 0; j < udims[1]; j++)
                for (int k = 0; k < udims[2]; k++)
                    *r++ = v[dims[2] * ((i + 1) * dims[1] + j) + k] -
                           v[dims[2] * (i * dims[1] + j) + k];

        REQUIRE(u == result);
    }
}
