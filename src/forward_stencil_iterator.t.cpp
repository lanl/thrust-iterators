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

    int dims[] = {3, 4};
    std::vector<T> v(dims[0] * dims[1]);
    randomize();
    std::generate(v.begin(), v.end(), f);

    {
        static constexpr int I = 1;
        std::vector<T> u(dims[0] * (dims[1] - 1));
        printf(" >>> u.size: %zu\n", u.size());
        forward_stencil_test_cuda<T, N, I>::transform(&v[0], dims, &u[0]);
        printf(" <<< u.size: %zu\n", u.size());
        std::vector<T> result(u.size());
        auto r = result.begin();
        for (int i = 0; i < dims[0]; i++)
            for (int j = 0; i < dims[1] - 1; j++)
                *r++ = v[i * dims[1] + j + 1] - v[i * dims[1] + j];

        REQUIRE(u == result);
    }
}
