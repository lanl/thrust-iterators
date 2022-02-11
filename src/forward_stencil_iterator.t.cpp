#include <catch2/catch_test_macros.hpp>

#include "forward_stencil_test.hpp"
#include <vector>

TEST_CASE("forward basic")
{
    using T = double;
    std::vector<T> v(10);
    forward_stencil_test_cuda<T>::init(&v[0], v.size());
}

TEST_CASE("forward basic 2D")
{
    using T = double;
    int dims[] = {3, 4};

    std::vector<T> v(dims[0] * dims[1]);

    forward_stencil_test_cuda<T>::init2D(&v[0], dims);
}

TEST_CASE("forward transformation")
{
    using T = double;
    std::vector<T> v{0, 1, -2, 3, 4, -5, 6, -7, 80, 9};
    int stride = 2;
    std::vector<T> u(v.size() - stride);
    //forward_stencil_test_cuda<T>::transform(&v[0], v.size(), 2, &u[0]);

    std::vector<T> result(u.size());
    for (int i = 0; i < result.size(); i++) result[i] = v[i + stride] - v[i];

    //REQUIRE(u == result);
}
