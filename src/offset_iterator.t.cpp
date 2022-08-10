#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include "offset_iterator_test.hpp"
#include <vector>

#include "md_device_vector.hpp"

#include <algorithm>

using Catch::Matchers::Approx;
using T = double;
using vec = std::vector<T>;

TEST_CASE("init 1d")
{
    const int dims[] = {10};
    int noff = 3;

    auto u = make_md_vec<T>(Ib{1, dims[0]});
    auto o = make_offset_vec(noff, 1);

    for (int i = 0; i < noff; i++) o.set_offset(i, i + 2);

    offset_test_cuda<T, 1>::init(u.data(), noff, o.data(), dims);

    std::vector<T> result = u;

    REQUIRE(result[2] == -1);
    REQUIRE(result[3] == -2);
    REQUIRE(result[4] == -3);
}

TEST_CASE("init 2d")
{
    const int dims[] = {7, 10};
    int noff = 3;

    auto u = make_md_vec<T>(Jb{0, dims[0] - 1}, Ib{0, dims[1] - 1});
    auto o = make_offset_vec(noff, 2);

    for (int i = 0; i < noff; i++) o.set_offset(i, i + 3, i + 2);

    offset_test_cuda<T, 2>::init(u.data(), noff, o.data(), dims);

    std::vector<T> result = u;

    REQUIRE(result[u.index(3, 2)] == -1);
    REQUIRE(result[u.index(4, 3)] == -2);
    REQUIRE(result[u.index(5, 4)] == -3);
}

TEST_CASE("init 3d")
{
    const int dims[] = {11, 7, 10};
    int noff = 3;

    auto u = make_md_vec<T>(Kb{0, dims[0] - 1}, Jb{0, dims[1] - 1}, Ib{0, dims[2] - 1});
    auto o = make_offset_vec(noff, 3);

    for (int i = 0; i < noff; i++) o.set_offset(i, i + 3, i + 2, i + 1);

    offset_test_cuda<T, 3>::init(u.data(), noff, o.data(), dims);

    std::vector<T> result = u;

    REQUIRE(result[u.index(3, 2, 1)] == -1);
    REQUIRE(result[u.index(4, 3, 2)] == -2);
    REQUIRE(result[u.index(5, 4, 3)] == -3);
}
