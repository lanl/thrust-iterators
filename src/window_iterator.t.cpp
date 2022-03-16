#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include "window_iterator_test.hpp"
#include <vector>

#include "random.hpp"
#include <algorithm>
#include <iostream>

using Catch::Matchers::Approx;

constexpr auto f = []() { return pick(0.0, 1.0); };

TEST_CASE("window_iterator basic")
{
    using T = double;
    std::vector<T> v(10);
    window_test_cuda<T>::init(&v[0], v.size());

    REQUIRE(v[0] == 1.0);
    REQUIRE(v[1] == 2.0);
    REQUIRE(v[5] == 3.0);
    REQUIRE(v[6] == 4.0);
}

TEST_CASE("window_iterator transform")
{
    using T = double;
    std::vector<T> v(10);
    window_test_cuda<T>::transform(&v[0], v.size());

    std::vector<T> u{1, 2, 1, 2, 1, 2, 1, 2, 1, 2};

    REQUIRE(v == u);
}

TEST_CASE("window_iterator transform2")
{
    using T = double;
    int w = 3;
    int sz = 5;

    std::vector<T> v(w * sz);
    randomize();

    std::generate(v.begin(), v.end(), f);

    window_test_cuda<T>::transform2(&v[0], v.size());

    std::vector<T> u(v.size());
    for (int i = 0; i < sz; i++) {
        int j = i * w;
        u[j + 1] = v[j + 1];
        u[j + 2] = v[j + 2];
        u[j] = -(u[j + 1] + u[j + 2]) + 3;
    }

    REQUIRE_THAT(v, Approx(u));
}

TEST_CASE("window_iterator transform3")
{
    using T = double;
    int w = 3;
    int sz = 5;

    std::vector<T> v(w * sz);
    std::vector<T> x(sz);
    randomize();

    std::generate(v.begin(), v.end(), f);
    std::generate(x.begin(), x.end(), f);

    window_test_cuda<T>::transform3(&v[0], v.size(), x.data());

    std::vector<T> u(v.size());
    for (int i = 0; i < sz; i++) {
        int j = i * w;
        u[j + 1] = v[j + 1];
        u[j + 2] = v[j + 2];
        u[j] = -(u[j + 1] + u[j + 2]) + 3 * x[i];
    }

    REQUIRE_THAT(v, Approx(u));
}

TEST_CASE("window_iterator transform4")
{
    using T = double;
    int w = 4;
    int sz = 5;

    std::vector<T> v(w * sz);
    std::vector<T> x(sz);
    randomize();

    std::generate(v.begin(), v.end(), f);
    std::generate(x.begin(), x.end(), f);

    window_test_cuda<T>::transform4(&v[0], v.size(), x.data());

    std::vector<T> u(v.size());
    for (int i = 0; i < sz; i++) {
        int j = i * w;
        u[j] = -(v[j + 2] + v[j + 3]) - 3 * x[i];
        u[j + 1] = 6 * x[i];
        u[j + 2] = v[j + 2];
        u[j + 3] = v[j + 3];
    }

    REQUIRE_THAT(v, Approx(u));
}

TEST_CASE("window_iterator rhs")
{
    using T = double;
    int w = 4;
    int sz = 5;

    std::vector<T> v(w * sz);
    std::vector<T> x(sz);
    std::vector<T> rhs(sz);
    randomize();

    std::generate(v.begin(), v.end(), f);
    std::generate(x.begin(), x.end(), f);
    std::generate(rhs.begin(), rhs.end(), f);

    std::vector<T> ans{rhs};

    window_test_cuda<T>::rhs(&rhs[0], v.data(), v.size(), x.data());

    for (int i = 0; i < sz; i++) {
        int j = i * w;
        std::cout << rhs[i] << ", " << ans[i] << ", " << (v[j + 1] * x[i]) << '\n';
    }

    for (int i = 0; i < sz; i++) {
        int j = i * w;
        ans[i] -= v[j + 1] * x[i];
    }

    REQUIRE_THAT(rhs, Approx(ans));
}
