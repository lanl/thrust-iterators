#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include "window_iterator_test.hpp"
#include <vector>

#include "md_device_vector.hpp"

#include <algorithm>
#include <iostream>

using Catch::Matchers::Approx;

constexpr auto f = []() { return pick(0.0, 1.0); };

TEST_CASE("window_iterator basic")
{
    using T = double;
    auto u = make_md_vec<T>(Ib{1, 10});

    window_test_cuda<T>::init(u.data(), u.size());

    const auto& v = u.host();

    REQUIRE(v[0] == 1.0);
    REQUIRE(v[1] == 2.0);
    REQUIRE(v[5] == 3.0);
    REQUIRE(v[6] == 4.0);
}

TEST_CASE("window_iterator transform")
{
    using T = double;
    auto v = make_md_vec<T>(Ib{1, 10});
    window_test_cuda<T>::transform(v.data(), v.size());

    std::vector<T> u{1, 2, 1, 2, 1, 2, 1, 2, 1, 2};
    std::vector<T> v_h = v;

    REQUIRE(v_h == u);
}

TEST_CASE("window_iterator transform2")
{
    using T = double;
    int w = 3;
    int sz = 5;

    auto vd = make_md_vec<T>(Ib(1, sz), Wb(1, w));
    randomize();

    vd.fill_random();

    window_test_cuda<T>::transform2(vd.data(), vd.size());

    std::vector<T> v = vd;
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

    auto vd = make_md_vec<T>(Ib{1, sz}, Wb{1, w});
    auto xd = make_md_vec<T>(Ib(1, sz));
    randomize();

    vd.fill_random();
    xd.fill_random();

    window_test_cuda<T>::transform3(vd.data(), vd.size(), xd.data());

    std::vector<T> v = vd;
    const auto& x = xd.host();
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

    auto vd = make_md_vec<T>(Ib{1, sz}, Wb{1, w});
    auto xd = make_md_vec<T>(Ib(1, sz));
    randomize();

    vd.fill_random();
    xd.fill_random();

    window_test_cuda<T>::transform4(vd.data(), vd.size(), xd.data());

    std::vector<T> v = vd;
    const auto& x = xd.host();
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

    auto vd = make_md_vec<T>(Ib{1, sz}, Wb{1, w});
    auto xd = make_md_vec<T>(Ib(1, sz));
    auto rhsd = make_md_vec<T>(Ib(1, sz));
    randomize();

    vd.fill_random();
    xd.fill_random();
    rhsd.fill_random();

    std::vector<T> ans = rhsd;

    window_test_cuda<T>::rhs(rhsd.data(), vd.data(), vd.size(), xd.data());

    std::vector<T> rhs = rhsd;
    const auto& v = vd.host();
    const auto& x = xd.host();

    for (int i = 0; i < sz; i++) {
        int j = i * w;
        ans[i] -= v[j + 1] * x[i];
    }

    REQUIRE_THAT(rhs, Approx(ans));
}
