#include <catch2/catch_test_macros.hpp>

#include "window_iterator_test.hpp"
#include <vector>

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

TEST_CASE("window_iterator transform") {
    using T = double;
    std::vector<T> v(10);
    window_test_cuda<T>::transform(&v[0], v.size());

    std::vector<T> u{1, 2, 1, 2, 1, 2, 1, 2, 1, 2};
    REQUIRE(v == u);
}
