#include <catch2/catch_test_macros.hpp>

#include "sliding_iterator_test.hpp"
#include <vector>

TEST_CASE("sliding_iterator basic") {
    using T = double;
    std::vector<T> v(10);
    sliding_test_cuda<T>::init(&v[0], v.size(), 2);
}

TEST_CASE("sliding_iterator transformation")
{
    using T = double;
    std::vector<T> v{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<T> u(v.size() - 1);
    sliding_test_cuda<T>::transform(&v[0], v.size(), &u[0], u.size(), 2);

    std::vector<T> result{1, 3, 5, 7, 9, 11, 13, 15, 17};
    REQUIRE(u == result);

}
