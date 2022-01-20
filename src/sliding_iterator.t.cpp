#include <catch2/catch_test_macros.hpp>

#include "sliding_iterator_test.hpp"
#include <vector>

TEST_CASE("sliding_iterator basic") {
    using T = double;
    std::vector<T> v(10);
    sliding_test_cuda<T>::init(&v[0], v.size(), 2);
}
