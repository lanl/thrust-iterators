#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include "coarse_to_fine_iterator_test.hpp"
#include <vector>

#include "random.hpp"
#include <algorithm>
#include <iostream>

TEST_CASE("init")
{
    using T = double;
    int ratio = 2;
    int fi0 = 2;
    int fi1 = 6;
    int ci0 = 1;
    int ci1 = 3;
    std::vector<T> fine(fi1 - fi0 + 1, -1);
    std::vector<T> coarse{1.0, 2.0, 3.0};

    test<T>::init(fi0, fi1, ci0, ci1, ratio, coarse.data(), fine.data());

    std::vector<T> res {1.0, 1.0, 2.0, 2.0, 3.0};

    REQUIRE(fine == res);
}
