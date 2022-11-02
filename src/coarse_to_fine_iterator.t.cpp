// Copyright (c) 2022. Triad National Security, LLC. All rights reserved.
// This program was produced under U.S. Government contract
// 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is
// operated by Triad National Security, LLC for the U.S. Department of
// Energy/National Nuclear Security Administration. All rights in the
// program are reserved by Triad National Security, LLC, and the
// U.S. Department of Energy/National Nuclear Security
// Administration. The Government is granted for itself and others acting
// on its behalf a nonexclusive, paid-up, irrevocable worldwide license
// in this material to reproduce, prepare derivative works, distribute
// copies to the public, perform publicly and display publicly, and to
// permit others to do so.


#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include "coarse_to_fine_iterator_test.hpp"
#include <vector>

#include <algorithm>
#include <iostream>

#include "md_device_vector.hpp"

TEST_CASE("init")
{
    using T = double;
    int ratio = 2;
    int fi0 = 2;
    int fi1 = 6;
    int ci0 = 1;
    int ci1 = 3;
    thrust::host_vector<T> fine(fi1 - fi0 + 1, -1);
    thrust::host_vector<T> coarse(3);

    coarse[0] = 1;
    coarse[1] = 2;
    coarse[2] = 3;

    thrust::device_vector<T> fd = fine;
    thrust::device_vector<T> cd = coarse;

    REQUIRE(fd.size() == 5);
    REQUIRE(cd.size() == 3);

    test<T>::init(fi0,
                  fi1,
                  ci0,
                  ci1,
                  ratio,
                  thrust::raw_pointer_cast(cd.data()),
                  thrust::raw_pointer_cast(fd.data()));

    std::vector<T> res{1.0, 1.0, 2.0, 2.0, 3.0};
    fine = fd;

    REQUIRE(fine == res);
}
