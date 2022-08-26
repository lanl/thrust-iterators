#include <array>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <iostream>

#include "prototypes3d.h"
#include "random.hpp"
#include <algorithm>

#include "ls_apply_st_3d_cuda.hpp"
#include "md_device_vector.hpp"

using Catch::Matchers::Approx;

constexpr auto f = []() { return pick(0.0, 1.0); };

TEST_CASE("point")
{
    using T = double;

    const int i0 = 10, j0 = 11, k0 = 12;
    const int i1 = 20, j1 = 30, k1 = 33;
    const int fgcw = pick(1, 3);
    const int ugcw = pick(1, 3);
    const int rgcw = pick(1, 3);
    const T a = pick(0.1, 10.0);
    const T b = pick(0.1, 10.0);

    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};
    const auto k = Kb{k0, k1};

    int sz = 1;
    auto w = Wb{1, sz};
    auto offsets = make_md_vec<int>(3, w);

    auto st = make_md_vec<T>(k, j, i, w);
    auto u = make_md_vec<T>(ugcw, k, j, i);
    auto ff = make_md_vec<T>(fgcw, k, j, i);
    auto r = make_md_vec<T>(rgcw, k, j, i);

    const int ic = pick(i0, i1);
    const int jc = pick(j0, j1);
    const int kc = pick(k0, k1);

    applystencilatpoint3d_(i0,
                           i1,
                           j0,
                           j1,
                           k0,
                           k1,
                           ic,
                           jc,
                           kc,
                           sz,
                           offsets.host_data(),
                           st.host_data(),
                           a,
                           b,
                           fgcw,
                           ff.host_data(),
                           ugcw,
                           u.host_data(),
                           rgcw,
                           r.host_data());
}
